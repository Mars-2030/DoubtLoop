"""`search(query)` over a fixed local corpus.

Fixed, local, and BM25-scored on purpose:

* fixed  - the same query returns the same passages in six months, so an
           ablation run today is comparable to one run after fine-tuning;
* local  - no network, so the evaluation cannot silently change under you;
* BM25   - transparent. When the model cites a passage you can see exactly why
           the retriever surfaced it, which matters when the claim under test
           is "did the model trust the evidence or its own prior?".

Swapping in a dense retriever means implementing `Retriever.search`; nothing
else in the loop touches the ranking.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from groundloop import config
from groundloop.schema import Passage
from groundloop.textutil import content_tokens

# OpenAI-style function schema; Qwen's chat template consumes the same shape.
SEARCH_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search the local evidence corpus for passages relevant to a query. "
            "Use it to check any factual claim before you commit to it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or a short question. Not a full sentence of prose.",
                },
                "k": {
                    "type": "integer",
                    "description": f"How many passages to return (default {config.TOP_K}, max 10).",
                },
            },
            "required": ["query"],
        },
    },
}


def load_corpus(path: str | Path | None = None) -> list[Passage]:
    path = Path(path or config.CORPUS_PATH)
    passages = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            passages.append(
                Passage(
                    id=d["id"],
                    title=d.get("title", ""),
                    text=d["text"],
                    world=d.get("world", "real"),
                    tags=list(d.get("tags", [])),
                )
            )
    if not passages:
        raise ValueError(f"empty corpus at {path}")
    return passages


class SearchTool:
    """BM25 retrieval over a fixed corpus, plus a JSON tool-call interface."""

    name = "search"
    spec = SEARCH_TOOL_SPEC

    def __init__(self, passages: list[Passage] | None = None, k1: float | None = None, b: float | None = None):
        self.passages = passages if passages is not None else load_corpus()
        self.k1 = config.BM25_K1 if k1 is None else k1
        self.b = config.BM25_B if b is None else b
        self.by_id = {p.id: p for p in self.passages}
        self._index()
        self.call_log: list[dict] = []

    # -- indexing ----------------------------------------------------------

    def _index(self) -> None:
        self._docs = [content_tokens(f"{p.title} {p.text}") for p in self.passages]
        self._lens = [len(d) for d in self._docs]
        self._avglen = sum(self._lens) / max(len(self._lens), 1)
        self._tf = [Counter(d) for d in self._docs]
        df: Counter[str] = Counter()
        for d in self._docs:
            df.update(set(d))
        n = len(self._docs)
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    # -- retrieval ---------------------------------------------------------

    def score(self, query: str, doc_idx: int) -> float:
        tf, dl = self._tf[doc_idx], self._lens[doc_idx]
        total = 0.0
        for term in content_tokens(query):
            f = tf.get(term, 0)
            if not f:
                continue
            idf = self._idf.get(term, 0.0)
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(self._avglen, 1e-9))
            total += idf * (f * (self.k1 + 1)) / denom
        return total

    def search(self, query: str, k: int | None = None) -> list[Passage]:
        k = config.TOP_K if k is None else max(1, min(int(k), 10))
        scored = [(self.score(query, i), i) for i in range(len(self.passages))]
        # Sort by score, then by id, so ties are broken deterministically.
        scored.sort(key=lambda t: (-t[0], self.passages[t[1]].id))
        out = []
        for s, i in scored[:k]:
            if s <= 0:
                continue
            p = self.passages[i]
            out.append(Passage(p.id, p.title, p.text, p.world, list(p.tags), score=round(s, 4)))
        return out

    # -- tool-call interface ----------------------------------------------

    def call(self, arguments: dict) -> dict:
        """Execute a parsed tool call. Returns the JSON payload for the model."""
        query = str(arguments.get("query", "")).strip()
        k = arguments.get("k", config.TOP_K)
        if not query:
            payload = {"error": "search requires a non-empty 'query' argument", "results": []}
            self.call_log.append({"query": "", "k": k, "n_results": 0, "ok": False})
            return payload
        hits = self.search(query, k)
        self.call_log.append({"query": query, "k": k, "n_results": len(hits), "ok": True})
        return {
            "query": query,
            "results": [
                {"id": p.id, "title": p.title, "text": p.text, "score": p.score} for p in hits
            ],
        }

    def render(self, passages: list[Passage]) -> str:
        """Evidence block as the model sees it in the critique prompt."""
        if not passages:
            return "(no passages retrieved)"
        return "\n\n".join(f"[{p.id}] {p.title}\n{p.text}" for p in passages)


# --- tool-call parsing -----------------------------------------------------

_QWEN_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_FUNC_STYLE = re.compile(r'search\(\s*(?:query\s*=\s*)?["\'](.+?)["\']\s*\)')


def _balanced_objects(text: str):
    """Yield every balanced ``{...}`` span in `text`.

    A regex cannot do this: the arguments object nests inside the call object,
    and any non-greedy pattern stops at the inner closing brace and hands back
    JSON that will not parse.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start : i + 1]
                start = -1
            elif depth < 0:
                depth = 0


def parse_tool_calls(text: str) -> list[dict]:
    """Recover `search` calls from raw model text.

    Small instruct models emit tool calls in whatever shape their template
    nudges them toward and in a few shapes it does not. Accepting all of them
    keeps tool-use *format* failures out of the tool-use *quality* metric.
    """
    calls: list[dict] = []
    for m in _QWEN_BLOCK.finditer(text):
        try:
            d = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        args = d.get("arguments", d.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"query": args}
        calls.append({"name": d.get("name", "search"), "arguments": args or {}})
    if calls:
        return calls
    for blob in _balanced_objects(text):
        try:
            d = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict) or "name" not in d:
            continue
        args = d.get("arguments", d.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"query": args}
        calls.append({"name": d.get("name", "search"), "arguments": args or {}})
    if calls:
        return calls
    for m in _FUNC_STYLE.finditer(text):
        calls.append({"name": "search", "arguments": {"query": m.group(1)}})
    return calls


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Query the GroundLoop evidence corpus.")
    ap.add_argument("query", nargs="+")
    ap.add_argument("-k", type=int, default=config.TOP_K)
    args = ap.parse_args(argv)
    tool = SearchTool()
    hits = tool.search(" ".join(args.query), args.k)
    if not hits:
        print("(no results)")
        return 0
    for p in hits:
        print(f"{p.score:>7.3f}  [{p.id}] {p.title}\n         {p.text[:160]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
