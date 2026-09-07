"""The three experimental conditions, run over a dataset.

    base            question -> draft                       (no critique at all)
    plain_critique  question -> draft -> critique -> revise  (constitution only)
    groundloop      question -> draft -> claims -> search    (constitution + evidence)
                             -> critique -> revise

The point of running all three through one function is that any difference in
the numbers has to come from the condition and not from three slightly
different harnesses.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Callable

from groundloop import config, prompts
from groundloop.datasets import load_qa, load_sycophancy, write_jsonl
from groundloop.llm import LLM, add_backend_args, backend_from_args, get_backend
from groundloop.loop import critique as critique_mod
from groundloop.loop.claims import build_query, extract_claims_model
from groundloop.loop.draft import draft as draft_stage
from groundloop.loop.revise import revise as revise_stage
from groundloop.schema import Passage, ToolCall, Trajectory
from groundloop.tools.search import SEARCH_TOOL_SPEC, SearchTool, parse_tool_calls

CONDITIONS = ("base", "plain_critique", "groundloop")


def _question_of(example: dict[str, Any]) -> str:
    return example.get("question") or example.get("prompt") or ""


def _research(llm: LLM, tool: SearchTool, question: str, drafted: str, k: int) -> tuple[list[ToolCall], list[Passage], str]:
    """Let the model issue the search call; fall back to a harness query.

    The fallback is logged rather than hidden: `tool_use_quality` reports how
    often it fired, which is exactly the "did the model actually learn to call
    the tool" number.
    """
    claims = extract_claims_model(llm, question, drafted)
    raw = llm.generate(
        prompts.tool_messages(question, drafted),
        stage="tool",
        ctx={"question": question, "draft": drafted, "claims": claims},
        tools=[SEARCH_TOOL_SPEC],
        max_new_tokens=128,
    )
    parsed = parse_tool_calls(raw)
    calls: list[ToolCall] = []
    evidence: list[Passage] = []
    query = ""

    for call in parsed[:1]:  # one search per round
        args = call.get("arguments") or {}
        name = call.get("name", "")
        if name != "search":
            calls.append(ToolCall(name=name, arguments=args, raw=raw, ok=False,
                                  error=f"unknown tool {name!r}"))
            continue
        query = str(args.get("query", "")).strip()
        if not query:
            calls.append(ToolCall(name="search", arguments=args, raw=raw, ok=False,
                                  error="empty query"))
            continue
        result = tool.call({"query": query, "k": k})
        calls.append(ToolCall(name="search", arguments={"query": query, "k": k}, raw=raw))
        evidence = [
            Passage(r["id"], r["title"], r["text"], tool.by_id[r["id"]].world, score=r.get("score", 0.0))
            for r in result.get("results", [])
        ]

    if not evidence:
        query = build_query(question, claims)
        result = tool.call({"query": query, "k": k})
        evidence = [
            Passage(r["id"], r["title"], r["text"], tool.by_id[r["id"]].world, score=r.get("score", 0.0))
            for r in result.get("results", [])
        ]
        calls.append(ToolCall(name="search", arguments={"query": query, "k": k}, raw=raw,
                              ok=False, error="harness fallback: model produced no usable call"))
    return calls, evidence, query


def run_condition(
    llm: LLM,
    example: dict[str, Any],
    condition: str = "groundloop",
    tool: SearchTool | None = None,
    k: int = config.TOP_K,
    rounds: int = 1,
) -> Trajectory:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; choose from {', '.join(CONDITIONS)}")
    question = _question_of(example)
    traj = Trajectory(example_id=example.get("id", ""), condition=condition, question=question)
    started = time.time()

    traj.draft = draft_stage(llm, question)

    if condition == "base":
        traj.meta["elapsed_s"] = round(time.time() - started, 3)
        return traj

    grounded = condition == "groundloop"
    # Two very different reasons a final answer can equal the draft: the critique
    # found nothing to fix, so no revision was ever attempted; or one was
    # attempted and came back unchanged (or empty, which falls back to the
    # draft). The first is a critique failure, the second a generation failure,
    # and they need opposite responses - so record which happened.
    traj.meta["revision_attempted"] = False
    current = traj.draft
    for _ in range(max(1, rounds)):
        evidence: list[Passage] = []
        evidence_block = ""
        query = ""
        if grounded:
            if tool is None:
                tool = SearchTool()
            calls, evidence, query = _research(llm, tool, question, current, k)
            traj.tool_calls.extend(calls)
            traj.evidence = evidence
            evidence_block = tool.render(evidence)

        crit = critique_mod.critique(
            llm, question, current,
            evidence=evidence, query=query, evidence_block=evidence_block, grounded=grounded,
        )
        traj.critique = crit
        if crit.verdict == "ok" and not crit.unsupported:
            traj.revision = current
            break
        traj.meta["revision_attempted"] = True
        current = revise_stage(
            llm, question, current, critique_mod.render(crit),
            evidence=evidence, evidence_block=evidence_block, grounded=grounded,
        )
        traj.revision = current

    traj.meta["elapsed_s"] = round(time.time() - started, 3)
    traj.meta["gold_support"] = example.get("support", [])
    if traj.critique is not None:
        traj.meta["critique_verdict"] = traj.critique.verdict
        traj.meta["critique_issues"] = len(traj.critique.unsupported)
        traj.meta["critique_chars"] = len(traj.critique.raw.strip())
        traj.meta["critique_notes"] = list(traj.critique.notes)
    return traj


def run_dataset(
    llm: LLM,
    examples: list[dict[str, Any]],
    condition: str,
    tool: SearchTool | None = None,
    k: int = config.TOP_K,
    rounds: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> list[Trajectory]:
    if tool is None and condition == "groundloop":
        tool = SearchTool()
    out = []
    for i, ex in enumerate(examples, 1):
        out.append(run_condition(llm, ex, condition, tool=tool, k=k, rounds=rounds))
        if progress:
            progress(i, len(examples))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one condition over the QA set and log trajectories.")
    ap.add_argument("--condition", default="groundloop", choices=list(CONDITIONS))
    ap.add_argument("--dataset", default="qa", choices=("qa", "sycophancy"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("-k", type=int, default=config.TOP_K, help="passages per search")
    ap.add_argument("--rounds", type=int, default=1, help="critique/revise rounds")
    ap.add_argument("--out", default=None, help="write trajectories to this JSONL path")
    ap.add_argument("--quiet", action="store_true")
    add_backend_args(ap)
    args = ap.parse_args(argv)

    llm = backend_from_args(args)
    examples = load_qa(limit=args.limit) if args.dataset == "qa" else load_sycophancy(limit=args.limit)

    def tick(i, n):
        if not args.quiet:
            print(f"\r  {args.condition}: {i}/{n}", end="", file=sys.stderr, flush=True)

    trajs = run_dataset(llm, examples, args.condition, k=args.k, rounds=args.rounds, progress=tick)
    if not args.quiet:
        print(file=sys.stderr)

    rows = [t.to_json() for t in trajs]
    if args.out:
        write_jsonl(args.out, rows)
        print(f"wrote {len(rows)} trajectories to {args.out}")
    else:
        for t in trajs:
            print(f"\n--- {t.example_id} [{t.condition}] ---")
            print(f"Q: {t.question}")
            print(f"draft:    {t.draft}")
            if t.evidence:
                print(f"evidence: {', '.join(p.id for p in t.evidence)}")
            if t.revision:
                print(f"final:    {t.revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
