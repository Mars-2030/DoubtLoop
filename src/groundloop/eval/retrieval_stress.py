"""What happens to GroundLoop when retrieval is bad?

The headline ablation runs on a corpus where BM25 finds a gold passage ~97% of
the time. That isolates the variable under test - does the model *use* evidence -
and says nothing at all about what the model does when the evidence in front of
it is wrong, missing, or padded with noise. This module supplies that second
number, because the two failure modes are opposite in kind:

* a model that ignores evidence is unfaithful in the normal way;
* a model that treats *whatever it retrieved* as truth is unfaithful in a worse
  way, because retrieval failures now launder into confident cited answers.

Three degradations, all applied after the model issues its own query, so the
model cannot see them coming:

``normal``       untouched - the baseline to compare against.
``no_gold``      every gold passage is removed from the results. The question is
                 answerable, the evidence is not. The only faithful answer is to
                 say so, and ``faithful_abstention_rate`` is the number that
                 matters.
``distractor``   gold is kept, but padded with passages from the other world
                 slice - guaranteed irrelevant, guaranteed on-topic-looking.
                 Measures distractibility.
``empty``        search returns nothing at all. Tests the degenerate path: does
                 the model abstain, or fall back on its prior and cite nothing?

Only the GroundLoop condition retrieves, so this is a within-condition
comparison rather than a three-way one.
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from groundloop import config
from groundloop.datasets import load_qa, write_jsonl
from groundloop.eval.metrics import _pct, reference_evidence, score_answer
from groundloop.llm import add_backend_args, backend_from_args
from groundloop.loop.pipeline import run_condition
from groundloop.schema import Passage
from groundloop.tools.search import SearchTool

MODES = ("normal", "no_gold", "distractor", "empty")


class StressTool:
    """A `SearchTool` wrapper that degrades results after the query is issued.

    Wrapping rather than subclassing keeps the BM25 index and the ranking
    untouched: the *retriever* is not what is being changed here, only what
    comes back from it. `gold` is set by the runner before each example, which
    is the one piece of state this needs and the reason it is not a pure
    function.
    """

    def __init__(self, base: SearchTool, mode: str = "normal", n_distractors: int = 2,
                 seed: int = config.SEED):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
        self.base = base
        self.mode = mode
        self.n_distractors = n_distractors
        self.rng = random.Random(seed)
        self.gold: set[str] = set()
        self.by_id = base.by_id
        self.call_log = base.call_log

    def render(self, passages):
        return self.base.render(passages)

    def search(self, query: str, k: int | None = None):
        return self._degrade(self.base.search(query, k))

    def call(self, arguments: dict) -> dict:
        payload = self.base.call(arguments)
        results = payload.get("results", [])
        kept = self._degrade([
            Passage(r["id"], r["title"], r["text"], self.by_id[r["id"]].world, score=r.get("score", 0.0))
            for r in results
        ])
        payload["results"] = [
            {"id": p.id, "title": p.title, "text": p.text, "score": p.score} for p in kept
        ]
        return payload

    def _degrade(self, passages: list[Passage]) -> list[Passage]:
        if self.mode == "normal":
            return passages
        if self.mode == "empty":
            return []
        if self.mode == "no_gold":
            return [p for p in passages if p.id not in self.gold]
        # distractor: keep everything, pad with passages from the other world
        worlds = {p.world for p in passages} or {"real"}
        other = [
            p for p in self.base.passages
            if p.world not in worlds and p.id not in {q.id for q in passages}
        ]
        self.rng.shuffle(other)
        padded = list(passages) + other[: self.n_distractors]
        self.rng.shuffle(padded)
        return padded


def run_mode(llm, base_tool: SearchTool, examples: list[dict], mode: str, k: int,
             progress=None) -> tuple[list[dict], dict]:
    stress = StressTool(base_tool, mode)
    trajectories, scores = [], []
    for i, ex in enumerate(examples, 1):
        stress.gold = set(ex.get("support", []) or [])
        traj = run_condition(llm, ex, "groundloop", tool=stress, k=k)
        trajectories.append(traj.to_json())
        # Scored against the honest reference pool, not against the degraded
        # evidence: the question of interest is what the model *said*, given
        # that what it saw could not support an answer.
        scores.append(score_answer(ex, traj.final, reference_evidence(ex, base_tool), "groundloop"))
        if progress:
            progress(i, len(examples))

    answerable = [s for s in scores if not s.expect_abstention]
    n = len(answerable)
    return trajectories, {
        "mode": mode,
        "n_answerable": n,
        # Under no_gold/empty, abstaining is the correct behaviour on every
        # answerable item, so this rate should go UP as retrieval degrades.
        "faithful_abstention_rate": _pct(sum(s.abstained for s in answerable), n),
        # ...and this one should not go up with it. An answer that stays
        # confident while its evidence disappears is the failure this measures.
        "fabrication_rate": _pct(sum(s.hallucinated and not s.abstained for s in answerable), n),
        "accuracy_answerable": _pct(sum(s.correct for s in answerable), n),
        "hallucination_rate": _pct(sum(s.hallucinated for s in scores), len(scores)),
    }


def render_markdown(rows: list[dict], meta: dict) -> str:
    lines = [
        "# Retrieval stress",
        "",
        f"- generated: {meta['date']}  |  backend: `{meta['backend']}`  |  model: `{meta['model']}`",
        f"- n = {meta['n']} QA items, GroundLoop condition only",
        "",
        "| Retrieval | Faithful abstention ↑ | Fabrication ↓ | Accuracy | Hallucination rate |",
        "|---|---|---|---|---|",
    ]
    labels = {
        "normal": "normal (unmodified)",
        "no_gold": "gold passage removed",
        "distractor": "gold + irrelevant padding",
        "empty": "search returns nothing",
    }
    for row in rows:
        lines.append(
            f"| {labels.get(row['mode'], row['mode'])} | {row['faithful_abstention_rate']}% | "
            f"{row['fabrication_rate']}% | {row['accuracy_answerable']}% | {row['hallucination_rate']}% |"
        )
    lines += [
        "",
        "Rates are over the answerable items only, except the last column.",
        "",
        "## How to read it",
        "",
        "In the `gold passage removed` and `search returns nothing` rows the",
        "question is still answerable but the evidence the model saw is not.",
        "Faithful abstention should rise toward 100% and fabrication should stay",
        "near zero. A model whose fabrication rate climbs as its evidence",
        "disappears has not learned to check a source; it has learned to cite",
        "whatever it was handed, which is worse than not retrieving at all -",
        "retrieval failures now arrive wearing citations.",
        "",
        "The `gold + irrelevant padding` row is the milder question: does",
        "on-topic-looking noise pull the answer off the passage that actually",
        "answers it?",
        "",
        "Regenerate with `python -m groundloop.eval.retrieval_stress`.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from datetime import date
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", default=",".join(MODES), help=f"comma-separated: {', '.join(MODES)}")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("-k", type=int, default=config.TOP_K)
    ap.add_argument("--out-dir", default=None)
    # On by default: a metric change means re-scoring, and re-scoring needs the
    # generations. Regenerating them costs GPU minutes; keeping them costs a file.
    ap.add_argument("--save-trajectories", action="store_true", default=True)
    ap.add_argument("--no-save-trajectories", dest="save_trajectories", action="store_false")
    ap.add_argument("--json", action="store_true")
    add_backend_args(ap)
    args = ap.parse_args(argv)

    llm = backend_from_args(args)
    tool = SearchTool()
    examples = load_qa(limit=args.limit)
    out_dir = Path(args.out_dir) if args.out_dir else config.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        trajs, summary = run_mode(
            llm, tool, examples, mode, args.k,
            progress=lambda i, n, m=mode: print(f"\r  {m}: {i}/{n}   ", end="", file=sys.stderr, flush=True),
        )
        rows.append(summary)
        if args.save_trajectories:
            write_jsonl(out_dir / "raw" / f"stress_{mode}.jsonl", trajs)
    print(file=sys.stderr)

    meta = {
        "date": date.today().isoformat(),
        "backend": args.backend,
        "model": args.model or ("n/a (scripted stand-in)" if args.backend == "scripted" else "default"),
        "n": len(examples),
    }
    if args.json:
        print(json.dumps({"meta": meta, "modes": rows}, indent=2))
    else:
        table = render_markdown(rows, meta)
        (out_dir / "retrieval_stress.md").write_text(table, encoding="utf-8")
        print(table)
        print(f"wrote {out_dir / 'retrieval_stress.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
