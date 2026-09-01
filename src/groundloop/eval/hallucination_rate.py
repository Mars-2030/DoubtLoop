"""Headline metric: how often does the final answer assert something the
evidence does not support?

Run it per condition:

    python -m groundloop.eval.hallucination_rate --condition base
    python -m groundloop.eval.hallucination_rate --condition plain_critique
    python -m groundloop.eval.hallucination_rate --condition groundloop

or score a trajectory log that already exists (`--trajectories run.jsonl`),
which is how you re-score an expensive run after changing a threshold.
"""

from __future__ import annotations

import argparse
import json
import sys

from groundloop import config
from groundloop.datasets import load_qa, read_jsonl, write_jsonl
from groundloop.eval.metrics import aggregate, reference_evidence, score_answer
from groundloop.llm import add_backend_args, backend_from_args
from groundloop.loop.pipeline import CONDITIONS, run_dataset
from groundloop.schema import Passage
from groundloop.tools.search import SearchTool


def score_trajectories(trajectories: list[dict], examples: list[dict], tool: SearchTool,
                       judge=None) -> tuple[list, dict]:
    by_id = {e["id"]: e for e in examples}
    scores = []
    for traj in trajectories:
        ex = by_id.get(traj["example_id"])
        if ex is None:
            continue
        answer = traj.get("revision") or traj.get("draft") or ""
        evidence = reference_evidence(ex, tool)
        if judge is not None:
            scores.append(judge.score(ex, answer, evidence, traj.get("condition", "")))
        else:
            scores.append(score_answer(ex, answer, evidence, traj.get("condition", "")))
    return scores, aggregate(scores)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", default="groundloop", choices=list(CONDITIONS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("-k", type=int, default=config.TOP_K)
    ap.add_argument("--trajectories", default=None, help="score an existing JSONL log instead of running")
    ap.add_argument("--save-trajectories", default=None)
    ap.add_argument("--judge", action="store_true", help="use an LLM judge instead of the lexical check")
    ap.add_argument("--json", action="store_true", help="print metrics as JSON")
    ap.add_argument("--show-unsupported", type=int, default=0, help="print N unsupported claims")
    add_backend_args(ap)
    args = ap.parse_args(argv)

    examples = load_qa(limit=args.limit)
    tool = SearchTool()

    if args.trajectories:
        trajectories = read_jsonl(args.trajectories)
    else:
        llm = backend_from_args(args)
        trajs = run_dataset(
            llm, examples, args.condition, tool=tool, k=args.k,
            progress=lambda i, n: print(f"\r  {args.condition}: {i}/{n}", end="", file=sys.stderr, flush=True),
        )
        print(file=sys.stderr)
        trajectories = [t.to_json() for t in trajs]
        if args.save_trajectories:
            write_jsonl(args.save_trajectories, trajectories)

    judge = None
    if args.judge:
        from groundloop.eval.judge import LLMJudge

        judge = LLMJudge(backend_from_args(args))

    scores, summary = score_trajectories(trajectories, examples, tool, judge=judge)
    summary["condition"] = args.condition
    summary["scorer"] = "llm_judge" if judge else "lexical"

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"\ncondition: {summary['condition']}  (scorer: {summary['scorer']})")
        for key in ("n", "hallucination_rate", "unsupported_claim_rate", "accuracy",
                    "accuracy_answerable", "correct_abstention_rate", "over_abstention_rate",
                    "citation_rate", "claims_per_answer"):
            if key in summary:
                val = summary[key]
                unit = "%" if key.endswith("rate") or key.startswith("accuracy") else ""
                print(f"  {key:<26} {val}{unit}")

    if args.show_unsupported:
        shown = 0
        print("\nunsupported claims:")
        for s in scores:
            for claim in s.unsupported_claims:
                print(f"  [{s.example_id}] {claim}")
                shown += 1
                if shown >= args.show_unsupported:
                    return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
