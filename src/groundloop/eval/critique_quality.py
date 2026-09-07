"""Does the model's self-critique agree with the evidence in front of it?

This is the measurement the project actually turns on. GroundLoop assumes that
showing a model the right passage lets it catch its own errors; if the critique
step endorses whatever the draft said, retrieval and revision are both wasted
and the method reduces to an expensive way of restating the first answer.

The number to watch is **false approval**: the model marked a claim
"supported", naming a passage, where the harness finds no support for it. A
worked example from a real run - the draft said "During Apollo 11, the crew of
the spacecraft stayed in lunar orbit instead of walking on the surface", and the
critique returned `{"status": "supported", "passage": "r01"}` for it, where r01
plainly says Armstrong and Aldrin walked while Collins stayed.

**This is agreement, not accuracy.** The reference here is the lexical support
check, which has its own false negatives - four were found and fixed in two days
of real-model runs, every one of them rejecting a correct claim. So treat
`false_approval_rate` as an upper bound on the model's error and
`false_alarm_rate` as a lower bound. Where the two disagree, read the claims
(`--show N`) rather than the rate.
"""

from __future__ import annotations

import argparse
import json
import sys

from groundloop import config
from groundloop.datasets import load_qa, read_jsonl
from groundloop.eval.metrics import _pct
from groundloop.llm import add_backend_args, backend_from_args
from groundloop.loop.pipeline import run_dataset
from groundloop.textutil import coverage
from groundloop.tools.search import SearchTool

# How much of a harness claim a model claim must cover to be the same claim.
PAIR_FLOOR = 0.6


def pair_claims(model_claims: list[dict], harness_claims: list[dict]):
    """Match the model's judgements to the harness's, claim by claim.

    Both lists describe the same draft, but the model quotes claims in its own
    words and the harness splits them its own way, so they are paired by
    overlap rather than by index.
    """
    pairs = []
    for harness in harness_claims:
        best, best_cov = None, 0.0
        for model in model_claims:
            cov = coverage(harness.get("text", ""), model.get("text", ""))
            if cov > best_cov:
                best, best_cov = model, cov
        pairs.append((harness, best if best_cov >= PAIR_FLOOR else None))
    return pairs


def score_trajectories(trajectories: list[dict]) -> tuple[list[dict], dict]:
    disagreements = []
    paired = agreed = false_approval = false_alarm = 0
    unpaired = 0
    n_examples = blind = with_errors = 0

    for traj in trajectories:
        critique = traj.get("critique") or {}
        harness_claims = critique.get("harness_claims") or []
        model_claims = critique.get("claims") or []
        if not harness_claims:
            continue
        n_examples += 1
        harness_bad = [c for c in harness_claims if not c.get("supported")]
        if harness_bad:
            with_errors += 1
            if all(c.get("supported") for c in model_claims) and model_claims:
                blind += 1

        for harness, model in pair_claims(model_claims, harness_claims):
            if model is None:
                unpaired += 1
                continue
            paired += 1
            if harness.get("supported") == model.get("supported"):
                agreed += 1
                continue
            if model.get("supported"):
                false_approval += 1
                kind = "model approved what the evidence does not support"
            else:
                false_alarm += 1
                kind = "model rejected what the evidence does support"
            disagreements.append({
                "example_id": traj.get("example_id", ""),
                "kind": kind,
                "claim": harness.get("text", ""),
                "model_passage": model.get("best_passage", ""),
                "harness_reason": harness.get("reason", ""),
                "harness_coverage": harness.get("coverage", 0.0),
            })

    return disagreements, {
        "n_examples": n_examples,
        "claims_paired": paired,
        "claims_unpaired": unpaired,
        "agreement_rate": _pct(agreed, paired),
        "false_approval_rate": _pct(false_approval, paired),
        "false_alarm_rate": _pct(false_alarm, paired),
        # The headline: the draft had a problem the evidence exposes, and the
        # critique waved the whole thing through.
        "blind_critique_rate": _pct(blind, with_errors),
        "examples_with_errors": with_errors,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("-k", type=int, default=config.TOP_K)
    ap.add_argument("--trajectories", default=None, help="score a saved groundloop run")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--show", type=int, default=0, help="print N disagreements in full")
    add_backend_args(ap)
    args = ap.parse_args(argv)

    if args.trajectories:
        trajectories = read_jsonl(args.trajectories)
    else:
        examples = load_qa(limit=args.limit)
        trajs = run_dataset(
            backend_from_args(args), examples, "groundloop", tool=SearchTool(), k=args.k,
            progress=lambda i, n: print(f"\r  critique: {i}/{n}", end="", file=sys.stderr, flush=True),
        )
        print(file=sys.stderr)
        trajectories = [t.to_json() for t in trajs]

    disagreements, summary = score_trajectories(trajectories)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("\nself-critique vs the evidence (GroundLoop condition)")
        for key, val in summary.items():
            unit = "%" if key.endswith("rate") else ""
            print(f"  {key:<24} {val}{unit}")
        print("\n  Agreement with the lexical check, not with ground truth:")
        print("  false_approval_rate is an upper bound on the model's error.")

    for row in disagreements[: args.show]:
        print(f"\n  [{row['example_id']}] {row['kind']}")
        print(f"    claim:   {row['claim'][:150]}")
        print(f"    model:   supported by [{row['model_passage']}]" if "approved" in row["kind"]
              else f"    model:   marked unsupported")
        print(f"    harness: {row['harness_reason']} (coverage {row['harness_coverage']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
