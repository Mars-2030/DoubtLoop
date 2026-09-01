"""Did the model actually use the tool, and did the tool help?

Faithfulness gains are only attributable to retrieval if retrieval happened.
This module separates four things that a single "it worked" number would blur:

* **call quality**  - did the model emit a well-formed `search` call at all, or
  did the harness have to build the query for it? (`harness_fallback_rate` is
  the honest denominator for any claim that a fine-tune taught tool use.)
* **retrieval quality** - did the query surface the gold passages? A perfect
  call with a useless query is not tool use, it is ritual.
* **evidence usage** - did the final answer cite what came back?
* **grounding in what was actually retrieved** - the answer's claims scored
  against the model's own evidence, not the oracle pool.
"""

from __future__ import annotations

import argparse
import json
import sys

from groundloop import config
from groundloop.datasets import load_qa, read_jsonl, write_jsonl
from groundloop.eval.metrics import _pct, cited_ids
from groundloop.llm import add_backend_args, backend_from_args
from groundloop.loop.pipeline import run_dataset
from groundloop.schema import Passage
from groundloop.textutil import check_claim_default, extract_claims
from groundloop.tools.search import SearchTool


def score_trajectories(trajectories: list[dict], examples: list[dict]) -> dict:
    by_id = {e["id"]: e for e in examples}
    n = 0
    model_calls = fallbacks = no_call = 0
    total_calls = 0
    gold_hit = gold_total = gold_found = 0
    with_gold = 0
    cited = 0
    claims = unsupported = 0
    empty_retrieval = 0

    for traj in trajectories:
        ex = by_id.get(traj["example_id"])
        if ex is None:
            continue
        n += 1
        calls = traj.get("tool_calls", []) or []
        total_calls += len(calls)
        good = [c for c in calls if c.get("ok")]
        bad = [c for c in calls if not c.get("ok")]
        if good:
            model_calls += 1
        if bad:
            fallbacks += 1
        if not calls:
            no_call += 1

        retrieved = traj.get("evidence", []) or []
        if not retrieved:
            empty_retrieval += 1
        retrieved_ids = {p["id"] for p in retrieved}
        gold = set(ex.get("support", []) or [])
        if gold:
            gold_total += len(gold)
            gold_found += len(gold & retrieved_ids)
            with_gold += 1
            if gold & retrieved_ids:
                gold_hit += 1

        answer = traj.get("revision") or traj.get("draft") or ""
        if retrieved_ids & cited_ids(answer):
            cited += 1
        passages = [Passage(p["id"], p.get("title", ""), p["text"]) for p in retrieved]
        for claim in extract_claims(answer):
            claims += 1
            if not check_claim_default(claim, passages)[0]:
                unsupported += 1

    return {
        "n": n,
        "calls_per_example": round(total_calls / n, 2) if n else 0.0,
        "well_formed_call_rate": _pct(model_calls, n),
        "harness_fallback_rate": _pct(fallbacks, n),
        "no_call_rate": _pct(no_call, n),
        "empty_retrieval_rate": _pct(empty_retrieval, n),
        "gold_passage_hit_rate": _pct(gold_hit, with_gold),
        "gold_passage_recall": _pct(gold_found, gold_total),
        "evidence_citation_rate": _pct(cited, n),
        "grounded_in_retrieved_rate": _pct(claims - unsupported, claims),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("-k", type=int, default=config.TOP_K)
    ap.add_argument("--trajectories", default=None)
    ap.add_argument("--save-trajectories", default=None)
    ap.add_argument("--json", action="store_true")
    add_backend_args(ap)
    args = ap.parse_args(argv)

    examples = load_qa(limit=args.limit)
    if args.trajectories:
        trajectories = read_jsonl(args.trajectories)
    else:
        llm = backend_from_args(args)
        trajs = run_dataset(
            llm, examples, "groundloop", tool=SearchTool(), k=args.k,
            progress=lambda i, n: print(f"\r  tool-use: {i}/{n}", end="", file=sys.stderr, flush=True),
        )
        print(file=sys.stderr)
        trajectories = [t.to_json() for t in trajs]
        if args.save_trajectories:
            write_jsonl(args.save_trajectories, trajectories)

    summary = score_trajectories(trajectories, examples)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("\ntool use (groundloop condition)")
        for key, val in summary.items():
            unit = "%" if key.endswith(("rate", "recall")) else ""
            print(f"  {key:<28} {val}{unit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
