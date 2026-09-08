"""The whole ablation in one command.

    python -m groundloop.eval.run_all --backend openai --model Qwen/Qwen3.5-0.8B-Instruct

Runs all three conditions over the QA set and the sycophancy probe, scores tool
use, and writes `results/comparison_table.md` and `results/metrics.json`.

The generated table carries a provenance header naming the backend and model it
came from. That is not decoration: with `--backend scripted` the numbers
describe the harness, not a model, and a table that does not say so is a table
someone will eventually quote as a result.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import date
from pathlib import Path

from groundloop import __version__, config
from groundloop.datasets import load_qa, load_sycophancy, read_jsonl, write_jsonl
from groundloop.eval import hallucination_rate as hall
from groundloop.eval import sycophancy_probe as syco
from groundloop.eval import critique_quality as critique_eval
from groundloop.eval import significance
from groundloop.eval import tool_use_quality as tools_eval
from groundloop.llm import add_backend_args, backend_from_args
from groundloop.loop.pipeline import CONDITIONS, run_dataset
from groundloop.tools.search import SearchTool

CONDITION_LABELS = {
    "base": "Base model, no critique",
    "plain_critique": "+ plain self-critique (constitution only)",
    "groundloop": "+ GroundLoop (constitution + retrieval)",
}


def _require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(
            f"{path} not found. --from-trajectories expects the raw/ directory that a "
            f"previous run wrote (qa_<condition>.jsonl and probe_<condition>.jsonl)."
        )
    return path


def _progress(label: str):
    def tick(i: int, n: int) -> None:
        print(f"\r  {label}: {i}/{n}   ", end="", file=sys.stderr, flush=True)

    return tick


def run(args, out_dir: Path) -> dict:
    """Generate (or re-load) the trajectories for all three conditions and score them.

    `--from-trajectories` re-scores a previous run instead of regenerating it.
    That path exists because a change to the *scorer* is not a change to the
    *model*: after a metric fix, the honest thing is to re-score the same
    generations, not to sample new ones and hope the difference was the metric.
    It also happens to save the half-hour of GPU time.
    """
    tool = SearchTool()
    qa = load_qa(limit=args.limit)
    probe = load_sycophancy(limit=args.limit)

    results: dict[str, dict] = {}
    qa_scores: dict[str, list] = {}
    probe_scores: dict[str, list] = {}
    raw_dir = out_dir / "raw"
    replay = Path(args.from_trajectories) if args.from_trajectories else None
    llm = None if replay else backend_from_args(args)

    for condition in CONDITIONS:
        if replay:
            qa_trajs = read_jsonl(_require(replay / f"qa_{condition}.jsonl"))
            probe_path = replay / f"probe_{condition}.jsonl"
            probe_trajs = read_jsonl(probe_path) if probe_path.exists() else []
            qa = [e for e in qa if e["id"] in {t["example_id"] for t in qa_trajs}]
            probe = [e for e in probe if e["id"] in {t["example_id"] for t in probe_trajs}]
        else:
            qa_trajs = [
                t.to_json()
                for t in run_dataset(llm, qa, condition, tool=tool, k=args.k,
                                     rounds=args.rounds, progress=_progress(f"qa/{condition}"))
            ]
            probe_trajs = [
                t.to_json()
                for t in run_dataset(llm, probe, condition, tool=tool, k=args.k,
                                     rounds=args.rounds, progress=_progress(f"probe/{condition}"))
            ]
            if args.save_trajectories:
                write_jsonl(raw_dir / f"qa_{condition}.jsonl", qa_trajs)
                write_jsonl(raw_dir / f"probe_{condition}.jsonl", probe_trajs)

        qa_scored, qa_summary = hall.score_trajectories(qa_trajs, qa, tool)
        probe_scored, probe_summary = syco.score_trajectories(probe_trajs, probe)
        qa_scores[condition] = qa_scored
        probe_scores[condition] = probe_scored
        entry = {"qa": qa_summary, "sycophancy": probe_summary}
        if condition == "groundloop":
            entry["tool_use"] = tools_eval.score_trajectories(qa_trajs, qa)
            entry["critique"] = critique_eval.score_trajectories(qa_trajs)[1]
        results[condition] = entry

    if not replay:
        print(file=sys.stderr)

    # Paired, because every condition answered the same questions in the same
    # order. At n=39 this is what separates a result from a direction.
    comparisons = [
        significance.compare(qa_scores, "hallucinated", higher_is_better=False),
        significance.compare(qa_scores, "correct", higher_is_better=True),
        significance.compare(probe_scores, "pushback", higher_is_better=True),
    ]
    return {
        "significance": comparisons,
        "meta": {
            "groundloop_version": __version__,
            "date": date.today().isoformat(),
            "backend": "re-scored" if args.from_trajectories else args.backend,
            "source": f"re-scored from {args.from_trajectories}" if args.from_trajectories else "generated",
            "model": (
                args.model
                or ("as recorded in the original run" if args.from_trajectories else None)
                or ("n/a (scripted stand-in)" if args.backend == "scripted" else "default")
            ),
            "scorer": "lexical (deterministic)",
            "k": args.k,
            "rounds": args.rounds,
            "n_qa": len(qa),
            "n_probe": len(probe),
            "python": platform.python_version(),
            "thresholds": {
                "sentence": config.SUPPORT_TAU_SENTENCE,
                "passage": config.SUPPORT_TAU_PASSAGE,
                "union": config.SUPPORT_TAU_UNION,
            },
        },
        "conditions": results,
    }


def render_markdown(payload: dict) -> str:
    meta = payload["meta"]
    conds = payload["conditions"]
    scripted = meta["backend"] == "scripted"
    replayed = meta.get("source", "generated") != "generated"

    lines = ["# GroundLoop: three-way comparison", ""]
    if replayed:
        lines += [
            "> Re-scored from saved trajectories, not regenerated. The generations are",
            "> whatever the original run produced - check that run for which model and",
            "> backend made them. Only the scoring is from this version of the code.",
            "",
        ]
    if scripted:
        lines += [
            "> **These numbers are not a model result.** They were produced with",
            "> `--backend scripted`, the deterministic stand-in that lets the",
            "> pipeline run with no weights. They demonstrate that the harness",
            "> measures what it claims to measure. Re-run with `--backend openai`",
            "> or `--backend transformers` against real weights to get a result.",
            "",
        ]
    lines += [
        f"- generated: {meta['date']}  (groundloop {meta['groundloop_version']})",
        f"- backend: `{meta['backend']}`  |  model: `{meta['model']}`"
        + (f"  |  {meta['source']}" if meta.get("source", "generated") != "generated" else ""),
        f"- scorer: {meta['scorer']}  |  top-k: {meta['k']}  |  critique rounds: {meta['rounds']}",
        f"- n = {meta['n_qa']} QA items, {meta['n_probe']} sycophancy probes",
        "",
        "## Headline",
        "",
        "| Setting | Hallucination rate | Unsupported claims | Accuracy | Sycophancy pushback |",
        "|---|---|---|---|---|",
    ]
    for cond in CONDITIONS:
        qa = conds[cond]["qa"]
        sy = conds[cond]["sycophancy"]
        lines.append(
            f"| {CONDITION_LABELS[cond]} | {qa['hallucination_rate']}% | "
            f"{qa['unsupported_claim_rate']}% | {qa['accuracy']}% | {sy['pushback_rate']}% |"
        )

    lines += [
        "",
        "Lower is better for the first two columns; higher is better for the last two.",
        "",
        "## Full QA metrics",
        "",
        "| Metric | " + " | ".join(CONDITION_LABELS[c] for c in CONDITIONS) + " |",
        "|---|" + "---|" * len(CONDITIONS),
    ]
    qa_keys = [
        ("hallucination_rate", "Hallucination rate (answer-level)", "%"),
        ("unsupported_claim_rate", "Unsupported claim rate", "%"),
        ("accuracy", "Accuracy (all items)", "%"),
        ("accuracy_answerable", "Accuracy (answerable only)", "%"),
        ("correct_abstention_rate", "Correct abstention (unanswerable)", "%"),
        ("over_abstention_rate", "Over-abstention (answerable)", "%"),
        ("citation_rate", "Cites a gold passage", "%"),
        ("claims_per_answer", "Claims per answer", ""),
    ]
    for key, label, unit in qa_keys:
        row = [str(conds[c]["qa"].get(key, "-")) + (unit if key in conds[c]["qa"] else "") for c in CONDITIONS]
        lines.append(f"| {label} | " + " | ".join(row) + " |")

    lines += [
        "",
        "## Sycophancy probe",
        "",
        "| Metric | " + " | ".join(CONDITION_LABELS[c] for c in CONDITIONS) + " |",
        "|---|" + "---|" * len(CONDITIONS),
    ]
    for key, label in [
        ("pushback_rate", "Pushback rate (states the correction)"),
        ("capitulation_rate", "Capitulation rate (accepts the false premise)"),
        ("correction_stated_rate", "Correction present anywhere in the answer"),
    ]:
        row = [f"{conds[c]['sycophancy'].get(key, '-')}%" for c in CONDITIONS]
        lines.append(f"| {label} | " + " | ".join(row) + " |")

    tu = conds["groundloop"].get("tool_use", {})
    if tu:
        lines += ["", "## Tool use (GroundLoop condition only)", "", "| Metric | Value |", "|---|---|"]
        labels = {
            "calls_per_example": "Search calls per question",
            "well_formed_call_rate": "Well-formed model-issued call",
            "harness_fallback_rate": "Harness had to build the query",
            "no_call_rate": "No call attempted",
            "empty_retrieval_rate": "Search returned nothing",
            "gold_passage_hit_rate": "Retrieved at least one gold passage",
            "gold_passage_recall": "Gold passage recall",
            "evidence_citation_rate": "Final answer cites retrieved evidence",
            "grounded_in_retrieved_rate": "Claims supported by what was retrieved",
        }
        for key, label in labels.items():
            if key in tu:
                unit = "%" if key.endswith(("rate", "recall")) else ""
                lines.append(f"| {label} | {tu[key]}{unit} |")

    cq = conds["groundloop"].get("critique", {})
    if cq.get("claims_paired"):
        lines += [
            "", "## Self-critique vs the evidence (GroundLoop condition)", "",
            "Whether the critique step agrees with the passages it was shown. If it",
            "endorses whatever the draft said, retrieval and revision are both wasted.",
            "", "| Metric | Value |", "|---|---|",
        ]
        for key, label in {
            "claims_paired": "Claims judged by both",
            "agreement_rate": "Agreement with the evidence check",
            "false_approval_rate": "Approved what the evidence does not support",
            "false_alarm_rate": "Rejected what the evidence does support",
            "blind_critique_rate": "Waved through a draft with a real error",
        }.items():
            if key in cq:
                unit = "%" if key.endswith("rate") else ""
                lines.append(f"| {label} | {cq[key]}{unit} |")
        lines += [
            "",
            "Agreement with the lexical check, not with ground truth, so",
            "false approval is an upper bound on the model's error and false alarm a",
            "lower bound. With the scripted stand-in these are trivially 100% / 0%:",
            "its critique *is* the lexical check, so it cannot disagree with itself.",
        ]

    lines += significance.render_markdown(payload.get("significance", []))

    lines += [
        "",
        "## Reading this table",
        "",
        "The middle row is the control that makes the experiment an experiment.",
        "Any gap between the base model and plain self-critique is what the model",
        "gains from *thinking harder* about its own answer. The gap between plain",
        "self-critique and GroundLoop is what it gains from *checking a source*.",
        "If the second gap were small, the retrieval step would not be earning its",
        "cost and the honest conclusion would be that vanilla CAI is enough.",
        "",
        "Regenerate with `make eval` (or `python -m groundloop.eval.run_all`).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("-k", type=int, default=config.TOP_K)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--from-trajectories", default=None, metavar="DIR",
                    help="re-score a previous run's raw/ directory instead of "
                         "regenerating it. Use this after a change to the scorer: the "
                         "generations are unaffected, so sampling new ones only adds noise.")
    ap.add_argument("--save-trajectories", action="store_true", default=True)
    ap.add_argument("--no-save-trajectories", dest="save_trajectories", action="store_false")
    add_backend_args(ap)
    args = ap.parse_args(argv)

    out_dir = config.RESULTS_DIR if args.out_dir is None else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = run(args, out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    table = render_markdown(payload)
    (out_dir / "comparison_table.md").write_text(table, encoding="utf-8")

    print(table)
    print(f"\nwrote {out_dir / 'comparison_table.md'} and {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
