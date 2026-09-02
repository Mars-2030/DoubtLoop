"""Is the three-way ordering a property of the method, or of my thresholds?

The support check has a knob (`SUPPORT_TAU_SENTENCE`) and the retriever has one
(`k`). Both were chosen by looking at cases where the metric got something
wrong, which is a legitimate way to set a threshold and also exactly the process
that produces a number tuned to flatter the result. The defence is to publish
the sweep: if the ordering base >= plain_critique > groundloop only holds at
tau = 0.70, that is a finding about the metric and it belongs in the open, not
in a footnote.

Two sweeps:

* **tau** - the three conditions are run once and then *re-scored* at each
  threshold. Same generations, same claims, different bar. Any change here is
  purely the metric moving.
* **k** - the GroundLoop condition is re-run at each retrieval depth, because k
  changes what the model sees and therefore what it says.

    python -m groundloop.eval.sensitivity
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from groundloop import config
from groundloop.datasets import load_qa
from groundloop.eval.hallucination_rate import score_trajectories
from groundloop.llm import add_backend_args, backend_from_args
from groundloop.loop.pipeline import CONDITIONS, run_dataset
from groundloop.tools.search import SearchTool

DEFAULT_TAUS = (0.50, 0.60, 0.70, 0.80, 0.90)
DEFAULT_KS = (1, 2, 4, 8)


def sweep_tau(trajectories: dict[str, list[dict]], examples, tool, taus) -> list[dict]:
    original = config.SUPPORT_TAU_SENTENCE
    rows = []
    try:
        for tau in taus:
            config.SUPPORT_TAU_SENTENCE = tau
            row = {"tau": tau}
            for condition in CONDITIONS:
                _, summary = score_trajectories(trajectories[condition], examples, tool)
                row[condition] = summary["hallucination_rate"]
            row["gap"] = round(row["plain_critique"] - row["groundloop"], 1)
            rows.append(row)
    finally:
        config.SUPPORT_TAU_SENTENCE = original
    return rows


def sweep_k(llm, examples, tool, ks, progress=None) -> list[dict]:
    rows = []
    for k in ks:
        trajs = [t.to_json() for t in run_dataset(llm, examples, "groundloop", tool=tool, k=k,
                                                  progress=progress)]
        _, summary = score_trajectories(trajs, examples, tool)
        rows.append({
            "k": k,
            "hallucination_rate": summary["hallucination_rate"],
            "accuracy": summary["accuracy"],
            "over_abstention_rate": summary.get("over_abstention_rate", 0.0),
            "correct_abstention_rate": summary.get("correct_abstention_rate", 0.0),
        })
    return rows


def render_markdown(tau_rows, k_rows, meta) -> str:
    lines = [
        "# Sensitivity",
        "",
        f"- generated: {meta['date']}  |  backend: `{meta['backend']}`  |  model: `{meta['model']}`",
        f"- n = {meta['n']} QA items  |  defaults in use: tau = {meta['tau']}, k = {meta['k']}",
        "",
        "## Support threshold (same generations, re-scored)",
        "",
        "Hallucination rate at each value of `SUPPORT_TAU_SENTENCE`. Nothing was",
        "re-generated; only the bar for calling a claim supported moved.",
        "",
        "| tau | base | plain critique | GroundLoop | gap (plain − GroundLoop) |",
        "|---|---|---|---|---|",
    ]
    for row in tau_rows:
        marker = " ←default" if abs(row["tau"] - meta["tau"]) < 1e-9 else ""
        lines.append(
            f"| {row['tau']:.2f}{marker} | {row['base']}% | {row['plain_critique']}% | "
            f"{row['groundloop']}% | {row['gap']} pts |"
        )
    lines += [
        "",
        "A metric doing its job gets stricter with tau in every column at once.",
        "What matters is the last column: if the gap survives the whole sweep,",
        "the effect is not an artefact of where the bar was set. If it collapses",
        "at one end, say so - that is the honest headline, not the default row.",
        "",
        "## Retrieval depth (GroundLoop re-run at each k)",
        "",
        "| k | hallucination rate | accuracy | over-abstention | correct abstention |",
        "|---|---|---|---|---|",
    ]
    for row in k_rows:
        marker = " ←default" if row["k"] == meta["k"] else ""
        lines.append(
            f"| {row['k']}{marker} | {row['hallucination_rate']}% | {row['accuracy']}% | "
            f"{row['over_abstention_rate']}% | {row['correct_abstention_rate']}% |"
        )
    lines += [
        "",
        "k trades two failures against each other: too small and the gold passage",
        "is missed, so the model correctly abstains on questions it could have",
        "answered; too large and the answer has more irrelevant text to be pulled",
        "off by. The right k is a choice about which error is cheaper, not a",
        "number to maximise.",
        "",
        "Regenerate with `python -m groundloop.eval.sensitivity`.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taus", default=",".join(str(t) for t in DEFAULT_TAUS))
    ap.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--json", action="store_true")
    add_backend_args(ap)
    args = ap.parse_args(argv)

    llm = backend_from_args(args)
    tool = SearchTool()
    examples = load_qa(limit=args.limit)
    taus = [float(t) for t in args.taus.split(",") if t.strip()]
    ks = [int(k) for k in args.ks.split(",") if k.strip()]

    def tick(label):
        return lambda i, n: print(f"\r  {label}: {i}/{n}   ", end="", file=sys.stderr, flush=True)

    trajectories = {
        c: [t.to_json() for t in run_dataset(llm, examples, c, tool=tool, progress=tick(c))]
        for c in CONDITIONS
    }
    tau_rows = sweep_tau(trajectories, examples, tool, taus)
    k_rows = sweep_k(llm, examples, tool, ks, progress=tick("k-sweep"))
    print(file=sys.stderr)

    meta = {
        "date": date.today().isoformat(),
        "backend": args.backend,
        "model": args.model or ("n/a (scripted stand-in)" if args.backend == "scripted" else "default"),
        "n": len(examples),
        "tau": config.SUPPORT_TAU_SENTENCE,
        "k": config.TOP_K,
    }
    if args.json:
        print(json.dumps({"meta": meta, "tau_sweep": tau_rows, "k_sweep": k_rows}, indent=2))
        return 0

    out_dir = Path(args.out_dir) if args.out_dir else config.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    table = render_markdown(tau_rows, k_rows, meta)
    (out_dir / "sensitivity.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"wrote {out_dir / 'sensitivity.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
