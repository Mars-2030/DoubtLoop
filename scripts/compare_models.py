#!/usr/bin/env python3
"""Put several models' runs side by side.

    python scripts/compare_models.py results-0.6b/metrics.json results-1.7b/metrics.json

The question this exists to answer is not "which model is better" - a bigger
model is better at everything - but **where the method starts working**. The
0.6B run showed retrieval already perfect (100% well-formed calls, 100% gold
recall) and the critique step approving 53% of the claims the evidence does not
support. If false approval falls with scale while retrieval stays pinned at
100%, the ceiling on GroundLoop is the critique step and the method has a size
at which it turns on. If false approval stays flat, the problem is the critique
*prompt*, not the model, and no amount of scale will fix it.

That is why the table below puts retrieval and critique next to the outcome
metrics: the outcome alone cannot distinguish those two explanations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta, conds = payload.get("meta", {}), payload.get("conditions", {})
    base, plain, ground = conds.get("base", {}), conds.get("plain_critique", {}), conds.get("groundloop", {})
    tool = ground.get("tool_use", {})
    critique = ground.get("critique", {})

    accuracy_p = None
    for comp in payload.get("significance", []):
        if comp.get("attribute") == "correct":
            accuracy_p = (comp.get("vs_base") or {}).get("p_value")

    return {
        "model": meta.get("model", path.parent.name),
        "backend": meta.get("backend", "?"),
        "n": meta.get("n_qa", 0),
        "base_accuracy": base.get("qa", {}).get("accuracy"),
        "plain_accuracy": plain.get("qa", {}).get("accuracy"),
        "ground_accuracy": ground.get("qa", {}).get("accuracy"),
        "accuracy_p": accuracy_p,
        "base_unsupported": base.get("qa", {}).get("unsupported_claim_rate"),
        "ground_unsupported": ground.get("qa", {}).get("unsupported_claim_rate"),
        "base_pushback": base.get("sycophancy", {}).get("pushback_rate"),
        "ground_pushback": ground.get("sycophancy", {}).get("pushback_rate"),
        "call_rate": tool.get("well_formed_call_rate"),
        "gold_recall": tool.get("gold_passage_recall"),
        "false_approval": critique.get("false_approval_rate"),
        "blind_critique": critique.get("blind_critique_rate"),
    }


def _pct(v):
    return "-" if v is None else f"{v}%"


def _delta(before, after):
    if before is None or after is None:
        return "-"
    return f"{before}% → {after}%"


def render(rows: list[dict]) -> str:
    sizes = {r["n"] for r in rows}
    lines = [
        "# Does the method work at other sizes?",
        "",
        "Every row is the same questions, the same corpus, the same prompts — only",
        "the model changes. All conditions are inference-time scaffolding over",
        "frozen weights.",
        "",
    ]
    if len(sizes) > 1:
        # Comparing models across different question sets measures the subsets,
        # not the models, and nothing downstream would reveal it.
        lines += [
            f"> ⚠️ **These runs cover different numbers of questions ({sorted(sizes)}).**",
            "> The rows are not comparable: re-run them over the same set (drop",
            "> `--limit`, or pass the same one) before reading anything into the",
            "> differences.",
            "",
        ]
    lines += [
        "## Outcome",
        "",
        "| Model | n | Accuracy (base → GroundLoop) | p | Unsupported claims | Sycophancy pushback |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        p = "-" if r["accuracy_p"] is None else str(r["accuracy_p"])
        lines.append(
            f"| `{r['model']}` | {r['n']} | {_delta(r['base_accuracy'], r['ground_accuracy'])} | {p} | "
            f"{_delta(r['base_unsupported'], r['ground_unsupported'])} | "
            f"{_delta(r['base_pushback'], r['ground_pushback'])} |"
        )
    lines += [
        "",
        "## Why it does or does not work",
        "",
        "| Model | Well-formed tool call | Gold passage recall | Critique false approval | Waved through a bad draft |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['model']}` | {_pct(r['call_rate'])} | {_pct(r['gold_recall'])} | "
            f"**{_pct(r['false_approval'])}** | {_pct(r['blind_critique'])} |"
        )
    lines += [
        "",
        "## How to read the second table",
        "",
        "Retrieval saturates early — a 0.6B model already calls the tool correctly",
        "every time and gets the gold passage every time. So the first two columns",
        "should be ~100% everywhere, and if they are, they are not the story.",
        "",
        "The story is false approval: the fraction of claims the model marks",
        "\"supported\", naming a passage, where the evidence does not support them.",
        "That is the ceiling on everything in the first table, because a critique",
        "that endorses the draft produces no revision.",
        "",
        "- **Falls with scale** → the method has a size at which it turns on, and",
        "  the interesting number is where the curve crosses into usefulness.",
        "- **Stays flat** → the problem is the critique prompt or the task framing,",
        "  not model capacity, and scaling is the wrong lever.",
        "",
        "Regenerate with `python scripts/compare_models.py <metrics.json> ...`.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics", nargs="+", help="metrics.json files, smallest model first")
    ap.add_argument("--out", default=None, help="write markdown here (default: stdout only)")
    args = ap.parse_args(argv)

    rows = []
    for name in args.metrics:
        path = Path(name)
        if not path.exists():
            raise SystemExit(f"{path} not found — pass the metrics.json from each run's --out-dir")
        rows.append(load(path))

    table = render(rows)
    print(table)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table, encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
