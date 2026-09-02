#!/usr/bin/env python3
"""Fail if the committed tables in `results/` are not what the code produces.

Regenerates everything into a temporary directory and compares. Provenance
lines that legitimately change between runs - the generation date - are
normalised out, so this catches content drift and nothing else.

The point is narrow but load-bearing: the README quotes these tables, and a
number in a README that no longer matches the code is worse than no number.

    python scripts/check_results_drift.py          # exit 1 on drift
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from groundloop.eval import retrieval_stress, run_all, sensitivity  # noqa: E402

# Lines whose content is a timestamp rather than a result.
_VOLATILE = (
    re.compile(r"^- generated: .*$"),
    re.compile(r'^\s*"date": ".*",?$'),
)

FILES = ("comparison_table.md", "retrieval_stress.md", "sensitivity.md", "metrics.json")


def normalise(text: str) -> str:
    out = []
    for line in text.splitlines():
        for pattern in _VOLATILE:
            if pattern.match(line):
                line = "<generated>"
                break
        out.append(line)
    return "\n".join(out)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        run_all.main(["--out-dir", tmp, "--no-save-trajectories"])
        retrieval_stress.main(["--out-dir", tmp])
        sensitivity.main(["--out-dir", tmp])

        drifted = []
        for name in FILES:
            committed = ROOT / "results" / name
            fresh = Path(tmp) / name
            if not committed.exists():
                drifted.append(f"{name}: not committed")
                continue
            if normalise(committed.read_text()) != normalise(fresh.read_text()):
                drifted.append(f"{name}: differs from a fresh run")

    if drifted:
        print("\nresults/ has drifted from the code:", file=sys.stderr)
        for item in drifted:
            print(f"  - {item}", file=sys.stderr)
        print("\nrun `make results` and commit the output.", file=sys.stderr)
        return 1
    print("\nresults/ matches a fresh run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
