"""Loading the evaluation sets, and writing run logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from groundloop import config


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_qa(path: str | Path | None = None, limit: int | None = None,
            kinds: list[str] | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(path or config.QA_PAIRS_PATH)
    if kinds:
        rows = [r for r in rows if r.get("type") in kinds]
    return rows[:limit] if limit else rows


def load_sycophancy(path: str | Path | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(path or config.SYCOPHANCY_PATH)
    return rows[:limit] if limit else rows
