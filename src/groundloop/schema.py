"""Dataclasses for the objects that move through the loop.

Everything here round-trips to plain JSON so a whole run can be logged to a
JSONL file and replayed later without the code that produced it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


def _asdict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _asdict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_asdict(v) for v in obj]
    return obj


@dataclass
class Passage:
    id: str
    title: str
    text: str
    world: str = "real"
    tags: list[str] = field(default_factory=list)
    score: float = 0.0

    def cite(self) -> str:
        return f"[{self.id}] {self.title}"


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str = ""
    ok: bool = True
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "ok": self.ok, "error": self.error}


@dataclass
class Claim:
    text: str
    supported: bool = False
    best_passage: str = ""
    coverage: float = 0.0
    reason: str = ""


@dataclass
class Critique:
    verdict: str = "ok"  # "ok" | "revise"
    claims: list[Claim] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def unsupported(self) -> list[Claim]:
        return [c for c in self.claims if not c.supported]


@dataclass
class Trajectory:
    """One question run end-to-end through one condition."""

    example_id: str
    condition: str  # "base" | "plain_critique" | "groundloop"
    question: str
    draft: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    evidence: list[Passage] = field(default_factory=list)
    critique: Critique | None = None
    revision: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def final(self) -> str:
        return self.revision or self.draft

    def to_json(self) -> dict[str, Any]:
        return _asdict(self)
