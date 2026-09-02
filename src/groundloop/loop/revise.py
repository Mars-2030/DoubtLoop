"""Stage 5: rewrite the draft so that it survives its own critique."""

from __future__ import annotations

from groundloop import prompts
from groundloop.llm import LLM
from groundloop.schema import Passage


def revise(
    llm: LLM,
    question: str,
    draft: str,
    critique_text: str,
    evidence: list[Passage] | None = None,
    evidence_block: str = "",
    grounded: bool = False,
    max_new_tokens: int = 384,
) -> str:
    text = llm.generate(
        prompts.revise_messages(question, draft, critique_text, evidence_block, grounded=grounded),
        stage="revise",
        ctx={
            "question": question,
            "draft": draft,
            "critique_text": critique_text,
            "evidence": evidence or [],
            "grounded": grounded,
        },
        max_new_tokens=max_new_tokens,
    )
    text = text.strip()
    # A revision that came back empty is worse than no revision at all.
    return text or draft
