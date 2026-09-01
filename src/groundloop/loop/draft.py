"""Stage 1: the first-pass answer, produced with no evidence in front of it.

This is deliberately the *un*-helped condition even inside GroundLoop. The
draft is what the model believes; the rest of the loop is what happens when
that belief meets a source.
"""

from __future__ import annotations

from groundloop import prompts
from groundloop.llm import LLM


def draft(llm: LLM, question: str, max_new_tokens: int = 256) -> str:
    text = llm.generate(
        prompts.draft_messages(question),
        stage="draft",
        ctx={"question": question},
        max_new_tokens=max_new_tokens,
    )
    return text.strip()
