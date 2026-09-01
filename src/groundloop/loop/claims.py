"""Stage 2: pull the checkable claims out of a draft, and turn them into queries.

Two implementations of the same step:

* ``extract_claims_lexical`` - sentence-level, deterministic, free. Used by the
  evaluation harness, where a claim splitter that drifts between conditions
  would corrupt the comparison.
* ``extract_claims_model``   - asks the model. Used inside the loop, because the
  trajectory we later fine-tune on should contain the model's own decomposition,
  not the harness's.
"""

from __future__ import annotations

from groundloop import prompts
from groundloop.llm import LLM
from groundloop.textutil import content_tokens, extract_claims as extract_claims_lexical  # noqa: F401


def extract_claims_model(llm: LLM, question: str, draft: str, max_claims: int = 8) -> list[str]:
    raw = llm.generate(
        prompts.claims_messages(question, draft),
        stage="claims",
        ctx={"question": question, "draft": draft},
        max_new_tokens=256,
    )
    out = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-*•0123456789. ").strip()
        if len(content_tokens(line)) >= 2:
            out.append(line)
    if not out:
        out = extract_claims_lexical(draft)
    return out[:max_claims]


def build_query(question: str, claims: list[str] | None = None, max_terms: int = 10) -> str:
    """A keyword query from the question, topped up with claim-specific terms.

    Question terms come first because the question is what the answer has to be
    about; claim terms are what makes the query able to *refute* the draft
    rather than merely re-confirm its topic.
    """
    seen: list[str] = []
    for tok in content_tokens(question):
        if tok not in seen:
            seen.append(tok)
    for claim in claims or []:
        for tok in content_tokens(claim):
            if tok not in seen:
                seen.append(tok)
    return " ".join(seen[:max_terms])
