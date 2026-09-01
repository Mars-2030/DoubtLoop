"""Optional LLM judge, as a drop-in replacement for the lexical support check.

The lexical checker is deterministic and auditable but blind to paraphrase.
The judge is the opposite. Reporting both is the honest thing to do: if the
three-way ordering only holds under one scorer, that is a finding about the
metric, not about the method.
"""

from __future__ import annotations

import re
from typing import Any

from groundloop.eval.metrics import AnswerScore, answer_matches_keys, cited_ids
from groundloop.llm import LLM
from groundloop.schema import Passage
from groundloop.textutil import extract_claims, is_abstention

JUDGE_SYSTEM = (
    "You are a strict grader. You judge only whether a claim is entailed by the "
    "passages you are given. You never use outside knowledge."
)

JUDGE_USER = """Passages:
{evidence}

Claim:
{claim}

Is the claim supported by the passages above? Answer with exactly one word:
SUPPORTED if the passages entail it, CONTRADICTED if they contradict it, or
UNSUPPORTED if they are silent on it. Do not use any knowledge of your own.
"""


class LLMJudge:
    """Same `score(...)` signature as `metrics.score_answer`."""

    def __init__(self, llm: LLM, max_claims: int = 12):
        self.llm = llm
        self.max_claims = max_claims
        self.cache: dict[tuple[str, str], str] = {}

    def _verdict(self, claim: str, evidence_block: str) -> str:
        key = (claim, evidence_block)
        if key in self.cache:
            return self.cache[key]
        raw = self.llm.generate(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_USER.format(evidence=evidence_block, claim=claim)},
            ],
            stage="judge",
            ctx={"claim": claim, "evidence_block": evidence_block},
            max_new_tokens=8,
            temperature=0.0,
        )
        m = re.search(r"\b(SUPPORTED|CONTRADICTED|UNSUPPORTED)\b", raw.upper())
        verdict = m.group(1) if m else "UNSUPPORTED"
        self.cache[key] = verdict
        return verdict

    def score(self, example: dict[str, Any], answer: str, evidence: list[Passage],
              condition: str = "") -> AnswerScore:
        block = "\n\n".join(f"[{p.id}] {p.title}\n{p.text}" for p in evidence) or "(none)"
        expect_abstention = bool(example.get("expect_abstention"))
        score = AnswerScore(
            example_id=example.get("id", ""), condition=condition,
            expect_abstention=expect_abstention, kind=example.get("type", ""),
        )
        score.abstained = is_abstention(answer)
        for claim in extract_claims(answer)[: self.max_claims]:
            score.n_claims += 1
            if self._verdict(claim, block) != "SUPPORTED":
                score.n_unsupported += 1
                score.unsupported_claims.append(claim)
        if expect_abstention:
            score.correct = score.abstained and score.n_unsupported == 0
            score.hallucinated = not score.abstained or score.n_unsupported > 0
        else:
            score.correct = answer_matches_keys(answer, example.get("answer_keys", []))
            score.hallucinated = score.n_unsupported > 0
        score.cited_gold = bool(set(example.get("support", []) or []) & cited_ids(answer))
        return score
