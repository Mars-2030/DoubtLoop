"""Prompt templates for every stage of the loop.

The critique prompt is the one that matters. Vanilla Constitutional AI shows
the model its own draft plus a constitution and asks it to find problems; the
model's only reference for "is this true?" is the same parametric memory that
produced the draft. Here the critique prompt also carries the retrieved
passages and instructs the model to treat *those* as the arbiter. Everything
else is scaffolding around that swap.
"""

from __future__ import annotations

import functools

from groundloop import config

# --- system prompts --------------------------------------------------------

DRAFT_SYSTEM = (
    "You are a careful assistant. Answer the user's question directly and "
    "concisely, in at most four sentences."
)

RESEARCH_SYSTEM = (
    "You are a careful assistant with access to a search tool over a fixed "
    "evidence corpus. Before you commit to a factual claim, look it up.\n\n"
    "To search, emit exactly one tool call and nothing else:\n"
    '<tool_call>{"name": "search", "arguments": {"query": "<keywords>"}}</tool_call>\n\n'
    "Use keywords, not a full sentence. One call per turn."
)

CRITIQUE_SYSTEM = (
    "You are auditing a draft answer for faithfulness. You are strict, "
    "specific, and you quote the passage id that settles each point."
)

REVISE_SYSTEM = (
    "You rewrite draft answers so that every claim they make is supported by "
    "the evidence provided, and nothing else is asserted."
)


_FALLBACK_CONSTITUTION = (
    "P3. Do not state anything unsupported by the evidence.\n"
    "P4. When the evidence contradicts your prior, follow the evidence.\n"
    "P5. Say when you do not know."
)


@functools.lru_cache(maxsize=1)
def constitution() -> str:
    """The principles from `constitution.md`, verbatim.

    Only the tiered principles are sent to the model; the surrounding prose in
    the file explains the design to a reader and would just be tokens the model
    has to skip.
    """
    try:
        raw = config.CONSTITUTION_PATH.read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - only when the repo is partially copied
        return _FALLBACK_CONSTITUTION
    start = raw.find("## Tier 1")
    end = raw.find("## How the critique step uses this")
    body = raw[start:end if end > start else None] if start >= 0 else raw
    body = body.replace("---", "").strip()
    return "The constitution you are auditing against:\n\n" + body


# --- stage templates -------------------------------------------------------

DRAFT_USER = "{question}"

TOOL_USER = (
    "Question: {question}\n\n"
    "Draft answer you are about to give:\n{draft}\n\n"
    "Search the corpus for the evidence that would confirm or refute this draft. "
    "Emit one tool call."
)

CLAIMS_USER = (
    "Question: {question}\n\nDraft answer:\n{draft}\n\n"
    "List the checkable factual claims in the draft, one per line, no numbering, "
    "no commentary. A claim is checkable if a document could confirm or refute it."
)

# Critique WITHOUT retrieval - the vanilla-CAI control condition.
CRITIQUE_PLAIN_USER = """{constitution}

---

Question: {question}

Draft answer:
{draft}

Critique this draft against the constitution above. Identify any claim that is
unsupported, overstated, or that accepts a false premise in the question.

Reply with JSON only:
{{"verdict": "ok" | "revise",
  "issues": [{{"claim": "<quoted from the draft>", "principle": "<P#>", "problem": "<what is wrong>", "fix": "<what to do instead>"}}]}}
"""

# Critique WITH retrieval - the GroundLoop condition. The instruction to prefer
# evidence over recollection (P4) is repeated inline because it is the single
# behaviour the whole method is trying to install.
CRITIQUE_GROUNDED_USER = """{constitution}

---

Question: {question}

Draft answer:
{draft}

Evidence retrieved by search("{query}"):
{evidence}

Check the draft against THE EVIDENCE ABOVE, not against your own recollection.
For every factual claim in the draft, decide whether the evidence supports it,
contradicts it, or is silent. If the evidence contradicts what you believe,
the evidence wins. If the evidence is silent on the question, the correct
critique is that the draft should abstain.

Reply with JSON only:
{{"verdict": "ok" | "revise",
  "issues": [{{"claim": "<quoted from the draft>", "principle": "<P#>", "status": "supported" | "contradicted" | "unsupported", "passage": "<id or empty>", "problem": "<what is wrong>", "fix": "<what to do instead>"}}]}}
"""

REVISE_PLAIN_USER = """Question: {question}

Draft answer:
{draft}

Critique:
{critique}

Rewrite the answer so it addresses every issue in the critique. Output the
revised answer only, no preamble.
"""

REVISE_GROUNDED_USER = """Question: {question}

Draft answer:
{draft}

Evidence:
{evidence}

Critique:
{critique}

Rewrite the answer so that every claim is supported by the evidence above.
Rules:
- Cite the passage id in square brackets after each claim, e.g. [r01].
- Drop any claim the evidence does not support. Do not replace it with a vaguer
  version of the same claim.
- If the evidence does not answer the question, say so explicitly and stop.
- If the question assumes something the evidence contradicts, correct it first.

Output the revised answer only, no preamble.
"""


def draft_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": DRAFT_SYSTEM},
        {"role": "user", "content": DRAFT_USER.format(question=question)},
    ]


def tool_messages(question: str, draft: str) -> list[dict]:
    return [
        {"role": "system", "content": RESEARCH_SYSTEM},
        {"role": "user", "content": TOOL_USER.format(question=question, draft=draft)},
    ]


def claims_messages(question: str, draft: str) -> list[dict]:
    return [
        {"role": "system", "content": CRITIQUE_SYSTEM},
        {"role": "user", "content": CLAIMS_USER.format(question=question, draft=draft)},
    ]


def critique_messages(question: str, draft: str, evidence: str = "", query: str = "",
                      grounded: bool | None = None) -> list[dict]:
    """`grounded` selects the template. It defaults to "did we get evidence?"
    for convenience, but the pipeline always passes it explicitly: an empty
    retrieval must still take the grounded path, or the GroundLoop condition
    quietly becomes the control condition on exactly the questions where the
    corpus had nothing to say.
    """
    if grounded is None:
        grounded = bool(evidence)
    if grounded:
        content = CRITIQUE_GROUNDED_USER.format(
            constitution=constitution(), question=question, draft=draft,
            evidence=evidence, query=query,
        )
    else:
        content = CRITIQUE_PLAIN_USER.format(
            constitution=constitution(), question=question, draft=draft,
        )
    return [
        {"role": "system", "content": CRITIQUE_SYSTEM},
        {"role": "user", "content": content},
    ]


def revise_messages(question: str, draft: str, critique: str, evidence: str = "",
                    grounded: bool | None = None) -> list[dict]:
    if grounded is None:
        grounded = bool(evidence)
    if grounded:
        content = REVISE_GROUNDED_USER.format(
            question=question, draft=draft, critique=critique, evidence=evidence
        )
    else:
        content = REVISE_PLAIN_USER.format(question=question, draft=draft, critique=critique)
    return [
        {"role": "system", "content": REVISE_SYSTEM},
        {"role": "user", "content": content},
    ]
