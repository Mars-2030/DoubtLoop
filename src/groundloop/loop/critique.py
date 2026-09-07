"""Stage 4: critique the draft against the constitution and the evidence.

The parser accepts a JSON object, a JSON object buried in prose, or a bare
list of problems, because a 0.8B model asked for JSON returns JSON most of the
time and returns something JSON-adjacent the rest of the time. A format miss
should not read as "the model found no issues" - that would quietly inflate the
measured faithfulness of whichever condition is worst at formatting.
"""

from __future__ import annotations

import json
import re

from groundloop import config, prompts
from groundloop.llm import LLM
from groundloop.schema import Claim, Critique, Passage
from groundloop.textutil import check_claim_default, extract_claims

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_critique(raw: str) -> Critique:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJ.search(text)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None

    if isinstance(data, dict):
        # "claims" is the grounded schema (every claim, each with a status);
        # "issues" is the plain-critique one (problems only). Accept both, so
        # trajectories logged under either shape still parse.
        issues = data.get("claims") or data.get("issues") or []
        claims = []
        for issue in issues:
            if not isinstance(issue, dict):
                claims.append(Claim(text=str(issue), supported=False, reason="unparsed issue"))
                continue
            claims.append(
                Claim(
                    text=str(issue.get("claim", "")).strip(),
                    supported=str(issue.get("status", "")).lower() == "supported",
                    best_passage=str(issue.get("passage", "")).strip(),
                    reason=" ".join(
                        s for s in [str(issue.get("principle", "")).strip(),
                                    str(issue.get("problem", "")).strip()] if s
                    ),
                )
            )
        # Findings outrank the label. Small models routinely return
        # {"verdict": "ok"} beside a claim they just marked contradicted, and
        # the header is the part they get wrong. With no findings, fall back to
        # the stated verdict - a model that says "revise" and then fails to
        # enumerate has still flagged something, and treating an enumeration
        # miss as an all-clear is the failure this parser exists to avoid.
        stated = str(data.get("verdict", "")).lower()
        if any(not c.supported for c in claims):
            verdict = "revise"
        elif stated in ("ok", "revise"):
            verdict = stated
        else:
            verdict = "ok"
        return Critique(verdict=verdict, claims=claims, raw=raw)

    # Not JSON. Fall back to reading it as prose: any non-empty line that is not
    # an explicit all-clear counts as an issue.
    lines = [ln.strip("-*• ").strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        # A model that produced nothing has not approved anything. This was the
        # one remaining path where a format miss read as an all-clear - and it
        # is indistinguishable downstream from a genuine "looks fine", so a
        # generation failure was being reported as the model endorsing its own
        # wrong answer. The evidence is already in hand, so leave it unresolved
        # and let the revision stage run.
        return Critique(verdict="revise", raw=raw,
                        notes=["critique came back empty; treating as unresolved, not as approval"])
    if len(lines) == 1 and re.fullmatch(r"(ok|no issues\.?|none\.?)", lines[0], re.I):
        return Critique(verdict="ok", raw=raw, notes=["explicit all-clear"])
    return Critique(
        verdict="revise",
        claims=[Claim(text=ln, supported=False, reason="unstructured critique") for ln in lines[:8]],
        notes=["critique was not valid JSON; parsed as prose"],
        raw=raw,
    )


def critique(
    llm: LLM,
    question: str,
    draft: str,
    evidence: list[Passage] | None = None,
    query: str = "",
    evidence_block: str = "",
    grounded: bool = False,
    max_new_tokens: int = 512,
) -> Critique:
    """Run the critique stage. `grounded=False` is the plain-CAI control.

    Note that `grounded=True` with empty `evidence` is a real and important
    state - the search ran and found nothing - and it must stay on the grounded
    path so the correct critique is "this draft should abstain".
    """
    raw = llm.generate(
        prompts.critique_messages(question, draft, evidence_block, query, grounded=grounded),
        stage="critique",
        ctx={"question": question, "draft": draft, "evidence": evidence or [],
             "query": query, "grounded": grounded},
        max_new_tokens=max_new_tokens,
    )
    parsed = parse_critique(raw)
    if grounded:
        parsed = annotate_with_evidence(parsed, draft, evidence or [])
    return parsed


def annotate_with_evidence(parsed: Critique, draft: str, evidence: list[Passage]) -> Critique:
    """Attach the harness's own support verdict to every claim in the draft.

    The model's critique is what gets trained on; this annotation is what gets
    measured. Keeping them separate is the difference between evaluating the
    method and evaluating the model's opinion of itself.
    """
    checked = []
    for text in extract_claims(draft):
        ok, pid, cov, reason = check_claim_default(text, evidence)
        checked.append(Claim(text=text, supported=ok, best_passage=pid, coverage=round(cov, 3), reason=reason))
    parsed.notes.append(f"harness: {sum(1 for c in checked if not c.supported)}/{len(checked)} draft claims unsupported")
    parsed.harness_claims = checked
    parsed.claims = parsed.claims or checked
    return parsed


def render(c: Critique) -> str:
    """The critique as text for the revision prompt."""
    if c.verdict == "ok" and not c.claims:
        return "No issues found."
    lines = [f"verdict: {c.verdict}"]
    for cl in c.claims:
        if cl.supported:
            continue
        bits = [f'- "{cl.text}"']
        if cl.reason:
            bits.append(f"  problem: {cl.reason}")
        if cl.best_passage:
            bits.append(f"  closest passage: [{cl.best_passage}]")
        lines.append("\n".join(bits))
    return "\n".join(lines)
