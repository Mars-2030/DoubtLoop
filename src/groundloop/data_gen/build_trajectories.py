"""Phase 4: run the loop and turn what comes out into training data.

Two artefacts, for the two training stages:

* **SFT** (`sft.jsonl`) - conversations that teach the *shape* of the behaviour:
  see a question, call `search`, read what comes back, answer from it with
  citations. Two styles:

  - ``distilled`` (default): question -> tool call -> tool result -> grounded
    answer. This is what you want the deployed model to do. The draft and the
    critique were scaffolding used to *find* the good answer; keeping them in
    the target would train the model to always emit a bad answer first.
  - ``trajectory``: the full draft -> critique -> revision transcript, for
    studying whether the explicit critique turn is itself load-bearing.

* **Preference pairs** (`prefs.jsonl`) - `chosen` = the grounded revision,
  `rejected` = the original draft, for DPO/ORPO.

**Rejection sampling is the point, not a detail.** A trajectory is only kept
when the revision is actually better than the draft by the harness's own
measure: every claim supported, and - for answerable items - the reference
answer's key facts present. Training on unfiltered self-revisions teaches the
model to imitate the *format* of grounding while inheriting whatever the loop
got wrong. `--min-improvement` controls how much better the revision has to be;
the run prints how many examples each filter dropped, because a pipeline that
silently keeps 8 of 39 examples is a pipeline you should know about.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from groundloop import config, prompts
from groundloop.datasets import load_qa, load_sycophancy, write_jsonl
from groundloop.eval.metrics import reference_evidence, score_answer, score_sycophancy
from groundloop.llm import add_backend_args, backend_from_args
from groundloop.loop.pipeline import run_condition
from groundloop.schema import Trajectory
from groundloop.tools.search import SEARCH_TOOL_SPEC, SearchTool


def _tool_call_text(query: str) -> str:
    return "<tool_call>" + json.dumps({"name": "search", "arguments": {"query": query}}) + "</tool_call>"


def _tool_result_text(traj: Trajectory) -> str:
    return json.dumps(
        {"results": [{"id": p.id, "title": p.title, "text": p.text} for p in traj.evidence]},
        ensure_ascii=False,
    )


def _query_of(traj: Trajectory) -> str:
    for call in traj.tool_calls:
        q = call.arguments.get("query")
        if q:
            return str(q)
    return traj.question


def to_sft_record(traj: Trajectory, style: str = "distilled") -> dict:
    """Render a trajectory as a chat conversation for SFT."""
    query = _query_of(traj)
    if style == "distilled":
        messages = [
            {"role": "system", "content": prompts.RESEARCH_SYSTEM},
            {"role": "user", "content": traj.question},
            {"role": "assistant", "content": _tool_call_text(query)},
            {"role": "tool", "name": "search", "content": _tool_result_text(traj)},
            {"role": "assistant", "content": traj.final},
        ]
    elif style == "trajectory":
        critique_text = traj.critique.raw if traj.critique else "{}"
        evidence_block = "\n\n".join(f"[{p.id}] {p.title}\n{p.text}" for p in traj.evidence)
        messages = [
            {"role": "system", "content": prompts.RESEARCH_SYSTEM},
            {"role": "user", "content": traj.question},
            {"role": "assistant", "content": traj.draft},
            {"role": "user", "content": prompts.TOOL_USER.format(question=traj.question, draft=traj.draft)},
            {"role": "assistant", "content": _tool_call_text(query)},
            {"role": "tool", "name": "search", "content": _tool_result_text(traj)},
            {"role": "user", "content": prompts.CRITIQUE_GROUNDED_USER.format(
                constitution=prompts.constitution(), question=traj.question,
                draft=traj.draft, evidence=evidence_block, query=query)},
            {"role": "assistant", "content": critique_text},
            {"role": "user", "content": "Now rewrite the answer so every claim is supported by the evidence."},
            {"role": "assistant", "content": traj.final},
        ]
    else:
        raise ValueError(f"unknown style {style!r}")
    return {
        "id": traj.example_id,
        "style": style,
        "messages": messages,
        "tools": [SEARCH_TOOL_SPEC],
    }


def to_preference_record(traj: Trajectory) -> dict:
    """Chosen = grounded revision, rejected = the model's own first draft.

    Both completions answer the *same* prompt, and that prompt already contains
    the evidence. Without that, DPO would mostly learn "prefer the answer that
    has citations in it", which is a formatting preference wearing a
    faithfulness costume.
    """
    evidence_block = "\n\n".join(f"[{p.id}] {p.title}\n{p.text}" for p in traj.evidence)
    prompt = [
        {"role": "system", "content": prompts.REVISE_SYSTEM},
        {"role": "user", "content": (
            f"Question: {traj.question}\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            "Answer the question using only the evidence above, citing passage ids. "
            "If the evidence does not answer it, say so."
        )},
    ]
    return {
        "id": traj.example_id,
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": traj.final}],
        "rejected": [{"role": "assistant", "content": traj.draft}],
    }


LOW_YIELD_PCT = 25.0


def summarize(payload: dict, selection_tau: float | None = None,
              reporting_tau: float | None = None) -> str:
    """What the run kept, what it dropped, and whether that is a problem.

    Yield is the number worth reading. A real model paraphrases, and the lexical
    support check is at its weakest on paraphrase, so a near-zero yield is as
    likely to be the metric rejecting good revisions as the model producing bad
    ones. Either way, training on what survives is not worth the GPU time until
    you know which.
    """
    n, kept = payload["n"], len(payload["sft"])
    pct = 100.0 * kept / n if n else 0.0
    lines = [
        f"examples run:        {n}",
        f"SFT records kept:    {kept}  ({pct:.0f}% yield)",
        f"preference pairs:    {len(payload['prefs'])}",
    ]
    if selection_tau is not None:
        lines.append(
            f"selection threshold: {selection_tau} "
            f"(reporting stays at {reporting_tau})"
        )
    if payload["drops"]:
        lines.append("dropped:")
        for reason, count in sorted(payload["drops"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  {count:>4}  {reason}")

    if pct < LOW_YIELD_PCT:
        drops = payload.get("drops", {})
        before = len(lines)
        lines += ["", "Yield is low. Read the buckets above before reaching for a flag:"]
        if drops.get("revision is grounded but misses the reference answer keys"):
            lines.append("  --no-require-correct   the revisions ARE grounded; they just miss the "
                         "reference keys")
        if drops.get("revision still asserts unsupported claims"):
            lines.append("  the revisions are not grounded. --tau 0.5 loosens SELECTION only, but "
                         "first")
            lines.append("  run with --show-rejected 5: if a revision quotes the right passage and "
                         "is still")
            lines.append("  rejected, that is the metric; if it repeats the draft, that is the model,")
            lines.append("  and a bigger model for THIS step is the fix rather than a looser filter.")
        if drops.get("probe: revision never states the correction"):
            lines.append("  the sycophancy probes are not being corrected at all - a model-capability")
            lines.append("  signal, not a threshold one.")
        if len(lines) == before + 2:  # no bucket matched a known lever
            lines.append("  run with --show-rejected 5 to see what is actually being thrown away.")
        lines += [
            "Fewer records than your effective batch means the trainer takes one step per",
            "epoch and learns nothing; groundloop.train.sft will refuse rather than pretend.",
        ]
    return "\n".join(lines)


def build(args) -> dict:
    llm = backend_from_args(args)
    tool = SearchTool()
    qa = load_qa(limit=args.limit)
    probe = load_sycophancy(limit=args.limit) if args.include_probe else []

    sft_rows, pref_rows = [], []
    drops: Counter[str] = Counter()
    rejected: list[dict] = []
    n = 0

    for ex in qa + probe:
        n += 1
        if not args.quiet:
            print(f"\r  building: {n}/{len(qa) + len(probe)}", end="", file=sys.stderr, flush=True)
        traj = run_condition(llm, ex, "groundloop", tool=tool, k=args.k, rounds=args.rounds)

        is_probe = "correction_keys" in ex
        if is_probe:
            after = score_sycophancy(ex, traj.final)
            before = score_sycophancy(ex, traj.draft)
            improved = after.pushback and not before.pushback
            good = after.pushback
        else:
            evidence = reference_evidence(ex, tool)
            after = score_answer(ex, traj.final, evidence)
            before = score_answer(ex, traj.draft, evidence)
            improved = (before.n_unsupported - after.n_unsupported) >= args.min_improvement or (
                after.correct and not before.correct
            )
            good = after.n_unsupported == 0 and (after.correct or not args.require_correct)

        if not traj.evidence:
            drops["no evidence retrieved"] += 1
            continue
        if not good:
            # One bucket per cause. "Still asserts things the evidence does not
            # support" and "grounded but does not match the reference answer"
            # are different failures needing different responses, and lumping
            # them together leaves you guessing which lever to pull.
            if is_probe:
                reason = "probe: revision never states the correction"
            elif after.n_unsupported:
                reason = "revision still asserts unsupported claims"
            else:
                reason = "revision is grounded but misses the reference answer keys"
            drops[reason] += 1
            rejected.append({
                "id": ex.get("id", ""),
                "reason": reason,
                "question": _question_text(ex),
                "draft": traj.draft,
                "revision": traj.final,
                "unsupported": list(getattr(after, "unsupported_claims", []))[:3],
                "evidence": [p.id for p in traj.evidence],
                "revision_attempted": traj.meta.get("revision_attempted", False),
                "critique_verdict": traj.meta.get("critique_verdict", ""),
                "critique_issues": traj.meta.get("critique_issues", 0),
            })
            continue
        sft_rows.append(to_sft_record(traj, args.style))

        if traj.final.strip() == traj.draft.strip():
            drops["revision identical to draft (no preference signal)"] += 1
            continue
        if not improved:
            drops["revision not measurably better than draft"] += 1
            continue
        pref_rows.append(to_preference_record(traj))

    if not args.quiet:
        print(file=sys.stderr)
    return {"sft": sft_rows, "prefs": pref_rows, "drops": dict(drops),
            "rejected": rejected, "n": n}


def _question_text(example: dict) -> str:
    return example.get("question") or example.get("prompt") or ""


def render_rejected(rejected: list[dict], limit: int) -> str:
    """Show what a rejected revision actually looked like.

    The counts say which filter fired; only the text says whether it should
    have. A revision that quotes the right passage and is rejected anyway is a
    metric problem; one that repeats the draft verbatim is a model problem.
    """
    lines = ["", "rejected revisions:"]
    for row in rejected[:limit]:
        lines += [
            "",
            f"  [{row['id']}] {row['reason']}",
            f"    Q:        {row['question']}",
            f"    draft:    {row['draft'][:200]}",
            f"    revision: {row['revision'][:200]}",
            f"    evidence: {', '.join(row['evidence'])}",
        ]
        for claim in row["unsupported"]:
            lines.append(f"    unsupported: {claim[:160]}")
        if row["draft"].strip() == row["revision"].strip():
            if row.get("revision_attempted"):
                lines.append("    NOTE: a revision was attempted and came back unchanged (or empty) "
                             "- generation failure")
            else:
                lines.append(f"    NOTE: no revision attempted - the critique returned "
                             f"{row.get('critique_verdict', '?')!r} with "
                             f"{row.get('critique_issues', 0)} issue(s), so the loop had nothing "
                             f"to act on - critique failure")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(config.GENERATED_DIR))
    ap.add_argument("--style", default="distilled", choices=("distilled", "trajectory"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("-k", type=int, default=config.TOP_K)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--include-probe", action="store_true", default=True,
                    help="also build data from the sycophancy prompts (default: on)")
    ap.add_argument("--no-include-probe", dest="include_probe", action="store_false")
    ap.add_argument("--min-improvement", type=int, default=1,
                    help="unsupported claims the revision must remove to count as a preference pair")
    ap.add_argument("--require-correct", action="store_true", default=True,
                    help="keep only revisions that also match the reference answer keys")
    ap.add_argument("--no-require-correct", dest="require_correct", action="store_false")
    ap.add_argument("--tau", type=float, default=None,
                    help="support threshold for SELECTION only (default: the reporting "
                         "threshold from config). Loosen it to keep more trajectories; the "
                         "eval still scores at the reporting threshold, so this trades "
                         "training-data purity for volume rather than flattering the result.")
    ap.add_argument("--show-rejected", type=int, default=0, metavar="N",
                    help="print N rejected revisions in full. The counts say which "
                         "filter fired; only the text says whether it should have.")
    ap.add_argument("--quiet", action="store_true")
    add_backend_args(ap)
    args = ap.parse_args(argv)

    reporting_tau = config.SUPPORT_TAU_SENTENCE
    if args.tau is not None:
        # Selection threshold, deliberately separate from the reporting one. A
        # real model paraphrases, and the lexical check is weakest on paraphrase,
        # so the default bar can reject almost every revision and leave nothing
        # to train on. Loosening selection is legitimate; loosening reporting
        # would just be marking your own homework.
        config.SUPPORT_TAU_SENTENCE = args.tau

    out_dir = Path(args.out_dir)
    payload = build(args)
    write_jsonl(out_dir / "sft.jsonl", payload["sft"])
    write_jsonl(out_dir / "prefs.jsonl", payload["prefs"])

    print(summarize(payload, args.tau, reporting_tau))
    if args.show_rejected and payload["rejected"]:
        print(render_rejected(payload["rejected"], args.show_rejected))
    print(f"\nwrote {out_dir / 'sft.jsonl'} and {out_dir / 'prefs.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
