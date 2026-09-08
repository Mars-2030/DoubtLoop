# PLAN

Status key: **done** · *partial* · todo

---

## Thesis

Most self-critique alignment loops — Constitutional AI included — ask a model to
critique its own output using only its own internal beliefs, which means a
confidently wrong model produces a confidently wrong revision. GroundLoop closes
that gap: during critique the model issues a tool call to a retrieval system,
checks its draft against real evidence, and only then revises.

Two threads scaled down into one project:

- **Faithfulness / knowledge-grounded generation** — the dissertation thread
- **Constitutional AI** — self-critique/revision + RLAIF, extended with retrieval

---

## Model

**Primary: Qwen3.5-0.8B-Instruct.** Apache 2.0; dense, so fine-tuning behaves
more predictably than an MoE of the same size; native tool-calling; thinking mode
available for the critique step; 262K context, so draft + evidence + constitution
fit in one pass without truncation games; small enough for LoRA on one consumer
GPU.

**Ablation:** Gemma 3 270M — does the technique survive at 270M, where the model
has almost no prior to override?
**Headroom:** Qwen3.5-2B, same family, same tool-calling design.

Nothing in the code is Qwen-specific beyond one `apply_chat_template` call.

---

## Phase 1 — Baseline & data · **done**

1. **done** — Model access behind a backend interface (`src/groundloop/llm.py`):
   `openai` (vLLM or any OpenAI-compatible server), `transformers` (local
   checkpoint + optional LoRA adapter), `scripted` (deterministic stand-in, no
   weights), `echo` (plumbing tests).
2. **done** — Context-QA dataset: 44 passages, 39 QA pairs, 12 sycophancy
   probes, hand-built rather than sampled from HotpotQA. See "Why a custom
   corpus" below.
3. **done** — Base condition with no critique loop at all
   (`--condition base`), which is the "before anything" number.

## Phase 2 — Research tool + corpus · **done**

4. **done** — `search(query)`: BM25 over the fixed local corpus, deterministic
   tie-breaking, `k` clamped, exposed as an OpenAI-style function schema.
5. **done** — Wired into the loop through the tool-call path, with a parser that
   accepts Qwen `<tool_call>` blocks, native `tool_calls` from the server, bare
   JSON, and `search("...")`.
6. **done** — Zero-shot behaviour is measurable before any fine-tuning:
   `tool_use_quality` reports how often the model emitted a usable call versus
   how often the harness had to build the query for it.

## Phase 3 — Constitution · **done**

7. **done** — Nine principles in three tiers (`constitution.md`), each one
   checkable against the evidence or the surface form of the answer.
8. **done** — Two critique templates in `prompts.py`: `CRITIQUE_PLAIN_USER`
   (constitution only — the vanilla-CAI control) and `CRITIQUE_GROUNDED_USER`
   (constitution + retrieved passages, with the instruction to prefer evidence
   over recollection restated inline). The difference between those two strings
   is the experiment.

## Phase 4 — Grounded self-critique/revision loop · **done**

9. **done** — `draft → claims → search → critique → revise`, one function for
   all three conditions (`loop/pipeline.py`), `--rounds` for iterated critique.
10. **done** — Full trajectories logged as JSONL: draft, tool calls, evidence,
    critique, revision, timings, gold support.
11. **done** — Preference pairs: grounded revision `chosen`, original draft
    `rejected`, both answering one prompt that already contains the evidence —
    otherwise DPO learns "prefer the text with brackets in it".

## Phase 5 — Fine-tuning · *not run (written, dry-run tested, never trained)*

**Everything above Phase 5 is inference-time.** The three conditions are
prompting strategies over frozen weights; nothing in the measured result
involves a parameter update. That is worth stating plainly because the project
description says "aligned", and alignment implies training. What exists is a
method plus the harness that measures it.

The blocker is now measured rather than assumed: on real weights the loop yields
~17% usable self-revisions, about nine records from 51 examples. The default SFT
style is also the wrong target for the observed failure - `distilled` teaches
question -> tool call -> answer, and the model already does both perfectly; the
critique turn, where false approval runs at 53%, only appears in
`--style trajectory`.

12. *partial* — `train/sft.py`: TRL `SFTTrainer` + LoRA, defaults sized for a
    single consumer GPU / free-tier T4. `--dry-run` validates data and config
    without importing torch. **Not yet run against real weights — no GPU in this
    environment.**
13. *partial* — `train/dpo.py`: TRL `DPOTrainer`, `--loss-type` covers the
    DPO/ORPO/IPO fork, `--adapter` merges the SFT adapter first so the reference
    model stays well defined. Same status.

**Open before running:** LoRA rank (16 is a guess, not a finding), whether one
DPO epoch is enough on ~20 pairs, and whether the dataset needs to be an order of
magnitude larger before DPO does anything but overfit. The current data run keeps
~30 SFT records and ~22 pairs from 51 examples; that is a pipeline demonstration,
not a training set.

## Phase 6 — Evaluation · **done**

14. **done** — Three-way hallucination comparison: base vs plain self-critique
    vs GroundLoop, claim-level and answer-level, plus accuracy, correct
    abstention on unanswerable items, and over-abstention on answerable ones.
    All three conditions scored against the same evidence pool.
15. **done** — Sycophancy probe: 12 wrong-premise prompts, pushback vs
    capitulation, where pushback requires the correction to actually be stated.
16. **done** — Tool-use quality: well-formed call rate, harness-fallback rate,
    gold-passage recall, citation rate, grounding against what was actually
    retrieved.

### Phase 6b — Robustness (was "next", now done)

- **done** — `eval/retrieval_stress.py`: gold removed, irrelevant padding, and
  empty results, applied *after* the model issues its query. The number that
  matters is faithful abstention when the evidence cannot support an answer.
  This found a design bug: on empty retrieval the loop took the no-evidence
  critique path, so the GroundLoop condition silently became the control
  condition on exactly the questions where the corpus was silent. The condition
  now selects the policy; that row went from 0% abstention / 66.7% fabrication
  to 100% / 0%.
- **done** — `eval/sensitivity.py`: the `tau` sweep (same generations,
  re-scored) and the `k` sweep (re-run). This found a metric bug: the
  contradiction veto lived inside the sentence tier as an early return, so
  raising `tau` let claims fall through to a more permissive tier and the
  hallucination rate *fell* as the metric got stricter. Non-monotone sweeps
  invalidate themselves; there is now a property test pinning it.
- **done** — CI runs the tests on 3.10-3.12 and then runs the whole pipeline,
  failing if the committed tables have drifted from what the code produces.

## Phase 7 — Package · **done**

17. **done** — README leads with the three-way table (labelled with its
    provenance) and three annotated transcripts, regenerated by
    `scripts/make_transcripts.py`.
18. **done** — `demo/app.py`: Gradio, or `--cli`, runnable with no training and
    no weights.
19. **done** — Limitations section tying the metric's failure modes back to the
    faithfulness thread. The write-up proper is still to write.

---

## Decisions made, and why

**Corpus: hand-built, not a HotpotQA subset.** A sampled subset gives you
realistic questions and no control over the one thing the project is about. The
fictional slice (an invented space station, every passage labelled) makes
"unsupported" unambiguous — the model has no prior about KS-9, so an invented
specific is *definitely* invented — and lets evidence-vs-prior conflicts be
constructed (a 31-hour day, 0.38 g) without writing down a false statement about
the real world. The real-world slice is there so the method is not rewarded for
simply deleting claims. Swapping in HotpotQA means keeping `{id, title, text}`;
nothing downstream would notice.

**Thinking mode during critique: available, off by default.** `--thinking` on the
transformers backend. Deliberation should matter more at critique time than at
draft time, but turning it on changes token budget and latency, so it belongs in
the ablation table rather than in the default path.

**Scoring: lexical by default, LLM judge optional.** The default scorer is
IDF-weighted coverage with numeric and polarity vetoes, tiered
sentence → passage → union with rising thresholds. Deterministic and auditable,
which is what a metric spanning three conditions needs. Its blind spot is
paraphrase and word-order ("both of them in Physics" versus a passage naming
Physics and Chemistry). `--judge` swaps in an LLM over the same interface.

---

## Next, in order of what would change a conclusion

1. **Run Phase 5 on real weights** and re-run `make eval-model` before and after.
   Until then the headline table demonstrates a harness, not a method. This is
   the only item on this list that the current environment cannot do.
2. **Report both scorers.** `--judge` exists but has never been run against a
   real judge model. If the three-way ordering only holds under the lexical
   check, that is a finding about the metric, and it should be in the README
   rather than in this file.
3. **Conflicting evidence.** `make stress` now covers retrieval that is missing,
   empty, or padded with noise. It does not cover retrieval that is confidently
   *wrong*: two passages that disagree, or one that states a plausible
   falsehood. That is the sharpest test of P4 and it needs two things this repo
   does not have — a conflicting corpus slice, and a constitution principle
   about what to do when sources disagree (report the conflict; do not silently
   pick). Adding the slice without adding the principle would just measure
   which passage BM25 ranked first.
4. **Scale the eval set** past the point where one item moves a percentage point.
   At n = 39 the difference between 2.6% and 5.1% is one example.
5. **Iterated rounds.** `--rounds` exists; whether a second critique pass helps
   or just adds hedging is unmeasured.
6. **The over-abstention trade.** GroundLoop's abstention gain has to be weighed
   against answerable questions it declines. The `k` sweep is the sharpest view
   of this: k=1 gives perfect abstention on unanswerable items and declines
   14% of answerable ones. The right operating point is a choice about which
   error is cheaper, and it has not been made.
