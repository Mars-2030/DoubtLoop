# Does the method work at other sizes?

**Transcribed from Colab runs on 2026-09-08.** Same 39 questions, same 12
sycophancy probes, same corpus, same prompts, same scorer — only the model
changes. Every condition is inference-time scaffolding over frozen weights;
nothing is fine-tuned. Regenerate with:

```bash
python scripts/compare_models.py results-0.6b/metrics.json results-1.7b/metrics.json \
    --out results/model_comparison.md
```

## Outcome

| Model | Accuracy (base → GroundLoop) | p | Unsupported claims | Sycophancy pushback |
|---|---|---|---|---|
| `Qwen3-0.6B` | 10.3% → 28.2% | ≈0.02 | 87.5% → 74.1% | 8.3% → 25.0% |
| `Qwen3-1.7B` | 30.8% → **64.1%** | **0.0002** | 89.9% → **50.7%** | 25.0% → **83.3%** |

## Why it works, or does not

| Model | Well-formed tool call | Gold passage recall | Critique false approval | Waved through a bad draft |
|---|---|---|---|---|
| `Qwen3-0.6B` | 100% | 100% | **53.3%** | 52.9% |
| `Qwen3-1.7B` | 100% | 100% | **0.0%** | 0.0% |

That is the whole finding in one row pair. **Retrieval is saturated at both
sizes** — 0.6B already calls the tool correctly every time and gets the gold
passage every time, so retrieval explains none of the difference. What changes is
the critique: at 0.6B the model approves 53% of the claims the evidence does not
support; at 1.7B it approves none of them.

The 0% is not the degenerate case of rejecting everything. Of 38 claims judged at
1.7B, 32 agreed with the evidence check and the 6 disagreements were all in the
cautious direction (the model rejected a claim the check supported). Every claim
the evidence rejected, the model rejected too.

## The control gets worse with scale

| Model | Base | Plain self-critique | Change |
|---|---|---|---|
| `Qwen3-0.6B` hallucination | 87.2% | 89.7% | +2.5 |
| `Qwen3-1.7B` hallucination | 87.2% | 92.3% | +5.1 |
| `Qwen3-1.7B` accuracy | 30.8% | 25.6% | −5.2 |

Self-critique without evidence does not merely fail to help — it hurts, and it
hurts *more* at 1.7B than at 0.6B. A more capable model asked to audit its own
answer with nothing to check it against produces a more convincing wrong
critique and talks itself into a worse revision. That is the mechanism the whole
project is aimed at, showing up in the control column.

## Read answer-level hallucination carefully

GroundLoop answers at 1.7B carry 3.59 claims each against 1.77 for the base
condition — the model quotes and cites much more. Answer-level hallucination
counts an answer as hallucinated if *any* claim is unsupported, so a longer
answer faces a strictly harder conjunctive bar.

If claims failed independently, 3.59 claims at 50.7% unsupported would make a
fully clean answer *less* likely at 1.7B (7.9%) than at 0.6B (15.5%). Observed
answer-level hallucination improves anyway, 87.2% → 76.9%, which means the
failures are correlated: answers tend to be grounded or not, rather than mostly
grounded with a scattered bad claim.

The per-claim rate is the fairer cross-size comparison, and it nearly halves:
89.9% → 50.7%.

## Still broken at both sizes

Correct abstention on unanswerable questions is **0.0% at 0.6B and 0.0% at
1.7B**. Neither model ever says "the corpus does not answer this", whatever the
constitution's P5 asks for and whatever the prompt says. Scale did not touch it.
Whether that is the prompt, the lack of any abstention example anywhere in the
context, or something about instruct-tuning is untested and is the most
interesting thing this table does not explain.
