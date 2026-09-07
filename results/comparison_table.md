# GroundLoop: three-way comparison

> **These numbers are not a model result.** They were produced with
> `--backend scripted`, the deterministic stand-in that lets the
> pipeline run with no weights. They demonstrate that the harness
> measures what it claims to measure. Re-run with `--backend openai`
> or `--backend transformers` against real weights to get a result.

- generated: 2026-09-07  (groundloop 0.1.0)
- backend: `scripted`  |  model: `n/a (scripted stand-in)`
- scorer: lexical (deterministic)  |  top-k: 4  |  critique rounds: 1
- n = 39 QA items, 12 sycophancy probes

## Headline

| Setting | Hallucination rate | Unsupported claims | Accuracy | Sycophancy pushback |
|---|---|---|---|---|
| Base model, no critique | 66.7% | 66.7% | 25.6% | 0.0% |
| + plain self-critique (constitution only) | 66.7% | 66.7% | 28.2% | 0.0% |
| + GroundLoop (constitution + retrieval) | 2.6% | 0.0% | 66.7% | 50.0% |

Lower is better for the first two columns; higher is better for the last two.

## Full QA metrics

| Metric | Base model, no critique | + plain self-critique (constitution only) | + GroundLoop (constitution + retrieval) |
|---|---|---|---|
| Hallucination rate (answer-level) | 66.7% | 66.7% | 2.6% |
| Unsupported claim rate | 66.7% | 66.7% | 0.0% |
| Accuracy (all items) | 25.6% | 28.2% | 66.7% |
| Accuracy (answerable only) | 27.8% | 30.6% | 66.7% |
| Correct abstention (unanswerable) | 0.0% | 0.0% | 66.7% |
| Over-abstention (answerable) | 0.0% | 0.0% | 5.6% |
| Cites a gold passage | 0.0% | 0.0% | 53.8% |
| Claims per answer | 1.0 | 1.0 | 1.46 |

## Sycophancy probe

| Metric | Base model, no critique | + plain self-critique (constitution only) | + GroundLoop (constitution + retrieval) |
|---|---|---|---|
| Pushback rate (states the correction) | 0.0% | 0.0% | 50.0% |
| Capitulation rate (accepts the false premise) | 100.0% | 8.3% | 25.0% |
| Correction present anywhere in the answer | 0.0% | 0.0% | 50.0% |

## Tool use (GroundLoop condition only)

| Metric | Value |
|---|---|
| Search calls per question | 1.0 |
| Well-formed model-issued call | 100.0% |
| Harness had to build the query | 0.0% |
| No call attempted | 0.0% |
| Search returned nothing | 0.0% |
| Retrieved at least one gold passage | 100.0% |
| Gold passage recall | 97.7% |
| Final answer cites retrieved evidence | 56.4% |
| Claims supported by what was retrieved | 100.0% |

## Self-critique vs the evidence (GroundLoop condition)

Whether the critique step agrees with the passages it was shown. If it
endorses whatever the draft said, retrieval and revision are both wasted.

| Metric | Value |
|---|---|
| Claims judged by both | 39 |
| Agreement with the evidence check | 100.0% |
| Approved what the evidence does not support | 0.0% |
| Rejected what the evidence does support | 0.0% |
| Waved through a draft with a real error | 0.0% |

Agreement with the lexical check, not with ground truth, so
false approval is an upper bound on the model's error and false alarm a
lower bound. With the scripted stand-in these are trivially 100% / 0%:
its critique *is* the lexical check, so it cannot disagree with itself.

## Reading this table

The middle row is the control that makes the experiment an experiment.
Any gap between the base model and plain self-critique is what the model
gains from *thinking harder* about its own answer. The gap between plain
self-critique and GroundLoop is what it gains from *checking a source*.
If the second gap were small, the retrieval step would not be earning its
cost and the honest conclusion would be that vanilla CAI is enough.

Regenerate with `make eval` (or `python -m groundloop.eval.run_all`).
