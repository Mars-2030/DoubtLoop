# Sensitivity

- generated: 2026-09-08  |  backend: `scripted`  |  model: `n/a (scripted stand-in)`
- n = 39 QA items  |  defaults in use: tau = 0.7, k = 4

## Support threshold (same generations, re-scored)

Hallucination rate at each value of `SUPPORT_TAU_SENTENCE`. Nothing was
re-generated; only the bar for calling a claim supported moved.

| tau | base | plain critique | GroundLoop | gap (plain − GroundLoop) |
|---|---|---|---|---|
| 0.50 | 61.5% | 61.5% | 2.6% | 58.9 pts |
| 0.60 | 64.1% | 64.1% | 2.6% | 61.5 pts |
| 0.70 ←default | 66.7% | 66.7% | 2.6% | 64.1 pts |
| 0.80 | 66.7% | 66.7% | 2.6% | 64.1 pts |
| 0.90 | 66.7% | 66.7% | 2.6% | 64.1 pts |

A metric doing its job gets stricter with tau in every column at once.
What matters is the last column: if the gap survives the whole sweep,
the effect is not an artefact of where the bar was set. If it collapses
at one end, say so - that is the honest headline, not the default row.

## Retrieval depth (GroundLoop re-run at each k)

| k | hallucination rate | accuracy | over-abstention | correct abstention |
|---|---|---|---|---|
| 1 | 0.0% | 64.1% | 11.1% | 100.0% |
| 2 | 2.6% | 64.1% | 8.3% | 66.7% |
| 4 ←default | 2.6% | 66.7% | 5.6% | 66.7% |
| 8 | 2.6% | 66.7% | 5.6% | 66.7% |

k trades two failures against each other: too small and the gold passage
is missed, so the model correctly abstains on questions it could have
answered; too large and the answer has more irrelevant text to be pulled
off by. The right k is a choice about which error is cheaper, not a
number to maximise.

Regenerate with `python -m groundloop.eval.sensitivity`.
