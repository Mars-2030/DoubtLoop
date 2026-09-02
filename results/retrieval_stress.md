# Retrieval stress

- generated: 2026-09-02  |  backend: `scripted`  |  model: `n/a (scripted stand-in)`
- n = 39 QA items, GroundLoop condition only

| Retrieval | Faithful abstention ↑ | Fabrication ↓ | Accuracy | Hallucination rate |
|---|---|---|---|---|
| normal (unmodified) | 5.6% | 0.0% | 66.7% | 2.6% |
| gold passage removed | 91.7% | 0.0% | 2.8% | 2.6% |
| gold + irrelevant padding | 5.6% | 0.0% | 66.7% | 2.6% |
| search returns nothing | 100.0% | 0.0% | 2.8% | 0.0% |

Rates are over the answerable items only, except the last column.

## How to read it

In the `gold passage removed` and `search returns nothing` rows the
question is still answerable but the evidence the model saw is not.
Faithful abstention should rise toward 100% and fabrication should stay
near zero. A model whose fabrication rate climbs as its evidence
disappears has not learned to check a source; it has learned to cite
whatever it was handed, which is worse than not retrieving at all -
retrieval failures now arrive wearing citations.

The `gold + irrelevant padding` row is the milder question: does
on-topic-looking noise pull the answer off the passage that actually
answers it?

Regenerate with `python -m groundloop.eval.retrieval_stress`.
