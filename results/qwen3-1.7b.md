# Qwen3-1.7B — real-weights run

**Transcribed from a Colab run on 2026-09-08.** Frozen weights; every condition
is inference-time scaffolding. Reproduce with:

```bash
python -m groundloop.eval.run_all --backend transformers \
    --model Qwen/Qwen3-1.7B --dtype float16 --out-dir results-1.7b
```

- n = 39 QA items, 12 sycophancy probes · top-k 4 · 1 critique round
- scorer: lexical (deterministic)

## Headline

| Setting | Hallucination rate | Unsupported claims | Accuracy | Sycophancy pushback |
|---|---|---|---|---|
| Base model, no critique | 87.2% | 89.9% | 30.8% | 25.0% |
| + plain self-critique (constitution only) | 92.3% | 93.0% | 25.6% | 25.0% |
| + GroundLoop (constitution + retrieval) | 76.9% | 50.7% | 64.1% | 83.3% |

## Full QA metrics

| Metric | Base | Plain critique | GroundLoop |
|---|---|---|---|
| Hallucination rate (answer-level) | 87.2% | 92.3% | 76.9% |
| Unsupported claim rate | 89.9% | 93.0% | 50.7% |
| Accuracy (all items) | 30.8% | 25.6% | 64.1% |
| Accuracy (answerable only) | 33.3% | 27.8% | 69.4% |
| Correct abstention (unanswerable) | 0.0% | 0.0% | 0.0% |
| Over-abstention (answerable) | 0.0% | 0.0% | 0.0% |
| Cites a gold passage | 0.0% | 0.0% | 84.6% |
| Claims per answer | 1.77 | 1.82 | 3.59 |

## Sycophancy probe

| Metric | Base | Plain critique | GroundLoop |
|---|---|---|---|
| Pushback rate (states the correction) | 25.0% | 25.0% | 83.3% |
| Capitulation rate (accepts the false premise) | 16.7% | 16.7% | 0.0% |
| Correction present anywhere | 25.0% | 25.0% | 91.7% |

## Tool use (GroundLoop condition)

| Metric | Value |
|---|---|
| Well-formed model-issued call | 100.0% |
| Harness had to build the query | 0.0% |
| Retrieved at least one gold passage | 100.0% |
| Gold passage recall | 100.0% |
| Final answer cites retrieved evidence | 94.9% |
| Claims supported by what was retrieved | 53.6% |

## Self-critique vs the evidence

| Metric | Value |
|---|---|
| Claims judged by both | 38 |
| Agreement with the evidence check | 84.2% |
| Approved what the evidence does not support | 0.0% |
| Rejected what the evidence does support | 15.8% |
| Waved through a draft with a real error | 0.0% |

## Is the gap real?

| Metric | Base | Plain critique | GroundLoop | vs base | vs plain critique |
|---|---|---|---|---|---|
| hallucination rate | 87.2% [76.9–97.4] | 92.3% [84.6–100.0] | 76.9% [61.5–89.7] | 4 better / 0 worse, p=0.125 | 6 better / 0 worse, p=0.0312 |
| accuracy | 30.8% [17.9–46.2] | 25.6% [12.8–41.0] | 64.1% [48.7–79.5] | 13 better / 0 worse, p=0.0002 | 15 better / 0 worse, p=0.0001 |
| sycophancy pushback | 25.0% [0.0–50.0] | 25.0% [0.0–50.0] | 83.3% [58.3–100.0] | 7 better / 0 worse, p=0.0156 | 7 better / 0 worse, p=0.0156 |

Accuracy and pushback are results. Answer-level hallucination against the base
condition is a direction (p=0.125) — see the note on answer length in
[`model_comparison.md`](model_comparison.md).
