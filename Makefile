PYTHON ?= python3
export PYTHONPATH := src

.DEFAULT_GOAL := help
.PHONY: help test smoke eval eval-model stress sensitivity results check-results compare demo data sft dpo search clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

test:  ## run the test suite (no GPU, no network, no weights)
	$(PYTHON) -m pytest -q

smoke:  ## run one question through all three conditions
	$(PYTHON) demo/app.py --cli "How many hours are there in a KS-9 station day?"

eval:  ## regenerate results/comparison_table.md with the scripted stand-in
	$(PYTHON) -m groundloop.eval.run_all

eval-model:  ## same, against a served model: make eval-model MODEL=Qwen/Qwen3.5-0.8B-Instruct
	$(PYTHON) -m groundloop.eval.run_all --backend openai --model $(MODEL)

stress:  ## how the loop behaves when retrieval is missing, empty, or noisy
	$(PYTHON) -m groundloop.eval.retrieval_stress

sensitivity:  ## does the result survive the threshold and k sweeps?
	$(PYTHON) -m groundloop.eval.sensitivity

results: eval stress sensitivity  ## regenerate every table and transcript in results/
	$(PYTHON) scripts/make_transcripts.py

check-results:  ## fail if committed results/ differs from a fresh run
	$(PYTHON) scripts/check_results_drift.py

compare:  ## across models: make compare RUNS="a/metrics.json b/metrics.json"
	$(PYTHON) scripts/compare_models.py $(RUNS) --out results/model_comparison.md

demo:  ## launch the Gradio demo (pip install gradio)
	$(PYTHON) demo/app.py

data:  ## build SFT trajectories and preference pairs into data/generated/
	$(PYTHON) -m groundloop.data_gen.build_trajectories

sft:  ## LoRA SFT on the generated trajectories (needs requirements-train.txt)
	$(PYTHON) -m groundloop.train.sft

dpo:  ## DPO on the generated preference pairs
	$(PYTHON) -m groundloop.train.dpo --adapter outputs/sft-lora

search:  ## query the corpus: make search Q="station day"
	$(PYTHON) -m groundloop.tools.search $(Q)

clean:  ## remove caches and generated artefacts
	rm -rf .pytest_cache **/__pycache__ data/generated results/raw
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
