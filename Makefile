PYTHON ?= python3
export PYTHONPATH := src

.DEFAULT_GOAL := help
.PHONY: help test smoke eval eval-model demo data sft dpo search clean

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
