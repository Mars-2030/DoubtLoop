"""Paths and tunable constants, in one place so experiments stay reproducible."""

from __future__ import annotations

import os
from pathlib import Path

# --- paths -----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("GROUNDLOOP_DATA", REPO_ROOT / "data"))
CORPUS_PATH = DATA_DIR / "qa_corpus" / "passages.jsonl"
QA_PAIRS_PATH = DATA_DIR / "qa_pairs.jsonl"
SYCOPHANCY_PATH = DATA_DIR / "sycophancy_probe.jsonl"

CONSTITUTION_PATH = REPO_ROOT / "constitution.md"
RESULTS_DIR = Path(os.environ.get("GROUNDLOOP_RESULTS", REPO_ROOT / "results"))
GENERATED_DIR = DATA_DIR / "generated"

# --- retrieval -------------------------------------------------------------

TOP_K = 4
BM25_K1 = 1.5
BM25_B = 0.75

# --- grounding metric ------------------------------------------------------

# Three tiers of evidence, three thresholds. A claim is supported when one
# SENTENCE of the evidence covers this (IDF-weighted) fraction of its content
# tokens. 0.70 rather than 0.60 because at 0.60 "Gutenberg invented movable type
# from scratch" scores as supported by a passage saying he introduced a press
# *using* movable type: the two words that carry the falsehood are exactly the
# two the passage does not have.
SUPPORT_TAU_SENTENCE = 0.70
# ...or one whole PASSAGE does (a claim that legitimately spans two sentences)...
SUPPORT_TAU_PASSAGE = 0.80
# ...or the UNION of all retrieved evidence does (genuine multi-hop).
# The thresholds rise as the evidence gets more scattered: "every word of this
# claim appears somewhere in four passages" is how a bag-of-words check talks
# itself into endorsing a fabrication.
SUPPORT_TAU_UNION = 0.90

# --- generation ------------------------------------------------------------

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
SEED = 20260901
