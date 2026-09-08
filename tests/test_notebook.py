"""The Colab notebook is a deliverable, so it gets checked like one.

It cannot be executed here (no GPU, and huggingface.co is not reachable), but
the failure that actually bit — `python -m groundloop...` in a subprocess that
never had the package installed — is a structural property of the notebook and
is checkable without running anything.
"""

import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "groundloop_colab.ipynb"


@pytest.fixture(scope="module")
def nb():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(nb):
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _runnable(cell) -> str:
    """A cell's source with comment lines dropped.

    Needed because the install cell *explains* `!python -m groundloop...` in a
    comment, and a naive substring search then reads that prose as the first
    use of the package.
    """
    return "\n".join(
        line for line in "".join(cell["source"]).splitlines()
        if not line.lstrip().startswith("#")
    )


def test_notebook_is_valid_json_with_cells(nb):
    assert nb["nbformat"] == 4 and nb["cells"]


def test_it_asks_for_a_gpu_runtime(nb):
    assert nb["metadata"].get("accelerator") == "GPU"


def test_the_package_is_installed_before_any_module_is_run(nb):
    """Regression: `make` exports PYTHONPATH=src, but the `!python -m` cells are
    separate subprocesses and saw no package at all."""
    sources = [_runnable(c) for c in _code_cells(nb)]
    install = next((i for i, s in enumerate(sources) if "pip install -q -e ." in s), None)
    assert install is not None, "the notebook never installs the package"

    first_use = next((i for i, s in enumerate(sources) if "python -m groundloop" in s), None)
    assert first_use is not None, "the notebook never runs the package"
    assert install <= first_use, "the package is installed after it is first used"


def test_the_install_is_verified_rather_than_assumed(nb):
    sources = "".join(_runnable(c) for c in _code_cells(nb))
    assert "import groundloop" in sources, "nothing checks that the install took"


def test_precision_is_chosen_from_the_hardware(nb):
    # T4 is Turing and has no usable bf16; hardcoding either value gets someone
    # a silent fp32 run or an emulated bf16 one.
    sources = "".join("".join(c["source"]) for c in _code_cells(nb))
    assert "get_device_capability" in sources
    assert "--fp16" in sources and "--dtype" in sources


def test_the_model_id_is_resolved_not_assumed(nb):
    # The target id in PLAN.md has never been verified against the Hub.
    sources = "".join("".join(c["source"]) for c in _code_cells(nb))
    assert "FALLBACKS" in sources and "model_info" in sources


def test_troubleshooting_covers_the_import_error(nb):
    prose = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "No module named 'groundloop'" in prose


def test_the_colab_torchao_conflict_is_handled(nb):
    # peft raises on torchao < 0.16 while probing for it, and Colab preinstalls
    # 0.10, so the SFT cell dies unless setup removes it.
    setup = "".join("".join(c["source"]) for c in _code_cells(nb))
    assert "uninstall -y -q torchao" in setup
    prose = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "torchao" in prose, "troubleshooting does not mention the error"


def test_low_yield_is_explained_before_training(nb):
    prose = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "yield" in prose.lower()
    assert "--no-require-correct" in prose and "--tau" in prose


class TestModelComparison:
    """Comparing models is the experiment that turns one table into a claim
    about where the method starts working - so it must not silently compare
    runs that are not comparable."""

    @staticmethod
    def _module():
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "scripts" / "compare_models.py"
        spec = importlib.util.spec_from_file_location("compare_models", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _metrics(model, n, base_acc, ground_acc, false_approval):
        return {
            "meta": {"model": model, "backend": "transformers", "n_qa": n},
            "conditions": {
                "base": {"qa": {"accuracy": base_acc, "unsupported_claim_rate": 90.0},
                         "sycophancy": {"pushback_rate": 8.3}},
                "plain_critique": {"qa": {"accuracy": base_acc},
                                   "sycophancy": {"pushback_rate": 8.3}},
                "groundloop": {
                    "qa": {"accuracy": ground_acc, "unsupported_claim_rate": 74.0},
                    "sycophancy": {"pushback_rate": 25.0},
                    "tool_use": {"well_formed_call_rate": 100.0, "gold_passage_recall": 100.0},
                    "critique": {"false_approval_rate": false_approval, "blind_critique_rate": 52.9},
                },
            },
            "significance": [{"attribute": "correct", "vs_base": {"p_value": 0.016}}],
        }

    def test_rows_carry_the_numbers_that_explain_the_outcome(self, tmp_path):
        mod = self._module()
        import json

        path = tmp_path / "metrics.json"
        path.write_text(json.dumps(self._metrics("Qwen/Qwen3-0.6B", 39, 10.3, 28.2, 53.3)))
        table = mod.render([mod.load(path)])
        assert "10.3% → 28.2%" in table
        assert "53.3%" in table, "false approval is the column the experiment turns on"
        assert "100.0%" in table, "retrieval has to be visible to rule it out"

    def test_mismatched_question_sets_are_flagged(self, tmp_path):
        mod = self._module()
        import json

        a, b = tmp_path / "a.json", tmp_path / "b.json"
        a.write_text(json.dumps(self._metrics("small", 39, 10.3, 28.2, 53.3)))
        b.write_text(json.dumps(self._metrics("large", 12, 20.0, 50.0, 20.0)))
        table = mod.render([mod.load(a), mod.load(b)])
        assert "not comparable" in table

    def test_matching_question_sets_are_not_flagged(self, tmp_path):
        mod = self._module()
        import json

        a, b = tmp_path / "a.json", tmp_path / "b.json"
        a.write_text(json.dumps(self._metrics("small", 39, 10.3, 28.2, 53.3)))
        b.write_text(json.dumps(self._metrics("large", 39, 20.0, 50.0, 20.0)))
        assert "not comparable" not in mod.render([mod.load(a), mod.load(b)])

    def test_a_missing_file_is_explained(self, tmp_path):
        mod = self._module()

        with pytest.raises(SystemExit, match="not found"):
            mod.main([str(tmp_path / "nope.json")])
