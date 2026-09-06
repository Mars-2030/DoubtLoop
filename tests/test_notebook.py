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
