"""Smoke tests for the command-line entry points, and one harness regression.

These run against the scripted backend, so they check the *plumbing*, not a
model. The ordering assertion below is deliberately weak for that reason: it
guards against a refactor that silently stops the retrieval step from mattering,
and it is not evidence about any real model.
"""

import json

import pytest

from groundloop.datasets import load_qa, load_sycophancy
from groundloop.eval import hallucination_rate as hall
from groundloop.eval import run_all
from groundloop.eval import sycophancy_probe as syco
from groundloop.eval import tool_use_quality as tq
from groundloop.llm import get_backend
from groundloop.loop.pipeline import CONDITIONS, run_dataset


@pytest.fixture(scope="module")
def runs():
    llm = get_backend("scripted")
    from groundloop.tools.search import SearchTool

    tool = SearchTool()
    qa = load_qa()
    probe = load_sycophancy()
    return {
        "tool": tool,
        "qa": qa,
        "probe": probe,
        "qa_trajs": {c: [t.to_json() for t in run_dataset(llm, qa, c, tool=tool)] for c in CONDITIONS},
        "probe_trajs": {c: [t.to_json() for t in run_dataset(llm, probe, c, tool=tool)] for c in CONDITIONS},
    }


def test_retrieval_beats_critique_alone_on_the_harness(runs):
    summaries = {
        c: hall.score_trajectories(runs["qa_trajs"][c], runs["qa"], runs["tool"])[1]
        for c in CONDITIONS
    }
    assert summaries["groundloop"]["hallucination_rate"] < summaries["plain_critique"]["hallucination_rate"]
    assert summaries["groundloop"]["accuracy"] > summaries["base"]["accuracy"]


def test_only_groundloop_ever_abstains_correctly(runs):
    summaries = {
        c: hall.score_trajectories(runs["qa_trajs"][c], runs["qa"], runs["tool"])[1]
        for c in CONDITIONS
    }
    assert summaries["groundloop"]["correct_abstention_rate"] > 0


def test_sycophancy_pushback_requires_evidence(runs):
    scored = {
        c: syco.score_trajectories(runs["probe_trajs"][c], runs["probe"])[1] for c in CONDITIONS
    }
    assert scored["base"]["pushback_rate"] == 0.0
    assert scored["groundloop"]["pushback_rate"] > scored["plain_critique"]["pushback_rate"]


def test_tool_use_metrics_are_populated(runs):
    summary = tq.score_trajectories(runs["qa_trajs"]["groundloop"], runs["qa"])
    assert summary["n"] == len(runs["qa"])
    assert summary["gold_passage_hit_rate"] > 50.0
    assert 0.0 <= summary["harness_fallback_rate"] <= 100.0


def test_run_all_writes_a_table_that_declares_its_provenance(tmp_path, capsys):
    rc = run_all.main(["--limit", "4", "--out-dir", str(tmp_path), "--no-save-trajectories"])
    assert rc == 0
    table = (tmp_path / "comparison_table.md").read_text()
    assert "not a model result" in table, "a scripted-backend table must say so"
    assert "| Base model, no critique |" in table
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert set(payload["conditions"]) == set(CONDITIONS)
    assert payload["meta"]["backend"] == "scripted"


def test_hallucination_cli(capsys):
    assert hall.main(["--condition", "base", "--limit", "3", "--json"]) == 0
    assert "hallucination_rate" in json.loads(capsys.readouterr().out)


def test_sycophancy_cli(capsys):
    assert syco.main(["--condition", "groundloop", "--limit", "3", "--json"]) == 0
    assert "pushback_rate" in json.loads(capsys.readouterr().out)


def test_demo_cli_runs_all_three_conditions(capsys):
    import runpy
    import sys
    from pathlib import Path

    demo = Path(__file__).resolve().parents[1] / "demo" / "app.py"
    argv = sys.argv
    sys.argv = [str(demo), "--cli", "How many hours are there in a KS-9 station day?"]
    try:
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(demo), run_name="__main__")
        assert exc.value.code == 0
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    for condition in CONDITIONS:
        assert condition in out
