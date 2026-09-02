"""Retrieval stress and metric sensitivity.

Both files exist to answer "is the headline an artefact?", so they get tests
that check the *properties* the answer depends on, not just that they run.
"""

import pytest

from groundloop import config
from groundloop.datasets import load_qa
from groundloop.eval import sensitivity
from groundloop.eval.retrieval_stress import MODES, StressTool, run_mode
from groundloop.loop.pipeline import CONDITIONS, run_condition, run_dataset
from groundloop.textutil import check_claim


class TestStressTool:
    def test_unknown_mode_is_rejected(self, tool):
        with pytest.raises(ValueError):
            StressTool(tool, "sabotage")

    def test_no_gold_removes_exactly_the_gold_passages(self, tool):
        stress = StressTool(tool, "no_gold")
        stress.gold = {"f03"}
        got = {p.id for p in stress.search("how long is a KS-9 station day", 4)}
        assert "f03" not in got and got

    def test_empty_returns_nothing(self, tool):
        stress = StressTool(tool, "empty")
        assert stress.search("Apollo 11", 4) == []
        assert stress.call({"query": "Apollo 11"})["results"] == []

    def test_distractor_keeps_gold_and_adds_other_world_noise(self, tool):
        stress = StressTool(tool, "distractor", n_distractors=2)
        stress.gold = {"r01"}
        got = stress.search("who stayed in lunar orbit apollo 11", 2)
        ids = {p.id for p in got}
        assert "r01" in ids
        assert any(p.world == "fictional" for p in got), "no distractors were injected"

    def test_normal_is_a_passthrough(self, tool):
        stress = StressTool(tool, "normal")
        assert [p.id for p in stress.search("penicillin", 3)] == [p.id for p in tool.search("penicillin", 3)]


class TestBehaviourUnderBadRetrieval:
    def test_empty_retrieval_abstains_rather_than_falling_back_to_the_draft(self, llm, tool):
        # Regression: when search came back empty the loop routed the critique
        # down the no-evidence path, so the GroundLoop condition silently became
        # the plain-critique condition on exactly the questions where the corpus
        # had nothing to say - and answered from its prior instead of abstaining.
        stress = StressTool(tool, "empty")
        traj = run_condition(llm, {"id": "x", "question": "How tall is Mount Everest?"},
                             "groundloop", tool=stress)
        assert not traj.evidence
        assert "does not answer" in traj.final.lower()
        assert traj.final != traj.draft

    def test_removing_the_gold_passage_raises_abstention_not_fabrication(self, llm, tool):
        examples = load_qa()[:12]
        _, normal = run_mode(llm, tool, examples, "normal", config.TOP_K)
        _, no_gold = run_mode(llm, tool, examples, "no_gold", config.TOP_K)
        assert no_gold["faithful_abstention_rate"] > normal["faithful_abstention_rate"]
        assert no_gold["fabrication_rate"] <= normal["fabrication_rate"] + 10.0

    @pytest.mark.parametrize("mode", MODES)
    def test_every_mode_produces_answers(self, llm, tool, mode):
        trajs, summary = run_mode(llm, tool, load_qa()[:4], mode, config.TOP_K)
        assert len(trajs) == 4 and summary["mode"] == mode


class TestMetricMonotonicity:
    CLAIMS = [
        ("The KS-9 station day is 31 standard hours long.",
         [("f03", "The KS-9 station day is 31 standard hours long. Crew schedules are keyed to the 31-hour day.")]),
        ("A KS-9 station day is 24 hours long.",
         [("f03", "The KS-9 station day is 31 standard hours long.")]),
        ("Gutenberg invented movable type from scratch in Mainz in the 1440s.",
         [("r12", "Johannes Gutenberg introduced a printing press using cast metal movable type in Mainz around the 1440s.")]),
    ]

    @pytest.mark.parametrize("claim,evidence", CLAIMS)
    def test_raising_the_threshold_never_turns_unsupported_into_supported(self, claim, evidence):
        # The bug this guards: the contradiction veto used to live inside the
        # sentence tier as an early return, so a higher threshold meant that
        # tier stopped firing and the claim fell through to a *more* permissive
        # one. The measured hallucination rate then fell as the metric got
        # stricter, which quietly invalidates any threshold sweep.
        verdicts = [check_claim(claim, evidence, tau_sentence=t)[0] for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
        for earlier, later in zip(verdicts, verdicts[1:]):
            assert not (later and not earlier), f"support reappeared for {claim!r}"

    def test_clean_single_sentence_support_beats_a_partial_match_elsewhere(self):
        # "keyed to the 31-hour day" covers much of the claim and never says
        # KS-9, so read alone it looks like a numeric disagreement.
        evidence = [("f03", "The KS-9 station day is 31 standard hours long. "
                            "Crew schedules are all keyed to the 31-hour day.")]
        assert check_claim("The KS-9 station day is 31 standard hours long.", evidence)[0]


class TestSensitivitySweep:
    def test_tau_sweep_is_monotone_and_leaves_config_untouched(self, llm, tool):
        examples = load_qa()[:10]
        trajectories = {
            c: [t.to_json() for t in run_dataset(llm, examples, c, tool=tool)] for c in CONDITIONS
        }
        before = config.SUPPORT_TAU_SENTENCE
        rows = sensitivity.sweep_tau(trajectories, examples, tool, [0.5, 0.7, 0.9])
        assert config.SUPPORT_TAU_SENTENCE == before, "the sweep leaked its threshold into the config"
        for condition in CONDITIONS:
            series = [row[condition] for row in rows]
            assert series == sorted(series), f"{condition} hallucination rate fell as tau rose: {series}"

    def test_k_sweep_reruns_and_reports(self, llm, tool):
        rows = sensitivity.sweep_k(llm, load_qa()[:6], tool, [1, 4])
        assert [r["k"] for r in rows] == [1, 4]
        assert all("hallucination_rate" in r for r in rows)


class TestResultsDriftCheck:
    """The drift checker is what keeps the README's numbers honest, so it gets
    a test of its own: it has to ignore the timestamp and nothing else."""

    @staticmethod
    def _normalise():
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "scripts" / "check_results_drift.py"
        spec = importlib.util.spec_from_file_location("check_results_drift", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.normalise

    def test_generation_date_is_ignored(self):
        normalise = self._normalise()
        a = "# T\n- generated: 2026-01-01  |  backend: `scripted`\n| base | 74.4% |"
        b = "# T\n- generated: 2099-12-31  |  backend: `scripted`\n| base | 74.4% |"
        assert normalise(a) == normalise(b)

    def test_a_changed_number_is_not_ignored(self):
        normalise = self._normalise()
        a = "- generated: 2026-01-01\n| base | 74.4% |"
        b = "- generated: 2026-01-01\n| base | 12.3% |"
        assert normalise(a) != normalise(b)

    def test_json_date_field_is_ignored(self):
        normalise = self._normalise()
        assert normalise('{\n  "date": "2026-01-01",\n  "n": 39\n}') == \
               normalise('{\n  "date": "2030-06-06",\n  "n": 39\n}')
