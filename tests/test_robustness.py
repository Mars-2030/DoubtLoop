"""Retrieval stress and metric sensitivity.

Both files exist to answer "is the headline an artefact?", so they get tests
that check the *properties* the answer depends on, not just that they run.
"""

import json

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


class TestRescoringWithoutRegenerating:
    """A change to the scorer is not a change to the model.

    After a metric fix the right move is to re-score the same generations, not
    to sample new ones and hope the difference was the metric. That also means
    a half-hour GPU run does not have to be repeated.
    """

    def test_saved_trajectories_land_under_the_requested_out_dir(self, tmp_path):
        # Regression: run() wrote to config.RESULTS_DIR/raw regardless of
        # --out-dir, so a run into results-before/ left its logs in results/.
        from groundloop.eval import run_all

        run_all.main(["--limit", "3", "--out-dir", str(tmp_path)])
        assert (tmp_path / "raw" / "qa_base.jsonl").exists()
        assert (tmp_path / "raw" / "probe_groundloop.jsonl").exists()

    def test_rescoring_reproduces_the_table_without_a_model(self, tmp_path):
        from groundloop.eval import run_all

        generated, rescored = tmp_path / "gen", tmp_path / "re"
        run_all.main(["--limit", "4", "--out-dir", str(generated)])
        run_all.main(["--from-trajectories", str(generated / "raw"), "--out-dir", str(rescored)])

        a = json.loads((generated / "metrics.json").read_text())["conditions"]
        b = json.loads((rescored / "metrics.json").read_text())["conditions"]
        assert a == b, "re-scoring the same trajectories changed the numbers"

    def test_a_rescored_table_says_so(self, tmp_path):
        from groundloop.eval import run_all

        generated, rescored = tmp_path / "gen", tmp_path / "re"
        run_all.main(["--limit", "3", "--out-dir", str(generated)])
        run_all.main(["--from-trajectories", str(generated / "raw"), "--out-dir", str(rescored)])
        table = (rescored / "comparison_table.md").read_text()
        assert "Re-scored from saved trajectories" in table
        # It must not silently inherit this run's backend as provenance.
        assert "n/a (scripted stand-in)" not in table

    def test_a_missing_directory_is_explained(self, tmp_path):
        from groundloop.eval import run_all

        with pytest.raises(SystemExit, match="--from-trajectories expects"):
            run_all.main(["--from-trajectories", str(tmp_path / "nope"), "--out-dir", str(tmp_path)])

    def test_retrieval_stress_keeps_its_trajectories_by_default(self, tmp_path):
        from groundloop.eval import retrieval_stress

        retrieval_stress.main(["--modes", "normal", "--limit", "3", "--out-dir", str(tmp_path)])
        assert (tmp_path / "raw" / "stress_normal.jsonl").exists()


class TestCritiqueQuality:
    """Whether the model's self-critique agrees with the evidence it was shown.

    The measurement the method turns on: if the critique endorses whatever the
    draft said, retrieval and revision are both wasted.
    """

    @staticmethod
    def _traj(model_supported, harness_supported, claim="the crew stayed in lunar orbit"):
        return {
            "example_id": "q01",
            "critique": {
                "claims": [{"text": claim, "supported": model_supported, "best_passage": "r01"}],
                "harness_claims": [{"text": claim, "supported": harness_supported,
                                    "reason": "not covered by the evidence", "coverage": 0.4}],
            },
        }

    def test_false_approval_is_counted(self):
        from groundloop.eval import critique_quality

        rows, summary = critique_quality.score_trajectories([self._traj(True, False)])
        assert summary["false_approval_rate"] == 100.0
        assert summary["blind_critique_rate"] == 100.0
        assert "approved what the evidence does not support" in rows[0]["kind"]

    def test_false_alarm_is_counted_separately(self):
        from groundloop.eval import critique_quality

        _rows, summary = critique_quality.score_trajectories([self._traj(False, True)])
        assert summary["false_alarm_rate"] == 100.0 and summary["false_approval_rate"] == 0.0

    def test_agreement(self):
        from groundloop.eval import critique_quality

        _rows, summary = critique_quality.score_trajectories(
            [self._traj(True, True), self._traj(False, False)])
        assert summary["agreement_rate"] == 100.0
        assert summary["blind_critique_rate"] == 0.0

    def test_claims_are_paired_by_overlap_not_by_index(self):
        from groundloop.eval.critique_quality import pair_claims

        harness = [{"text": "The Calvin cycle runs in the stroma of the chloroplast."},
                   {"text": "It does not require direct light."}]
        model = [{"text": "It does not require direct light"},
                 {"text": "The Calvin cycle runs in the stroma"}]
        pairs = pair_claims(model, harness)
        assert pairs[0][1]["text"].startswith("The Calvin cycle")
        assert pairs[1][1]["text"].startswith("It does not")

    def test_an_unmatched_claim_is_not_silently_scored(self):
        from groundloop.eval import critique_quality

        traj = self._traj(True, False)
        traj["critique"]["claims"] = [{"text": "something else entirely", "supported": True}]
        _rows, summary = critique_quality.score_trajectories([traj])
        assert summary["claims_unpaired"] == 1 and summary["claims_paired"] == 0

    def test_the_harness_verdict_is_recorded_alongside_the_models(self, llm, tool):
        from groundloop.loop.pipeline import run_condition

        traj = run_condition(llm, {"id": "q25", "question": "How many hours in a KS-9 station day?"},
                             "groundloop", tool=tool)
        assert traj.critique.harness_claims, "the harness verdict was not kept"
        assert traj.to_json()["critique"]["harness_claims"]
