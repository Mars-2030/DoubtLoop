"""Dataset integrity, generated training data, and the training entry points."""

import json

import pytest

from groundloop import config
from groundloop.data_gen.build_trajectories import to_preference_record, to_sft_record
from groundloop.datasets import load_qa, load_sycophancy, read_jsonl
from groundloop.loop.pipeline import run_condition
from groundloop.train import dpo as dpo_mod
from groundloop.train import sft as sft_mod


class TestDatasetIntegrity:
    def test_qa_ids_are_unique(self):
        rows = load_qa()
        assert len({r["id"] for r in rows}) == len(rows)

    def test_every_support_id_resolves_to_a_passage(self, tool):
        for rows in (load_qa(), load_sycophancy()):
            for row in rows:
                for pid in row.get("support", []):
                    assert pid in tool.by_id, f"{row['id']} cites missing passage {pid}"

    def test_unanswerable_items_have_no_support(self):
        for row in load_qa():
            if row.get("expect_abstention"):
                assert not row.get("support")

    def test_answerable_items_declare_answer_keys(self):
        for row in load_qa():
            if not row.get("expect_abstention"):
                assert row.get("answer_keys"), f"{row['id']} has no answer_keys to score against"

    def test_gold_passages_are_actually_retrievable(self, tool):
        # If BM25 cannot surface the gold passage, a failure downstream is a
        # retrieval failure being misread as a faithfulness failure.
        misses = []
        for row in load_qa():
            gold = set(row.get("support", []))
            if not gold:
                continue
            got = {p.id for p in tool.search(row["question"], 4)}
            if not (gold & got):
                misses.append(row["id"])
        assert not misses, f"gold passage unreachable for {misses}"

    def test_fictional_passages_are_labelled_as_such(self, tool):
        for p in tool.passages:
            if p.world == "fictional":
                assert "FICTIONAL" in p.title.upper()

    def test_the_corpus_has_both_worlds(self, tool):
        worlds = {p.world for p in tool.passages}
        assert {"real", "fictional"} <= worlds


@pytest.fixture(scope="module")
def traj():
    from groundloop.llm import get_backend
    from groundloop.tools.search import SearchTool

    return run_condition(
        get_backend("scripted"),
        {"id": "q25", "question": "How many hours are there in a KS-9 station day?"},
        "groundloop",
        tool=SearchTool(),
    )


class TestTrainingRecords:
    def test_distilled_record_teaches_search_then_answer(self, traj):
        rec = to_sft_record(traj, "distilled")
        roles = [m["role"] for m in rec["messages"]]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]
        assert "<tool_call>" in rec["messages"][2]["content"]
        # The bad draft must not be the thing we train on.
        assert traj.draft not in rec["messages"][-1]["content"]

    def test_trajectory_record_keeps_the_critique_turn(self, traj):
        rec = to_sft_record(traj, "trajectory")
        assert len(rec["messages"]) > 5
        assert any("critique" in m["content"].lower() for m in rec["messages"] if m["role"] == "user")

    def test_unknown_style_is_rejected(self, traj):
        with pytest.raises(ValueError):
            to_sft_record(traj, "freeform")

    def test_preference_pair_shares_one_prompt(self, traj):
        rec = to_preference_record(traj)
        assert rec["chosen"][0]["content"] == traj.final
        assert rec["rejected"][0]["content"] == traj.draft
        # Both completions must answer the same evidence-bearing prompt.
        assert "Evidence:" in rec["prompt"][1]["content"]


class TestTrainingEntryPoints:
    def test_sft_dry_run(self, tmp_path, capsys):
        path = tmp_path / "sft.jsonl"
        path.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
        # --allow-tiny-dataset because this checks argument handling, not the
        # size guard; one record would otherwise (correctly) be refused.
        assert sft_mod.main(["--data", str(path), "--dry-run", "--allow-tiny-dataset"]) == 0
        assert '"stage": "sft"' in capsys.readouterr().out

    def test_dpo_dry_run(self, tmp_path, capsys):
        path = tmp_path / "prefs.jsonl"
        path.write_text(json.dumps({
            "prompt": [{"role": "user", "content": "q"}],
            "chosen": [{"role": "assistant", "content": "good"}],
            "rejected": [{"role": "assistant", "content": "bad"}],
        }) + "\n")
        assert dpo_mod.main(["--data", str(path), "--dry-run", "--allow-tiny-dataset"]) == 0
        assert '"stage": "dpo"' in capsys.readouterr().out

    def test_sft_rejects_records_without_messages(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"text": "nope"}) + "\n")
        with pytest.raises(ValueError, match="messages"):
            sft_mod.load_records(path)

    def test_dpo_rejects_incomplete_pairs(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"prompt": [], "chosen": []}) + "\n")
        with pytest.raises(ValueError, match="rejected"):
            dpo_mod.load_records(path)

    def test_empty_file_is_an_error_not_an_empty_run(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        with pytest.raises(ValueError):
            sft_mod.load_records(path)


class TestNoOpTrainingGuard:
    """A fine-tune that takes three gradient steps looks like a successful run
    in the logs, produces an adapter, and makes the before/after report that the
    method did nothing. It has to fail loudly instead."""

    def test_step_count_matches_what_the_trainer_would_do(self):
        # HF floors len(dataloader) // grad_accum, then clamps to at least 1.
        assert sft_mod.optimizer_steps(4, 1, 16, 3.0) == 3      # the Colab case
        assert sft_mod.optimizer_steps(320, 1, 16, 3.0) == 60
        assert sft_mod.optimizer_steps(32, 2, 8, 1.0) == 2

    def test_a_dataset_too_small_to_train_on_is_refused(self, tmp_path):
        path = tmp_path / "tiny.jsonl"
        path.write_text('{"messages": [{"role": "user", "content": "a"}]}\n' * 4)
        with pytest.raises(SystemExit, match="will not train anything"):
            sft_mod.main(["--data", str(path), "--dry-run"])

    def test_the_refusal_names_a_smaller_grad_accum(self, tmp_path, capsys):
        path = tmp_path / "tiny.jsonl"
        path.write_text('{"messages": [{"role": "user", "content": "a"}]}\n' * 4)
        with pytest.raises(SystemExit) as exc:
            sft_mod.main(["--data", str(path), "--dry-run"])
        assert "--grad-accum" in str(exc.value) and "--no-require-correct" in str(exc.value)

    def test_the_override_lets_it_through(self, tmp_path):
        path = tmp_path / "tiny.jsonl"
        path.write_text('{"messages": [{"role": "user", "content": "a"}]}\n' * 4)
        assert sft_mod.main(["--data", str(path), "--dry-run", "--allow-tiny-dataset"]) == 0

    def test_a_reasonable_dataset_is_not_blocked(self, tmp_path):
        path = tmp_path / "ok.jsonl"
        path.write_text('{"messages": [{"role": "user", "content": "a"}]}\n' * 200)
        assert sft_mod.main(["--data", str(path), "--dry-run"]) == 0

    def test_dpo_has_the_same_guard(self, tmp_path):
        path = tmp_path / "tiny.jsonl"
        path.write_text(json.dumps({
            "prompt": [{"role": "user", "content": "q"}],
            "chosen": [{"role": "assistant", "content": "a"}],
            "rejected": [{"role": "assistant", "content": "b"}],
        }) + "\n")
        with pytest.raises(SystemExit, match="will not train anything"):
            dpo_mod.main(["--data", str(path), "--dry-run"])


class TestYieldReporting:
    def test_low_yield_prints_the_levers(self):
        from groundloop.data_gen.build_trajectories import summarize

        out = summarize({"n": 51, "sft": [1] * 4, "prefs": [], "drops": {"failed": 47}})
        assert "8% yield" in out
        assert "--no-require-correct" in out and "--tau" in out

    def test_healthy_yield_stays_quiet(self):
        from groundloop.data_gen.build_trajectories import summarize

        out = summarize({"n": 51, "sft": [1] * 33, "prefs": [1] * 26, "drops": {}})
        assert "65% yield" in out and "--no-require-correct" not in out

    def test_selection_and_reporting_thresholds_are_reported_separately(self):
        from groundloop.data_gen.build_trajectories import summarize

        out = summarize({"n": 10, "sft": [1] * 9, "prefs": [], "drops": {}}, 0.5, 0.7)
        assert "0.5" in out and "reporting stays at 0.7" in out
