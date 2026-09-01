"""End-to-end behaviour of the three conditions."""

import pytest

from groundloop.loop.critique import parse_critique, render
from groundloop.loop.pipeline import CONDITIONS, run_condition


class TestConditions:
    def test_base_produces_a_draft_and_nothing_else(self, llm, tool):
        traj = run_condition(llm, {"id": "x", "question": "Who is KS-9's director?"}, "base", tool=tool)
        assert traj.draft and not traj.tool_calls and traj.critique is None

    def test_plain_critique_revises_without_retrieving(self, llm, tool):
        traj = run_condition(llm, {"id": "x", "question": "How tall is Mount Everest?"},
                             "plain_critique", tool=tool)
        assert traj.critique is not None
        assert not traj.evidence, "the control condition must never see evidence"

    def test_groundloop_retrieves_and_corrects_a_wrong_draft(self, llm, tool):
        traj = run_condition(llm, {"id": "q25", "question": "How many hours are there in a KS-9 station day?"},
                             "groundloop", tool=tool)
        assert "f03" in {p.id for p in traj.evidence}
        assert "24" in traj.draft, "fixture precondition: the draft applies a terrestrial prior"
        assert "31" in traj.final and "[f03]" in traj.final

    def test_groundloop_abstains_when_the_corpus_is_silent(self, llm, tool):
        traj = run_condition(llm, {"id": "q37", "question": "Who is KS-9's chief medical officer?"},
                             "groundloop", tool=tool)
        assert "does not answer" in traj.final.lower()

    def test_unknown_condition_is_rejected(self, llm, tool):
        with pytest.raises(ValueError):
            run_condition(llm, {"id": "x", "question": "?"}, "magic", tool=tool)

    @pytest.mark.parametrize("condition", CONDITIONS)
    def test_every_condition_yields_a_final_answer(self, llm, tool, condition):
        traj = run_condition(llm, {"id": "x", "question": "When was smallpox eradicated?"},
                             condition, tool=tool)
        assert traj.final.strip()


class TestCritiqueParsing:
    def test_json(self):
        c = parse_critique('{"verdict": "revise", "issues": [{"claim": "x", "problem": "y", "principle": "P3"}]}')
        assert c.verdict == "revise" and c.unsupported[0].text == "x"

    def test_fenced_json(self):
        c = parse_critique('```json\n{"verdict": "ok", "issues": []}\n```')
        assert c.verdict == "ok" and not c.claims

    def test_json_buried_in_prose(self):
        c = parse_critique('Sure! {"verdict": "revise", "issues": []} Hope that helps.')
        assert c.verdict == "revise"

    def test_prose_fallback_does_not_read_as_all_clear(self):
        # A model that fails to emit JSON must not be scored as having found
        # no problems - that would flatter whichever condition formats worst.
        c = parse_critique("- the date is wrong\n- the name is invented")
        assert c.verdict == "revise" and len(c.unsupported) == 2
        assert any("not valid JSON" in n for n in c.notes)

    def test_explicit_all_clear(self):
        assert parse_critique("ok").verdict == "ok"

    def test_render_lists_only_the_problems(self):
        c = parse_critique('{"verdict": "revise", "issues": [{"claim": "bad", "problem": "invented"}]}')
        out = render(c)
        assert "bad" in out and "invented" in out


def test_trajectory_round_trips_to_json(llm, tool):
    traj = run_condition(llm, {"id": "x", "question": "What is the Halden Array?"}, "groundloop", tool=tool)
    blob = traj.to_json()
    assert blob["example_id"] == "x"
    assert isinstance(blob["evidence"], list) and isinstance(blob["tool_calls"], list)
