"""Retrieval has to be right and it has to be *stable* - an ablation compares
runs, so a retriever that reorders itself invalidates the comparison."""

import json

import pytest

from groundloop.tools.search import SearchTool, parse_tool_calls


class TestRetrieval:
    def test_finds_the_gold_passage_for_a_direct_question(self, tool):
        hits = tool.search("how long is a KS-9 station day", 3)
        assert hits[0].id == "f03"

    def test_ranking_is_deterministic(self, tool):
        a = [p.id for p in tool.search("Nobel Prize penicillin", 4)]
        b = [p.id for p in tool.search("Nobel Prize penicillin", 4)]
        assert a == b

    def test_k_is_clamped(self, tool):
        assert len(tool.search("KS-9", 99)) <= 10

    def test_a_query_matching_nothing_returns_nothing(self, tool):
        assert tool.search("zzzqqq nonexistenttoken") == []

    def test_no_passage_scores_above_zero_without_a_shared_term(self, tool):
        hits = tool.search("photosynthesis chloroplast", 4)
        assert all(p.score > 0 for p in hits)


class TestToolCallInterface:
    def test_call_returns_payload_and_logs(self, tool):
        n_before = len(tool.call_log)
        out = tool.call({"query": "Rosetta Stone", "k": 2})
        assert out["results"] and len(tool.call_log) == n_before + 1

    def test_empty_query_is_an_error_not_a_crash(self, tool):
        out = tool.call({"query": "  "})
        assert out["results"] == [] and "error" in out

    def test_render_is_citable(self, tool):
        block = tool.render(tool.search("Apollo 11", 1))
        assert block.startswith("[r01]")


class TestToolCallParsing:
    def test_qwen_block(self):
        raw = '<tool_call>{"name": "search", "arguments": {"query": "apollo"}}</tool_call>'
        assert parse_tool_calls(raw) == [{"name": "search", "arguments": {"query": "apollo"}}]

    def test_stringified_arguments(self):
        raw = '<tool_call>{"name": "search", "arguments": "{\\"query\\": \\"apollo\\"}"}</tool_call>'
        assert parse_tool_calls(raw)[0]["arguments"]["query"] == "apollo"

    def test_bare_json_without_the_wrapper(self):
        raw = 'I will look it up: {"name": "search", "arguments": {"query": "insulin"}}'
        assert parse_tool_calls(raw)[0]["arguments"]["query"] == "insulin"

    def test_python_style_call(self):
        assert parse_tool_calls('search("dead sea depth")')[0]["arguments"]["query"] == "dead sea depth"

    def test_malformed_json_yields_no_calls_rather_than_raising(self):
        assert parse_tool_calls('<tool_call>{"name": "search", ') == []

    def test_prose_with_no_call(self):
        assert parse_tool_calls("I already know the answer.") == []


def test_corpus_ids_are_unique(tool):
    ids = [p.id for p in tool.passages]
    assert len(ids) == len(set(ids))
