"""Tools the model may call during the loop."""

from __future__ import annotations

__all__ = ["SEARCH_TOOL_SPEC", "SearchTool", "load_corpus", "parse_tool_calls"]


def __getattr__(name):  # lazy, so `python -m groundloop.tools.search` stays clean
    if name in __all__:
        from groundloop.tools import search as _search

        return getattr(_search, name)
    raise AttributeError(name)
