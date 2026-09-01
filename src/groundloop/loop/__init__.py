"""The GroundLoop itself: draft -> claims -> search -> critique -> revise."""

from __future__ import annotations

__all__ = ["run_condition", "run_dataset", "CONDITIONS"]


def __getattr__(name):
    if name in __all__:
        from groundloop.loop import pipeline as _p

        return getattr(_p, name)
    raise AttributeError(name)
