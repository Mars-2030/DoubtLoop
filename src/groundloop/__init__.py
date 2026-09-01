"""GroundLoop: research-grounded self-critique for small language models.

The package is deliberately dependency-free at import time. Heavy optional
backends (transformers, trl, peft) are imported lazily inside the modules that
need them, so the loop, the retrieval tool, and the evaluation suite all run on
a bare Python 3.10+ interpreter.
"""

__version__ = "0.1.0"

from groundloop.schema import Claim, Critique, Trajectory, ToolCall  # noqa: F401
