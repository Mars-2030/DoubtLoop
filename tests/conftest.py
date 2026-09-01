import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from groundloop.llm import get_backend
from groundloop.tools.search import SearchTool


@pytest.fixture(scope="session")
def tool():
    return SearchTool()


@pytest.fixture()
def llm():
    return get_backend("scripted")
