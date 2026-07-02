from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory import LexicalMemory
from core.project import ProjectRuntime


@pytest.fixture(autouse=True)
def _disable_model_classification():
    """Disable model-based classification unless a test opts in."""
    with patch(
        "agent.classifier.TurnClassifier._should_model_classify",
        return_value=False,
    ):
        yield


@pytest.fixture
def project(tmp_path: Path) -> ProjectRuntime:
    home = tmp_path / "home"
    ws = home / "project"
    home.mkdir(parents=True)
    ws.mkdir(parents=True)
    return ProjectRuntime(project_root=str(ws))


@pytest.fixture
def memory(tmp_path: Path) -> LexicalMemory:
    path = tmp_path / "memory.pkl"
    return LexicalMemory(storage_path=str(path))
