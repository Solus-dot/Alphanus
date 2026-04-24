from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory import LexicalMemory
from core.workspace import WorkspaceManager


@pytest.fixture(autouse=True)
def _disable_model_classification():
    """Disable model-based classification unless a test opts in."""
    with patch(
        "agent.classifier.TurnClassifier._should_model_classify",
        return_value=False,
    ):
        yield


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    home = tmp_path / "home"
    ws = home / "workspace"
    home.mkdir(parents=True)
    ws.mkdir(parents=True)
    return WorkspaceManager(workspace_root=str(ws), home_root=str(home))


@pytest.fixture
def memory(tmp_path: Path) -> LexicalMemory:
    path = tmp_path / "memory.pkl"
    return LexicalMemory(storage_path=str(path))
