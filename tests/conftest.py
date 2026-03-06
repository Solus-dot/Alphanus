from __future__ import annotations

from pathlib import Path

import pytest

from core.memory import VectorMemory
from core.workspace import WorkspaceManager


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    home = tmp_path / "home"
    ws = home / "workspace"
    home.mkdir(parents=True)
    ws.mkdir(parents=True)
    return WorkspaceManager(workspace_root=str(ws), home_root=str(home))


@pytest.fixture
def memory(tmp_path: Path) -> VectorMemory:
    path = tmp_path / "memory.pkl"
    return VectorMemory(storage_path=str(path))
