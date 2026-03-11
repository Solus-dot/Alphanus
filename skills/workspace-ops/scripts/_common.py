from __future__ import annotations

import os

from core.tool_script import emit, read_args
from core.workspace import WorkspaceManager


def load_workspace() -> WorkspaceManager:
    workspace_root = os.getenv("ALPHANUS_WORKSPACE_ROOT", "").strip()
    if not workspace_root:
        raise ValueError("ALPHANUS_WORKSPACE_ROOT is required")
    home_root = os.getenv("ALPHANUS_HOME_ROOT", "").strip() or None
    return WorkspaceManager(workspace_root=workspace_root, home_root=home_root)


__all__ = ["emit", "read_args", "load_workspace"]
