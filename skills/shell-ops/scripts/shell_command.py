#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.tool_script import emit, read_args  # noqa: E402
from core.workspace import WorkspaceManager  # noqa: E402


def main() -> int:
    args = read_args()
    workspace_root = os.getenv("ALPHANUS_WORKSPACE_ROOT", "").strip()
    if not workspace_root:
        raise ValueError("ALPHANUS_WORKSPACE_ROOT is required")
    home_root = os.getenv("ALPHANUS_HOME_ROOT", "").strip() or None
    workspace = WorkspaceManager(workspace_root=workspace_root, home_root=home_root)
    emit(workspace.run_shell_command(str(args["command"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
