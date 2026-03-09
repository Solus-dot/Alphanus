#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.workspace import WorkspaceManager  # noqa: E402


def _err(code: str, message: str) -> Dict[str, Any]:
    return {"ok": False, "data": None, "error": {"code": code, "message": message}, "meta": {}}


def _read_args() -> Dict[str, Any]:
    raw = os.getenv("ALPHANUS_TOOL_ARGS_JSON", "").strip()
    if not raw:
        raw = sys.stdin.read().strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Tool args must be a JSON object")
    return parsed


def main() -> int:
    tool_name = sys.argv[1] if len(sys.argv) > 1 else os.getenv("ALPHANUS_TOOL_NAME", "").strip()
    if tool_name != "shell_command":
        print(json.dumps(_err("E_UNSUPPORTED", f"Unsupported tool: {tool_name}"), ensure_ascii=False))
        return 2

    try:
        args = _read_args()
        command = str(args["command"])
        workspace_root = os.getenv("ALPHANUS_WORKSPACE_ROOT", "").strip()
        if not workspace_root:
            raise ValueError("ALPHANUS_WORKSPACE_ROOT is required")
        home_root = os.getenv("ALPHANUS_HOME_ROOT", "").strip() or None
        workspace = WorkspaceManager(workspace_root=workspace_root, home_root=home_root)
        out = workspace.run_shell_command(command)
    except ValueError as exc:
        out = _err("E_VALIDATION", str(exc))
    except Exception as exc:  # pragma: no cover - defensive safeguard
        out = _err("E_IO", str(exc))

    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
