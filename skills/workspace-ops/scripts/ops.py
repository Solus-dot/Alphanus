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


def _ok(data: Any) -> Dict[str, Any]:
    return {"ok": True, "data": data, "error": None, "meta": {}}


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


def _manager() -> WorkspaceManager:
    workspace_root = os.getenv("ALPHANUS_WORKSPACE_ROOT", "").strip()
    if not workspace_root:
        raise ValueError("ALPHANUS_WORKSPACE_ROOT is required")
    home_root = os.getenv("ALPHANUS_HOME_ROOT", "").strip() or None
    return WorkspaceManager(workspace_root=workspace_root, home_root=home_root)


def _run(tool_name: str, args: Dict[str, Any], workspace: WorkspaceManager) -> Dict[str, Any]:
    if tool_name == "create_file":
        path = workspace.create_file(str(args["filepath"]), str(args["content"]))
        return _ok({"filepath": path})

    if tool_name == "edit_file":
        path = workspace.edit_file(str(args["filepath"]), str(args["content"]))
        return _ok({"filepath": path})

    if tool_name == "read_file":
        text = workspace.read_file(str(args["filepath"]))
        return _ok({"content": text})

    if tool_name == "list_files":
        names = workspace.list_files(str(args.get("path", ".")))
        return _ok({"files": names})

    if tool_name == "delete_file":
        path = workspace.delete_file(str(args["filepath"]))
        return _ok({"filepath": path})

    if tool_name == "workspace_tree":
        max_depth = int(args.get("max_depth", 3))
        return _ok({"tree": workspace.workspace_tree(max_depth=max(1, max_depth))})

    return _err("E_UNSUPPORTED", f"Unsupported tool: {tool_name}")


def main() -> int:
    tool_name = sys.argv[1] if len(sys.argv) > 1 else os.getenv("ALPHANUS_TOOL_NAME", "").strip()
    if not tool_name:
        print(json.dumps(_err("E_VALIDATION", "Missing tool name"), ensure_ascii=False))
        return 2

    try:
        args = _read_args()
        workspace = _manager()
        out = _run(tool_name, args, workspace)
    except ValueError as exc:
        out = _err("E_VALIDATION", str(exc))
    except FileNotFoundError as exc:
        out = _err("E_NOT_FOUND", str(exc))
    except PermissionError as exc:
        out = _err("E_POLICY", str(exc))
    except TimeoutError as exc:
        out = _err("E_TIMEOUT", str(exc))
    except Exception as exc:  # pragma: no cover - defensive safeguard
        out = _err("E_IO", str(exc))

    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
