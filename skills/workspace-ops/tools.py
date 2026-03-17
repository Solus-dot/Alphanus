from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Dict

from core.skills import ToolExecutionEnv

TOOL_SPECS = {
    "create_directory": {
        "capability": "workspace_write",
        "description": "Create a directory in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    "create_file": {
        "capability": "workspace_write",
        "description": "Create or overwrite a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["filepath", "content"],
        },
    },
    "create_files": {
        "capability": "workspace_write",
        "description": "Create multiple files in the workspace in one call.",
        "parameters": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "filepath": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["filepath", "content"],
                    },
                }
            },
            "required": ["files"],
        },
    },
    "edit_file": {
        "capability": "workspace_edit",
        "description": "Replace content of an existing workspace file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["filepath", "content"],
        },
    },
    "read_file": {
        "capability": "workspace_read",
        "description": "Read a file under home/workspace policy.",
        "parameters": {
            "type": "object",
            "properties": {"filepath": {"type": "string"}},
            "required": ["filepath"],
        },
    },
    "list_files": {
        "capability": "workspace_read",
        "description": "List files in a directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": [],
        },
    },
    "delete_file": {
        "capability": "workspace_delete",
        "description": "Delete a workspace file.",
        "parameters": {
            "type": "object",
            "properties": {"filepath": {"type": "string"}},
            "required": ["filepath"],
        },
    },
    "delete_path": {
        "capability": "workspace_delete",
        "description": "Delete a workspace file or directory inside the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {"type": "boolean"},
            },
            "required": ["path"],
        },
    },
    "workspace_tree": {
        "capability": "workspace_tree",
        "description": "Render the workspace tree.",
        "parameters": {
            "type": "object",
            "properties": {"max_depth": {"type": "integer"}},
            "required": [],
        },
    },
}


def _line_count(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + 1


def _changed_line_count(before: str, after: str) -> int:
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    max_len = max(len(before_lines), len(after_lines))
    changed = 0
    for idx in range(max_len):
        left = before_lines[idx] if idx < len(before_lines) else None
        right = after_lines[idx] if idx < len(after_lines) else None
        if left != right:
            changed += 1
    return changed


def _path_info(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    return {
        "filepath": path_str,
        "basename": path.name,
    }


def _create_file(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    content = str(args["content"])
    path_str = env.workspace.create_file(str(args["filepath"]), content)
    data = _path_info(path_str)
    data.update(
        {
            "created": True,
            "bytes_written": len(content.encode("utf-8")),
            "line_count": _line_count(content),
        }
    )
    return data


def _create_directory(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    path_str = env.workspace.create_directory(str(args["path"]))
    data = _path_info(path_str)
    data.update({"created": True, "kind": "directory"})
    return data


def _create_files(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    created: list[Dict[str, Any]] = []
    for item in args.get("files") or []:
        if not isinstance(item, dict):
            continue
        created.append(
            _create_file(
                {
                    "filepath": str(item.get("filepath", "")),
                    "content": str(item.get("content", "")),
                },
                env,
            )
        )
    return {"created": created, "count": len(created)}


def _edit_file(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    filepath = str(args["filepath"])
    before = env.workspace.read_file(filepath)
    after = str(args["content"])
    path_str = env.workspace.edit_file(filepath, after)
    diff_text = "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"{filepath} (before)",
            tofile=f"{filepath} (after)",
            lineterm="",
        )
    )
    data = _path_info(path_str)
    data.update(
        {
            "edited": True,
            "changed": before != after,
            "bytes_before": len(before.encode("utf-8")),
            "bytes_after": len(after.encode("utf-8")),
            "line_count_before": _line_count(before),
            "line_count_after": _line_count(after),
            "changed_lines": _changed_line_count(before, after),
            "diff": diff_text,
        }
    )
    return data


def _read_file(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    filepath = str(args["filepath"])
    content = env.workspace.read_file(filepath)
    data = _path_info(filepath)
    data.update(
        {
            "content": content,
            "size_bytes": len(content.encode("utf-8")),
            "line_count": _line_count(content),
        }
    )
    return data


def _list_files(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    path = str(args.get("path", "."))
    names = env.workspace.list_files(path)
    return {"path": path, "files": names, "count": len(names)}


def _delete_file(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    filepath = str(args["filepath"])
    target = env.workspace._resolve_read_path(filepath)
    size_bytes = target.stat().st_size
    path_str = env.workspace.delete_file(filepath)
    data = _path_info(path_str)
    data.update(
        {
            "deleted": True,
            "size_bytes": size_bytes,
            "kind": "file",
        }
    )
    return data


def _delete_path(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    path = str(args["path"])
    recursive = bool(args.get("recursive", False))
    target = env.workspace._resolve_read_path(path)
    is_dir = target.is_dir()
    size_bytes = 0
    file_count = 0
    if is_dir:
        for child in target.rglob("*"):
            if child.is_file():
                size_bytes += child.stat().st_size
                file_count += 1
    else:
        size_bytes = target.stat().st_size
        file_count = 1
    path_str = env.workspace.delete_path(path, recursive=recursive)
    data = _path_info(path_str)
    data.update(
        {
            "deleted": True,
            "recursive": recursive,
            "kind": "directory" if is_dir else "file",
            "size_bytes": size_bytes,
            "file_count": file_count,
        }
    )
    return data


def _workspace_tree(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    max_depth = max(1, int(args.get("max_depth", 3)))
    tree = env.workspace.workspace_tree(max_depth=max_depth)
    return {"tree": tree, "max_depth": max_depth}


def execute(tool_name: str, args: Dict[str, Any], env: ToolExecutionEnv):
    if tool_name == "create_directory":
        return _create_directory(args, env)
    if tool_name == "create_file":
        return _create_file(args, env)
    if tool_name == "create_files":
        return _create_files(args, env)
    if tool_name == "edit_file":
        return _edit_file(args, env)
    if tool_name == "read_file":
        return _read_file(args, env)
    if tool_name == "list_files":
        return _list_files(args, env)
    if tool_name == "delete_file":
        return _delete_file(args, env)
    if tool_name == "delete_path":
        return _delete_path(args, env)
    if tool_name == "workspace_tree":
        return _workspace_tree(args, env)
    raise ValueError(f"Unsupported tool: {tool_name}")
