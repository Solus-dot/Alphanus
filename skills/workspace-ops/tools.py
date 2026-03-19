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
        "description": "Edit an existing workspace file. Prefer localized replacement with old_string/new_string; use content only for full-file replacement.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string"},
                "content": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["filepath"],
            "additionalProperties": False,
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
    "read_files": {
        "capability": "workspace_read",
        "description": "Read multiple files under home/workspace policy with per-file truncation metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "max_chars_per_file": {"type": "integer"},
            },
            "required": ["paths"],
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
    "search_code": {
        "capability": "workspace_read",
        "description": "Search the workspace codebase with ripgrep when available.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "path": {"type": "string"},
                "glob": {"type": "string"},
                "max_results": {"type": "integer"},
                "case_sensitive": {"type": "boolean"},
                "fixed_strings": {"type": "boolean"},
            },
            "required": ["query"],
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
    "run_checks": {
        "capability": "workspace_execute",
        "description": "Run tests, lint, or other verification commands with explicit argv.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "path": {"type": "string"},
                "timeout_s": {"type": "integer"},
            },
            "required": ["command"],
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
    if "content" in args:
        after = str(args["content"])
        edit_mode = "replace_full"
        replacements_applied = None
    else:
        has_old = "old_string" in args
        has_new = "new_string" in args
        if has_old != has_new:
            raise ValueError("edit_file requires both old_string and new_string when content is omitted")

        old_string = str(args.get("old_string", ""))
        new_string = str(args.get("new_string", ""))
        if not old_string:
            raise ValueError("edit_file old_string must be non-empty")

        match_count = before.count(old_string)
        if match_count == 0:
            raise ValueError("edit_file old_string was not found in the file")

        replace_all = bool(args.get("replace_all", False))
        if match_count > 1 and not replace_all:
            raise ValueError(
                "edit_file old_string matched multiple locations; provide a more specific old_string or set replace_all=true"
            )

        replacements_applied = match_count if replace_all else 1
        after = before.replace(old_string, new_string, -1 if replace_all else 1)
        edit_mode = "replace_all" if replace_all else "replace_one"

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
            "edit_mode": edit_mode,
            "replacements_applied": replacements_applied,
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


def _read_files(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    max_chars = int(args.get("max_chars_per_file", 20000))
    paths = [str(item) for item in args.get("paths") or []]
    files = env.workspace.read_files(paths, max_chars_per_file=max_chars)
    return {"files": files, "count": len(files), "max_chars_per_file": max_chars}


def _list_files(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    path = str(args.get("path", "."))
    names = env.workspace.list_files(path)
    return {"path": path, "files": names, "count": len(names)}


def _search_code(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    return env.workspace.search_code(
        str(args["query"]),
        path=str(args.get("path", ".")),
        glob=str(args["glob"]) if "glob" in args and args.get("glob") is not None else None,
        max_results=int(args.get("max_results", 50)),
        case_sensitive=bool(args.get("case_sensitive", False)),
        fixed_strings=bool(args.get("fixed_strings", True)),
    )


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


def _run_checks(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    return env.workspace.run_checks(
        str(args["command"]),
        args=[str(item) for item in args.get("args") or []],
        path=str(args.get("path", ".")),
        timeout_s=int(args.get("timeout_s", 120)),
    )


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
    if tool_name == "read_files":
        return _read_files(args, env)
    if tool_name == "list_files":
        return _list_files(args, env)
    if tool_name == "search_code":
        return _search_code(args, env)
    if tool_name == "delete_file":
        return _delete_file(args, env)
    if tool_name == "delete_path":
        return _delete_path(args, env)
    if tool_name == "workspace_tree":
        return _workspace_tree(args, env)
    if tool_name == "run_checks":
        return _run_checks(args, env)
    raise ValueError(f"Unsupported tool: {tool_name}")
