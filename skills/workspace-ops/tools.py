from __future__ import annotations

import difflib
import re
from pathlib import Path

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
    "edit_file": {
        "capability": "workspace_edit",
        "description": (
            "Edit an existing workspace file. Supports exactly one mode per call: full content replace, "
            "localized old_string/new_string, or regex replacement. Optional line-range and anchor selectors "
            "can scope non-content edits."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string"},
                "content": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
                "regex_pattern": {"type": "string"},
                "regex_replacement": {"type": "string"},
                "regex_flags": {"type": "string"},
                "regex_replace_all": {"type": "boolean"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
                "before_anchor": {"type": "string"},
                "after_anchor": {"type": "string"},
            },
            "required": ["filepath"],
            "additionalProperties": False,
        },
    },
    "read_file": {
        "capability": "workspace_read",
        "description": "Read a file under home/workspace policy. Optional line-range and anchor selectors can return a specific section.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
                "before_anchor": {"type": "string"},
                "after_anchor": {"type": "string"},
                "include_line_numbers": {"type": "boolean"},
            },
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
        "description": "Search the workspace codebase with ripgrep when available. Optional context lines can be returned per match.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "path": {"type": "string"},
                "glob": {"type": "string"},
                "max_results": {"type": "integer"},
                "case_sensitive": {"type": "boolean"},
                "fixed_strings": {"type": "boolean"},
                "before_context": {"type": "integer"},
                "after_context": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    "move_path": {
        "capability": "workspace_write",
        "description": "Move or rename a workspace file or directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_path": {"type": "string"},
                "destination_path": {"type": "string"},
                "overwrite": {"type": "boolean"},
            },
            "required": ["source_path", "destination_path"],
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


def _path_info(path_str: str) -> dict[str, object]:
    path = Path(path_str)
    return {
        "filepath": path_str,
        "basename": path.name,
    }


def _optional_positive_int(args: dict[str, object], key: str) -> int | None:
    if key not in args or args.get(key) is None:
        return None
    value = int(args[key])
    if value < 1:
        raise ValueError(f"{key} must be >= 1")
    return value


def _section_selectors(args: dict[str, object]) -> dict[str, object]:
    return {
        "start_line": _optional_positive_int(args, "start_line"),
        "end_line": _optional_positive_int(args, "end_line"),
        "before_anchor": str(args.get("before_anchor", "")),
        "after_anchor": str(args.get("after_anchor", "")),
    }


def _has_section_selectors(section: dict[str, object]) -> bool:
    return (
        section.get("start_line") is not None
        or section.get("end_line") is not None
        or bool(section.get("before_anchor"))
        or bool(section.get("after_anchor"))
    )


def _regex_flags(flags_text: str) -> int:
    flags = 0
    valid = {"i", "m", "s"}
    for token in str(flags_text or ""):
        if token.isspace():
            continue
        lowered = token.lower()
        if lowered not in valid:
            raise ValueError("regex_flags supports only i, m, and s")
        if lowered == "i":
            flags |= re.IGNORECASE
        elif lowered == "m":
            flags |= re.MULTILINE
        elif lowered == "s":
            flags |= re.DOTALL
    return flags


def _optional_non_negative_int(args: dict[str, object], key: str, default: int = 0) -> int:
    raw = args.get(key, default)
    value = int(raw)
    if value < 0:
        raise ValueError(f"{key} must be >= 0")
    return value


def _render_numbered_content(content: str, start_line: int) -> str:
    lines = content.splitlines(keepends=True)
    if not lines:
        return ""
    return "".join(f"{start_line + idx}: {line}" for idx, line in enumerate(lines))


def _create_file(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
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


def _create_directory(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    path_str = env.workspace.create_directory(str(args["path"]))
    data = _path_info(path_str)
    data.update({"created": True, "kind": "directory"})
    return data


def _edit_file(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    filepath = str(args["filepath"])
    before = env.workspace.read_file(filepath)
    section = _section_selectors(args)
    has_section_selectors = _has_section_selectors(section)

    has_content_mode = "content" in args
    has_old_token = "old_string" in args or "new_string" in args
    has_regex_token = "regex_pattern" in args or "regex_replacement" in args
    modes_selected = int(has_content_mode) + int(has_old_token) + int(has_regex_token)
    if modes_selected != 1:
        raise ValueError("edit_file requires exactly one edit mode: content, old_string/new_string, or regex_pattern/regex_replacement")

    if has_content_mode:
        if has_section_selectors:
            raise ValueError("edit_file content mode cannot be combined with line-range or anchor selectors")
        after = str(args["content"])
        edit_mode = "replace_full"
        replacements_applied = None
        resolved_start_line = 1 if before else 0
        resolved_end_line = _line_count(before)
        total_line_count_before = _line_count(before)
        total_line_count_after = _line_count(after)
    else:
        resolved = env.workspace.resolve_text_section(
            before,
            start_line=section["start_line"],
            end_line=section["end_line"],
            before_anchor=str(section["before_anchor"] or ""),
            after_anchor=str(section["after_anchor"] or ""),
        )
        target_before = str(resolved["content"])
        if has_old_token:
            has_old = "old_string" in args
            has_new = "new_string" in args
            if has_old != has_new:
                raise ValueError("edit_file requires both old_string and new_string when content is omitted")

            old_string = str(args.get("old_string", ""))
            new_string = str(args.get("new_string", ""))
            if not old_string:
                raise ValueError("edit_file old_string must be non-empty")

            match_count = target_before.count(old_string)
            if match_count == 0:
                raise ValueError("edit_file old_string was not found in the selected section")

            replace_all = bool(args.get("replace_all", False))
            if match_count > 1 and not replace_all:
                raise ValueError(
                    "edit_file old_string matched multiple locations; provide a more specific old_string or set replace_all=true"
                )
            replacements_applied = match_count if replace_all else 1
            target_after = target_before.replace(old_string, new_string, -1 if replace_all else 1)
            edit_mode = "replace_all" if replace_all else "replace_one"
        else:
            has_pattern = "regex_pattern" in args
            has_replacement = "regex_replacement" in args
            if has_pattern != has_replacement:
                raise ValueError("edit_file requires both regex_pattern and regex_replacement for regex mode")
            pattern_text = str(args.get("regex_pattern", ""))
            replacement_text = str(args.get("regex_replacement", ""))
            if not pattern_text:
                raise ValueError("edit_file regex_pattern must be non-empty")
            regex = re.compile(pattern_text, _regex_flags(str(args.get("regex_flags", ""))))
            match_count = len(list(regex.finditer(target_before)))
            if match_count == 0:
                raise ValueError("edit_file regex_pattern did not match the selected section")
            replace_all = bool(args.get("regex_replace_all", False))
            if match_count > 1 and not replace_all:
                raise ValueError(
                    "edit_file regex_pattern matched multiple locations; set regex_replace_all=true or tighten the pattern"
                )
            target_after, replacements_applied = regex.subn(
                replacement_text,
                target_before,
                0 if replace_all else 1,
            )
            edit_mode = "regex_replace_all" if replace_all else "regex_replace_one"

        after = (
            before[: int(resolved["start_offset"])]
            + target_after
            + before[int(resolved["end_offset"]) :]
        )
        resolved_start_line = int(resolved["start_line"])
        resolved_end_line = int(resolved["end_line"])
        total_line_count_before = int(resolved["total_line_count"])
        total_line_count_after = _line_count(after)

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
            "section_scoped": not has_content_mode and has_section_selectors,
            "resolved_start_line": resolved_start_line,
            "resolved_end_line": resolved_end_line,
            "total_line_count_before": total_line_count_before,
            "total_line_count_after": total_line_count_after,
            "diff": diff_text,
        }
    )
    return data


def _read_file(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    filepath = str(args["filepath"])
    content = env.workspace.read_file(filepath)
    section = _section_selectors(args)
    resolved = env.workspace.resolve_text_section(
        content,
        start_line=section["start_line"],
        end_line=section["end_line"],
        before_anchor=str(section["before_anchor"] or ""),
        after_anchor=str(section["after_anchor"] or ""),
    )
    selected_content = str(resolved["content"])
    resolved_start_line = int(resolved["start_line"])
    resolved_end_line = int(resolved["end_line"])
    returned_line_count = 0
    if resolved_start_line > 0 and resolved_end_line >= resolved_start_line:
        returned_line_count = resolved_end_line - resolved_start_line + 1
    include_line_numbers = bool(args.get("include_line_numbers", False))
    content_out = (
        _render_numbered_content(selected_content, resolved_start_line)
        if include_line_numbers
        else selected_content
    )
    data = _path_info(filepath)
    data.update(
        {
            "content": content_out,
            "size_bytes": len(content.encode("utf-8")),
            "line_count": _line_count(content),
            "section_scoped": bool(resolved["selectors_applied"]),
            "resolved_start_line": resolved_start_line,
            "resolved_end_line": resolved_end_line,
            "before_anchor_line": int(resolved["before_anchor_line"]),
            "after_anchor_line": int(resolved["after_anchor_line"]),
            "total_line_count": int(resolved["total_line_count"]),
            "returned_line_count": returned_line_count,
            "include_line_numbers": include_line_numbers,
        }
    )
    return data


def _read_files(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    max_chars = int(args.get("max_chars_per_file", 20000))
    paths = [str(item) for item in args.get("paths") or []]
    files = env.workspace.read_files(paths, max_chars_per_file=max_chars)
    return {"files": files, "count": len(files), "max_chars_per_file": max_chars}


def _list_files(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    path = str(args.get("path", "."))
    names = env.workspace.list_files(path)
    return {"path": path, "files": names, "count": len(names)}


def _search_code(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    before_context = _optional_non_negative_int(args, "before_context", 0)
    after_context = _optional_non_negative_int(args, "after_context", 0)
    return env.workspace.search_code(
        str(args["query"]),
        path=str(args.get("path", ".")),
        glob=str(args["glob"]) if "glob" in args and args.get("glob") is not None else None,
        max_results=int(args.get("max_results", 50)),
        case_sensitive=bool(args.get("case_sensitive", False)),
        fixed_strings=bool(args.get("fixed_strings", True)),
        before_context=before_context,
        after_context=after_context,
    )


def _delete_path(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
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


def _move_path(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    source_path = str(args["source_path"])
    destination_path = str(args["destination_path"])
    overwrite = bool(args.get("overwrite", False))

    source = env.workspace._resolve_read_path(source_path)
    is_dir = source.is_dir()
    moved_to = env.workspace.move_path(source_path, destination_path, overwrite=overwrite)
    data = _path_info(moved_to)
    data.update(
        {
            "moved": True,
            "kind": "directory" if is_dir else "file",
            "source_path": str(source.resolve()),
            "destination_path": moved_to,
            "overwrite": overwrite,
        }
    )
    return data


def _workspace_tree(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    max_depth = max(1, int(args.get("max_depth", 3)))
    tree = env.workspace.workspace_tree(max_depth=max_depth)
    return {"tree": tree, "max_depth": max_depth}


def _run_checks(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    return env.workspace.run_checks(
        str(args["command"]),
        args=[str(item) for item in args.get("args") or []],
        path=str(args.get("path", ".")),
        timeout_s=int(args.get("timeout_s", 120)),
    )


def execute(tool_name: str, args: dict[str, object], env: ToolExecutionEnv):
    if tool_name == "create_directory":
        return _create_directory(args, env)
    if tool_name == "create_file":
        return _create_file(args, env)
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
    if tool_name == "move_path":
        return _move_path(args, env)
    if tool_name == "delete_path":
        return _delete_path(args, env)
    if tool_name == "workspace_tree":
        return _workspace_tree(args, env)
    if tool_name == "run_checks":
        return _run_checks(args, env)
    raise ValueError(f"Unsupported tool: {tool_name}")
