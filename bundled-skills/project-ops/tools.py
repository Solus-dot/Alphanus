from __future__ import annotations

import difflib
import hashlib
import re
from pathlib import Path

from skills.runtime import ToolExecutionEnv

WRITE_PREVIEW_CHARS = 1200
WRITE_PREVIEW_HEAD_CHARS = 800
WRITE_PREVIEW_TAIL_CHARS = 400
READ_CONTENT_MAX_CHARS = 64000
EDIT_DIFF_MAX_CHARS = 12000

TOOL_SPECS = {
    "create_directory": {
        "capability": "project_write",
        "mutates": True,
        "actions": ["create"],
        "description": "Create a directory in the project, or at an explicit absolute path when policy allows it.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    "create_file": {
        "capability": "project_write",
        "mutates": True,
        "actions": ["create", "write", "save"],
        "description": "Create or overwrite a file in the project, or at an explicit absolute path when policy allows it.",
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
        "capability": "project_edit",
        "mutates": True,
        "actions": ["edit", "update", "write"],
        "description": (
            "Edit an existing project file, or an explicit absolute file when policy allows it. Supports exactly one mode per call: full content replace, "
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
        "capability": "project_read",
        "mutates": False,
        "actions": ["read"],
        "description": "Read a project file, or an explicit absolute file when policy allows it. Optional line-range and anchor selectors can return a specific section.",
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
        "capability": "project_read",
        "mutates": False,
        "actions": ["read"],
        "description": "Read multiple project or explicit absolute files with per-file truncation metadata.",
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
        "capability": "project_read",
        "mutates": False,
        "actions": ["list", "read"],
        "description": "List one directory level when the exact directory is already known. Use find_files to locate unknown files.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": [],
        },
    },
    "find_files": {
        "capability": "project_read",
        "mutates": False,
        "actions": ["list", "read", "check"],
        "description": "Find project files by exact name, path fragment, or glob without walking directories one level at a time.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "name": {"type": "string"},
                "glob": {"type": "string"},
                "include_dirs": {"type": "boolean"},
                "case_sensitive": {"type": "boolean"},
                "max_results": {"type": "integer"},
            },
            "required": [],
        },
    },
    "search_code": {
        "capability": "project_read",
        "mutates": False,
        "actions": ["read", "check"],
        "description": "Search the project codebase with ripgrep when available. Optional context lines can be returned per match.",
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
        "capability": "project_write",
        "mutates": True,
        "actions": ["move", "rename"],
        "description": "Move or rename a project path, or an explicit absolute path when policy allows it.",
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
        "capability": "project_delete",
        "mutates": True,
        "actions": ["delete", "remove"],
        "description": "Delete a project path, or an explicit absolute path when policy allows it.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {"type": "boolean"},
            },
            "required": ["path"],
        },
    },
    "project_tree": {
        "capability": "project_tree",
        "mutates": False,
        "actions": ["list", "read"],
        "description": "Render a bounded tree for the project or a specific allowed directory path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_depth": {"type": "integer"},
                "max_entries": {"type": "integer"},
            },
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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _line_count_from_text(text: str) -> int:
    return text.count("\n") + (1 if text else 0)


def _middle_truncate(text: str, limit: int) -> tuple[str, bool, int]:
    if limit <= 0 or len(text) <= limit:
        return text, False, 0
    if limit <= 32:
        omitted = len(text) - limit
        return text[:limit], True, omitted
    head_len = max(1, limit // 2)
    tail_len = max(1, limit - head_len)
    omitted = len(text) - head_len - tail_len
    marker = f"\n...[{omitted} chars truncated]...\n"
    budget_for_text = max(2, limit - len(marker))
    head_len = max(1, min(head_len, budget_for_text // 2))
    tail_len = max(1, budget_for_text - head_len)
    omitted = len(text) - head_len - tail_len
    marker = f"\n...[{omitted} chars truncated]...\n"
    return text[:head_len] + marker + text[-tail_len:], True, omitted


def _bounded_write_preview(text: str) -> tuple[str, bool, int]:
    if len(text) <= WRITE_PREVIEW_CHARS:
        return text, False, 0
    omitted = len(text) - WRITE_PREVIEW_HEAD_CHARS - WRITE_PREVIEW_TAIL_CHARS
    marker = f"\n...[{omitted} chars omitted from write preview]...\n"
    return text[:WRITE_PREVIEW_HEAD_CHARS] + marker + text[-WRITE_PREVIEW_TAIL_CHARS:], True, omitted


def _bounded_diff(text: str) -> tuple[str, bool, int]:
    return _middle_truncate(text, EDIT_DIFF_MAX_CHARS)


def _content_payload(content: str, *, limit: int) -> dict[str, object]:
    returned, truncated, omitted = _middle_truncate(content, limit)
    total_chars = len(content)
    payload: dict[str, object] = {
        "content": returned,
        "content_truncated": truncated,
        "total_chars": total_chars,
        "returned_chars": len(returned),
        "omitted_chars": omitted,
        "truncation": None,
    }
    if truncated:
        payload["truncation"] = (
            f"Warning: truncated output (original char count: {total_chars})\n"
            f"Total output lines: {_line_count_from_text(content)}"
        )
    return payload


def _path_info(path_str: str) -> dict[str, object]:
    path = Path(path_str)
    return {
        "filepath": path_str,
        "basename": path.name,
    }


def _require_approval(env: ToolExecutionEnv, request: dict[str, object]) -> bool:
    if env.project.permission_mode == "danger-full-access":
        return True
    if not env.request_approval:
        raise PermissionError("Approval callback is required")
    approved = bool(env.request_approval(request))
    if not approved:
        raise PermissionError("Project operation rejected by user")
    return approved


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
    filepath = str(args["filepath"])
    approved = False
    if env.project.write_path_requires_approval(filepath, overwrite=True):
        approved = _require_approval(
            env,
            {
                "kind": "project_file_write",
                "path": filepath,
                "operation": "overwrite",
                "reason": "External file overwrite crosses the project-write approval boundary.",
            },
        )
    path_str = env.project.create_file(filepath, content, approved=approved)
    preview, preview_truncated, preview_omitted = _bounded_write_preview(content)
    data = _path_info(path_str)
    data.update(
        {
            "created": True,
            "write_verified": True,
            "sha256": _sha256_text(content),
            "content_preview": preview,
            "content_preview_truncated": preview_truncated,
            "preview_chars": len(preview),
            "preview_omitted_chars": preview_omitted,
            "bytes_written": len(content.encode("utf-8")),
            "chars_written": len(content),
            "line_count": _line_count_from_text(content),
        }
    )
    return data


def _create_directory(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    path_str = env.project.create_directory(str(args["path"]))
    data = _path_info(path_str)
    data.update({"created": True, "kind": "directory"})
    return data


def _edit_file(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    filepath = str(args["filepath"])
    before = env.project.read_file(filepath)
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
        resolved = env.project.resolve_text_section(
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

    path_str = env.project.edit_file(filepath, after)
    diff_text = "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"{filepath} (before)",
            tofile=f"{filepath} (after)",
            lineterm="",
        )
    )
    diff_out, diff_truncated, diff_omitted = _bounded_diff(diff_text)
    preview, preview_truncated, preview_omitted = _bounded_write_preview(after)
    data = _path_info(path_str)
    data.update(
        {
            "edited": True,
            "changed": before != after,
            "sha256": _sha256_text(after),
            "content_preview": preview,
            "content_preview_truncated": preview_truncated,
            "preview_chars": len(preview),
            "preview_omitted_chars": preview_omitted,
            "bytes_before": len(before.encode("utf-8")),
            "bytes_after": len(after.encode("utf-8")),
            "line_count_before": _line_count_from_text(before),
            "line_count_after": _line_count_from_text(after),
            "changed_lines": _changed_line_count(before, after),
            "edit_mode": edit_mode,
            "replacements_applied": replacements_applied,
            "section_scoped": not has_content_mode and has_section_selectors,
            "resolved_start_line": resolved_start_line,
            "resolved_end_line": resolved_end_line,
            "total_line_count_before": total_line_count_before,
            "total_line_count_after": total_line_count_after,
            "diff": diff_out,
            "diff_truncated": diff_truncated,
            "diff_omitted_chars": diff_omitted,
        }
    )
    return data


def _read_file(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    filepath = str(args["filepath"])
    content = env.project.read_file(filepath)
    section = _section_selectors(args)
    resolved = env.project.resolve_text_section(
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
    content_payload = _content_payload(content_out, limit=READ_CONTENT_MAX_CHARS)
    data = _path_info(filepath)
    data.update(
        {
            **content_payload,
            "size_bytes": len(content.encode("utf-8")),
            "line_count": _line_count_from_text(content),
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
    max_chars = min(max(1, int(args.get("max_chars_per_file", 20000))), READ_CONTENT_MAX_CHARS)
    paths = [str(item) for item in args.get("paths") or []]
    files = []
    for path in paths:
        content = env.project.read_file(path)
        payload = _content_payload(content, limit=max_chars)
        payload.update(
            {
                "filepath": path,
                "size_bytes": len(content.encode("utf-8")),
                "line_count": _line_count_from_text(content),
                "truncated": bool(payload["content_truncated"]),
            }
        )
        files.append(payload)
    return {"files": files, "count": len(files), "max_chars_per_file": max_chars}


def _list_files(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    path = str(args.get("path", "."))
    names = env.project.list_files(path)
    return {"path": path, "files": names, "count": len(names)}


def _find_files(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    return env.project.find_files(
        path=str(args.get("path", ".")),
        name=str(args.get("name", "")),
        glob=str(args.get("glob", "")),
        include_dirs=bool(args.get("include_dirs", False)),
        case_sensitive=bool(args.get("case_sensitive", False)),
        max_results=int(args.get("max_results", 50)),
    )


def _search_code(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    before_context = _optional_non_negative_int(args, "before_context", 0)
    after_context = _optional_non_negative_int(args, "after_context", 0)
    return env.project.search_code(
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
    approved = False
    if env.project.delete_path_requires_approval(path, recursive=recursive):
        approved = _require_approval(
            env,
            {
                "kind": "project_path_delete",
                "path": path,
                "recursive": recursive,
                "reason": "Delete crosses the project-write approval boundary.",
            },
        )
    metadata = env.project.delete_path_metadata(path)
    is_dir = bool(metadata["is_dir"])
    path_str = env.project.delete_path(path, recursive=recursive, approved=approved)
    data = _path_info(path_str)
    data.update(
        {
            "deleted": True,
            "recursive": recursive,
            "kind": "directory" if is_dir else "file",
            "size_bytes": int(metadata["size_bytes"]),
            "file_count": int(metadata["file_count"]),
        }
    )
    return data


def _move_path(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    source_path = str(args["source_path"])
    destination_path = str(args["destination_path"])
    overwrite = bool(args.get("overwrite", False))
    approved = False
    if env.project.move_path_requires_approval(source_path, destination_path, overwrite=overwrite):
        approved = _require_approval(
            env,
            {
                "kind": "project_path_move",
                "source_path": source_path,
                "destination_path": destination_path,
                "overwrite": overwrite,
                "reason": "Move crosses the project-write approval boundary.",
            },
        )

    source = env.project._resolve_read_path(source_path)
    is_dir = source.is_dir()
    moved_to = env.project.move_path(source_path, destination_path, overwrite=overwrite, approved=approved)
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


def _project_tree(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    max_depth = max(1, int(args.get("max_depth", 3)))
    max_entries = max(1, int(args.get("max_entries", 500)))
    return env.project.project_tree(
        path=str(args.get("path", ".")),
        max_depth=max_depth,
        max_entries=max_entries,
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
    if tool_name == "find_files":
        return _find_files(args, env)
    if tool_name == "search_code":
        return _search_code(args, env)
    if tool_name == "move_path":
        return _move_path(args, env)
    if tool_name == "delete_path":
        return _delete_path(args, env)
    if tool_name == "project_tree":
        return _project_tree(args, env)
    raise ValueError(f"Unsupported tool: {tool_name}")
