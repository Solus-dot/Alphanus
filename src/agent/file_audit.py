from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from core.message_types import ChatMessage, JSONValue
from core.types import ToolExecutionRecord

_FILE_TOOLS = {"create_file", "create_directory", "edit_file", "move_path", "delete_path"}


def _relativize_path(path: object, workspace_root: str | Path | None) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    if workspace_root is None:
        return text
    root = Path(workspace_root).expanduser().resolve()
    raw = Path(text).expanduser()
    try:
        candidate = raw.resolve(strict=False) if raw.is_absolute() else (root / raw).resolve(strict=False)
        return candidate.relative_to(root).as_posix()
    except (OSError, ValueError):
        return text


def _num(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _base_row(action: str, tool: str, *, status: str) -> dict[str, JSONValue]:
    return {"action": action, "tool": tool, "status": status}


def _file_row_from_success(name: str, data: dict[str, object], workspace_root: str | Path | None) -> dict[str, JSONValue] | None:
    if name == "create_file":
        row = _base_row("created", name, status="success")
        row["path"] = _relativize_path(data.get("filepath"), workspace_root)
        row["kind"] = "file"
        if (value := _num(data.get("bytes_written"))) is not None:
            row["bytes"] = value
        if (value := _num(data.get("line_count"))) is not None:
            row["lines"] = value
        return row if row.get("path") else None

    if name == "create_directory":
        row = _base_row("created", name, status="success")
        row["path"] = _relativize_path(data.get("filepath"), workspace_root)
        row["kind"] = "directory"
        return row if row.get("path") else None

    if name == "edit_file":
        row = _base_row("edited", name, status="success")
        row["path"] = _relativize_path(data.get("filepath"), workspace_root)
        row["kind"] = str(data.get("kind") or "file")
        row["changed"] = bool(data.get("changed"))
        if (value := _num(data.get("changed_lines"))) is not None:
            row["changed_lines"] = value
        if (value := _num(data.get("bytes_before"))) is not None:
            row["bytes_before"] = value
        if (value := _num(data.get("bytes_after"))) is not None:
            row["bytes_after"] = value
        diff = str(data.get("diff") or "")
        if diff:
            row["diff"] = diff
        return row if row.get("path") else None

    if name == "move_path":
        row = _base_row("moved", name, status="success")
        row["from"] = _relativize_path(data.get("source_path"), workspace_root)
        row["to"] = _relativize_path(data.get("destination_path") or data.get("filepath"), workspace_root)
        row["kind"] = str(data.get("kind") or "path")
        row["overwrite"] = bool(data.get("overwrite"))
        return row if row.get("from") or row.get("to") else None

    if name == "delete_path":
        row = _base_row("deleted", name, status="success")
        row["path"] = _relativize_path(data.get("filepath"), workspace_root)
        row["kind"] = str(data.get("kind") or "path")
        row["recursive"] = bool(data.get("recursive"))
        if (value := _num(data.get("size_bytes"))) is not None:
            row["bytes"] = value
        if (value := _num(data.get("file_count"))) is not None:
            row["file_count"] = value
        return row if row.get("path") else None

    return None


def _file_row_from_failure(name: str, args: dict[str, object], result: dict[str, object], workspace_root: str | Path | None) -> dict[str, JSONValue] | None:
    if name not in _FILE_TOOLS:
        return None
    action = {
        "create_file": "created",
        "create_directory": "created",
        "edit_file": "edited",
        "move_path": "moved",
        "delete_path": "deleted",
    }.get(name, "changed")
    row = _base_row(action, name, status="failed")
    if name == "move_path":
        row["from"] = _relativize_path(args.get("source_path"), workspace_root)
        row["to"] = _relativize_path(args.get("destination_path"), workspace_root)
    else:
        path_key = "path" if name in {"create_directory", "delete_path"} else "filepath"
        row["path"] = _relativize_path(args.get(path_key), workspace_root)
    error = result.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or "").strip()
        code = str(error.get("code") or "").strip()
        if message:
            row["error"] = message
        if code:
            row["error_code"] = code
    return row if row.get("path") or row.get("from") or row.get("to") else None


def _shell_row(name: str, result: dict[str, object]) -> dict[str, JSONValue] | None:
    if name != "shell_command":
        return None
    meta = result.get("meta")
    if not isinstance(meta, dict) or not bool(meta.get("workspace_changed")):
        return None
    data = result.get("data")
    data_obj = data if isinstance(data, dict) else {}
    row = _base_row("workspace_changed", name, status="success" if result.get("ok") else "failed")
    row["paths_known"] = False
    row["command"] = str(data_obj.get("command") or "")
    row["cwd"] = str(data_obj.get("cwd") or "")
    if (value := _num(data_obj.get("returncode"))) is not None:
        row["returncode"] = value
    if (value := _num(meta.get("duration_ms"))) is not None:
        row["duration_ms"] = value
    return row


def _row_from_tool_result(
    name: str,
    args: dict[str, object],
    result: dict[str, object],
    workspace_root: str | Path | None,
) -> dict[str, JSONValue] | None:
    if shell_row := _shell_row(name, result):
        return shell_row
    data = result.get("data")
    data_obj = data if isinstance(data, dict) else {}
    if result.get("ok"):
        return _file_row_from_success(name, data_obj, workspace_root)
    return _file_row_from_failure(name, args, result, workspace_root)


def build_file_audit_from_evidence(
    evidence: list[ToolExecutionRecord],
    *,
    workspace_root: str | Path | None = None,
) -> list[dict[str, JSONValue]]:
    rows: list[dict[str, JSONValue]] = []
    for item in evidence:
        row = _row_from_tool_result(item.name, cast(dict[str, object], item.args), cast(dict[str, object], item.result), workspace_root)
        if row is not None:
            rows.append(row)
    return rows


def _tool_calls_from_message(message: ChatMessage) -> list[tuple[str, str, dict[str, object]]]:
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []
    out: list[tuple[str, str, dict[str, object]]] = []
    for raw_call in raw_calls:
        call = raw_call if isinstance(raw_call, dict) else {}
        call_id = str(call.get("id") or "")
        function = call.get("function")
        fn = function if isinstance(function, dict) else {}
        name = str(fn.get("name") or "").strip()
        raw_args = fn.get("arguments")
        args: object = {}
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        if name:
            out.append((call_id, name, cast(dict[str, object], args if isinstance(args, dict) else {})))
    return out


def build_file_audit_from_skill_exchanges(
    skill_exchanges: list[ChatMessage],
    *,
    workspace_root: str | Path | None = None,
) -> list[dict[str, JSONValue]]:
    rows: list[dict[str, JSONValue]] = []
    pending_by_id: dict[str, tuple[str, dict[str, object]]] = {}
    pending_by_name: dict[str, list[dict[str, object]]] = {}
    for message in skill_exchanges:
        if message.get("role") == "assistant":
            for call_id, name, args in _tool_calls_from_message(message):
                if call_id:
                    pending_by_id[call_id] = (name, args)
                pending_by_name.setdefault(name, []).append(args)
            continue
        if message.get("role") != "tool":
            continue
        name = str(message.get("name") or "").strip()
        raw_content = message.get("content")
        try:
            result = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
        except json.JSONDecodeError:
            result = {}
        if not name or not isinstance(result, dict):
            continue
        args: dict[str, object] = {}
        tool_call_id = str(message.get("tool_call_id") or "")
        if tool_call_id and tool_call_id in pending_by_id:
            _call_name, args = pending_by_id.pop(tool_call_id)
        elif pending_by_name.get(name):
            args = pending_by_name[name].pop(0)
        row = _row_from_tool_result(name, args, cast(dict[str, object], result), workspace_root)
        if row is not None:
            rows.append(row)
    return rows
