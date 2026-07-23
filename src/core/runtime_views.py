from __future__ import annotations

import json
from typing import Any, cast

from core.conv_tree import Turn
from core.message_types import JSONValue

TRANSCRIPT_PAGE_SIZE = 100
TRANSCRIPT_PAGE_BYTES = 800 * 1024
OVERSIZED_ASSISTANT_BYTES = 400 * 1024
OVERSIZED_USER_BYTES = 64 * 1024
OVERSIZED_ACTIVITY_BYTES = 128 * 1024
TREE_PAGE_SIZE = 250
SESSION_PAGE_SIZE = 100
TRANSCRIPT_FIELD_CHARS = 768
EVENT_DATA_BYTES = 256 * 1024
COMPLETION_CONTENT_CHARS = 64 * 1024
MAX_ACTIVITY_ITEMS = 256
MAX_PERSISTED_REASONING_CHARS = 512 * 1024
TOOL_PREVIEW_CHARS = 8_000
TOOL_PREVIEW_LINES = 140


def _clip(value: Any, limit: int = TRANSCRIPT_FIELD_CHARS) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit} characters omitted]"


def _clip_utf8(value: Any, limit: int) -> str:
    text = str(value or "")
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    marker = f"\n… [{len(encoded) - limit} bytes omitted]"
    content_limit = max(0, limit - len(marker.encode("utf-8")))
    clipped = encoded[:content_limit].decode("utf-8", errors="ignore")
    return clipped + marker


class _BoundedTextChunks:
    __slots__ = ("_chunks", "_length", "_limit", "_truncated")

    def __init__(self, limit: int) -> None:
        self._chunks: list[str] = []
        self._length = 0
        self._limit = max(0, int(limit))
        self._truncated = False

    def append(self, text: str) -> str:
        if not text or self._length >= self._limit:
            self._truncated = self._truncated or bool(text)
            return ""
        remaining = self._limit - self._length
        accepted = text[:remaining]
        if accepted:
            self._chunks.append(accepted)
            self._length += len(accepted)
        if len(accepted) < len(text):
            self._truncated = True
        return accepted

    def value(self) -> str:
        text = "".join(self._chunks)
        return text + ("\n… [reasoning truncated]" if self._truncated else "")


def _bounded_event_data(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError):
        return {"truncated": True, "message": "runtime event contained non-JSON data"}
    if len(encoded) <= EVENT_DATA_BYTES:
        return payload
    bounded: dict[str, Any] = {"truncated": True, "original_bytes": len(encoded)}
    for key in ("name", "status", "path", "command", "reason", "text"):
        if key in payload:
            bounded[key] = _clip(payload[key])
    return bounded


def _tool_activity(turn: Turn) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for message in turn.skill_exchanges:
        role = str(message.get("role") or "")
        if role == "assistant":
            calls = message.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for call in calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                name = str(function.get("name") or "tool") if isinstance(function, dict) else "tool"
                tool_id = str(call.get("id") or "")
                item = {"id": tool_id, "name": name, "completed": False}
                tools.append(item)
                if tool_id:
                    by_id[tool_id] = item
        elif role == "tool":
            tool_id = str(message.get("tool_call_id") or "")
            name = str(message.get("name") or "tool")
            failed = False
            content = message.get("content")
            if isinstance(content, str):
                try:
                    decoded = json.loads(content)
                except json.JSONDecodeError:
                    decoded = None
                failed = isinstance(decoded, dict) and decoded.get("ok") is False
            match = by_id.get(tool_id) if tool_id else None
            if match is None:
                match = next(
                    (item for item in reversed(tools) if not item["completed"] and item["name"] == name),
                    None,
                )
            if match is None:
                tools.append({"id": tool_id, "name": name, "completed": True, **({"failed": True} if failed else {})})
            else:
                match["completed"] = True
                if failed:
                    match["failed"] = True
    return tools


def _activity_trace(turn: Turn, *, field_limit: int | None = TRANSCRIPT_FIELD_CHARS) -> list[dict[str, Any]]:
    if turn.activity_trace:
        activity = [dict(item) for item in turn.activity_trace]
        if field_limit is not None:
            for item in activity:
                if item.get("kind") == "reasoning":
                    item["text"] = _clip(item.get("text"), field_limit)
        return activity
    # Older sessions did not record event chronology. Preserve their content in
    # the least surprising order while all newly-created turns use exact events.
    activity: list[dict[str, Any]] = []
    if turn.reasoning_content:
        activity.append({"kind": "reasoning", "text": turn.reasoning_content})
    activity.extend({"kind": "tool", **item} for item in _tool_activity(turn))
    return activity


def _bounded_tool_preview(content: str) -> tuple[str, bool]:
    lines = content.splitlines()
    preview = "\n".join(lines[:TOOL_PREVIEW_LINES])
    truncated = len(lines) > TOOL_PREVIEW_LINES
    if len(preview) > TOOL_PREVIEW_CHARS:
        preview = preview[:TOOL_PREVIEW_CHARS]
        truncated = True
    return preview, truncated


def _tool_preview_fields(name: str, payload: dict[str, Any], *, completed: bool = False) -> dict[str, JSONValue]:
    canonical = name.split(":")[-1].split(".")[-1]
    if canonical not in {"create_file", "edit_file"}:
        return {}
    source = payload.get("result") if completed else payload.get("arguments")
    if not isinstance(source, dict):
        return {}
    if completed:
        data = source.get("data")
        if not isinstance(data, dict):
            return {}
        filepath = str(data.get("filepath") or "")
        content = data.get("diff") if canonical == "edit_file" else data.get("content_preview")
        language = "diff" if canonical == "edit_file" and isinstance(content, str) else ""
        already_truncated = bool(data.get("diff_truncated") or data.get("content_preview_truncated"))
    else:
        filepath = str(source.get("filepath") or "")
        content = source.get("content")
        language = ""
        already_truncated = False
    fields: dict[str, JSONValue] = {"filepath": filepath}
    if isinstance(content, str) and content:
        preview, truncated = _bounded_tool_preview(content)
        fields.update(
            {
                "preview": preview,
                "preview_truncated": truncated or already_truncated,
                "language": language,
            }
        )
    return fields


def _turn_view(turn: Turn, *, field_limit: int | None = TRANSCRIPT_FIELD_CHARS) -> dict[str, Any]:
    def display(value: Any, limit: int | None = field_limit) -> str:
        return str(value or "") if limit is None else _clip(value, limit)

    return {
        "id": turn.id,
        "user": display(turn.user_text()),
        "attachments": display(turn.attachment_summary(), 256),
        "assistant": display(turn.assistant_content),
        "activity": _activity_trace(turn, field_limit=field_limit),
        "assistant_state": turn.assistant_state,
        "parent": turn.parent,
        "children": list(turn.children),
        "label": turn.label,
        "branch_root": turn.branch_root,
        "tool_exchange_count": len(turn.skill_exchanges),
    }


def _tree_turn_view(turn: Turn) -> dict[str, Any]:
    view = _turn_view(turn)
    view["user"] = _clip(turn.user_text(), 120)
    return {key: view[key] for key in ("id", "user", "assistant_state", "parent", "children", "label", "branch_root")}


def _encoded_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _bounded_activity_view(turn: Turn, byte_limit: int) -> tuple[list[dict[str, Any]], bool]:
    selected: list[dict[str, Any]] = []
    used = 2  # JSON list delimiters.
    truncated = False
    for raw_item in _activity_trace(turn, field_limit=None):
        item = dict(raw_item)
        if item.get("kind") == "reasoning":
            item["text"] = _clip_utf8(item.get("text"), 16 * 1024)
        elif item.get("kind") == "tool" and "preview" in item:
            item["preview"] = _clip_utf8(item.get("preview"), TOOL_PREVIEW_CHARS)
        size = _encoded_size(item) + (1 if selected else 0)
        if len(selected) >= MAX_ACTIVITY_ITEMS or used + size > byte_limit:
            truncated = True
            break
        selected.append(item)
        used += size
    return selected, truncated


def _bounded_full_turn_view(turn: Turn) -> dict[str, Any]:
    view = _turn_view(turn, field_limit=None)
    if _encoded_size(view) <= TRANSCRIPT_PAGE_BYTES:
        return view
    activity, activity_truncated = _bounded_activity_view(turn, OVERSIZED_ACTIVITY_BYTES)
    view = _turn_view(turn, field_limit=None)
    view["user"] = _clip_utf8(turn.user_text(), OVERSIZED_USER_BYTES)
    view["assistant"] = _clip_utf8(turn.assistant_content, OVERSIZED_ASSISTANT_BYTES)
    view["activity"] = activity
    view["activity_truncated"] = activity_truncated

    # The fixed budgets above leave ample room for metadata, but enforce the
    # protocol invariant even for pathological legacy tool identifiers.
    while _encoded_size(view) > TRANSCRIPT_PAGE_BYTES and view["activity"]:
        cast(list[dict[str, Any]], view["activity"]).pop()
        view["activity_truncated"] = True
    if _encoded_size(view) > TRANSCRIPT_PAGE_BYTES:
        view["assistant"] = _clip_utf8(view["assistant"], OVERSIZED_ASSISTANT_BYTES // 2)
    if _encoded_size(view) > TRANSCRIPT_PAGE_BYTES:
        view = {
            "id": turn.id,
            "user": _clip_utf8(turn.user_text(), 16 * 1024),
            "assistant": _clip_utf8(turn.assistant_content, 64 * 1024),
            "activity": [],
            "assistant_state": turn.assistant_state,
            "parent": turn.parent,
            "children": list(turn.children),
            "label": turn.label,
            "branch_root": turn.branch_root,
            "content_truncated": True,
        }
    while _encoded_size(view) > TRANSCRIPT_PAGE_BYTES and view["children"]:
        cast(list[str], view["children"]).pop()
        view["children_truncated"] = True
    if _encoded_size(view) > TRANSCRIPT_PAGE_BYTES:
        for key in ("id", "parent", "label"):
            view[key] = _clip_utf8(view.get(key), 1024)
    return view


def _transcript_page(path: list[Turn], offset: int | None) -> tuple[int, list[dict[str, Any]]]:
    if not path:
        return 0, []
    if offset is None:
        selected: list[dict[str, Any]] = []
        total = 0
        start = len(path)
        for index in range(len(path) - 1, -1, -1):
            view = _bounded_full_turn_view(path[index])
            size = _encoded_size(view)
            if selected and (len(selected) >= TRANSCRIPT_PAGE_SIZE or total + size > TRANSCRIPT_PAGE_BYTES):
                break
            selected.append(view)
            total += size
            start = index
        selected.reverse()
        return start, selected

    start = min(max(0, offset), len(path))
    selected = []
    total = 0
    for turn in path[start : start + TRANSCRIPT_PAGE_SIZE]:
        view = _bounded_full_turn_view(turn)
        size = _encoded_size(view)
        if selected and total + size > TRANSCRIPT_PAGE_BYTES:
            break
        selected.append(view)
        total += size
    return start, selected


def _previous_transcript_offset(path: list[Turn], current_offset: int) -> int | None:
    if current_offset <= 0:
        return None
    start, _page = _transcript_page(path[:current_offset], None)
    return start
