from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO, cast

from core.attachments import build_content, classify_attachment
from core.configuration import config_for_editor_view, config_to_toml, load_global_config, save_global_config
from core.conv_tree import Turn
from core.message_types import JSONValue, MessageContentPart
from core.runtime_protocol import MAX_RUNTIME_FRAME_BYTES, RuntimeEmitter, RuntimeProtocolError, decode_runtime_frame
from core.sessions import ChatSession, SessionStore
from core.themes import available_theme_ids, reload_themes, theme_payload

TRANSCRIPT_PAGE_SIZE = 100
TRANSCRIPT_PAGE_BYTES = 800 * 1024
OVERSIZED_ASSISTANT_BYTES = 400 * 1024
OVERSIZED_USER_BYTES = 64 * 1024
OVERSIZED_REASONING_BYTES = 48 * 1024
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
        "reasoning": display(turn.reasoning_content),
        "tools": _tool_activity(turn)[:MAX_ACTIVITY_ITEMS],
        "activity": _activity_trace(turn, field_limit=field_limit),
        "assistant_state": turn.assistant_state,
        "parent": turn.parent,
        "children": list(turn.children),
        "label": turn.label,
        "branch_root": turn.branch_root,
        "tool_exchange_count": len(turn.skill_exchanges),
    }


def _tree_turn_view(turn: Turn) -> dict[str, Any]:
    return {
        "id": turn.id,
        "user": _clip(turn.user_text(), 120),
        "assistant_state": turn.assistant_state,
        "parent": turn.parent,
        "children": list(turn.children),
        "label": turn.label,
        "branch_root": turn.branch_root,
    }


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
    view["reasoning"] = _clip_utf8(turn.reasoning_content, OVERSIZED_REASONING_BYTES)
    view["activity"] = activity
    view["activity_truncated"] = activity_truncated

    # The fixed budgets above leave ample room for metadata, but enforce the
    # protocol invariant even for pathological legacy tool identifiers.
    while _encoded_size(view) > TRANSCRIPT_PAGE_BYTES and view["activity"]:
        cast(list[dict[str, Any]], view["activity"]).pop()
        view["activity_truncated"] = True
    while _encoded_size(view) > TRANSCRIPT_PAGE_BYTES and view["tools"]:
        cast(list[dict[str, Any]], view["tools"]).pop()
        view["tools_truncated"] = True
    if _encoded_size(view) > TRANSCRIPT_PAGE_BYTES:
        view["assistant"] = _clip_utf8(view["assistant"], OVERSIZED_ASSISTANT_BYTES // 2)
    if _encoded_size(view) > TRANSCRIPT_PAGE_BYTES:
        view = {
            "id": turn.id,
            "user": _clip_utf8(turn.user_text(), 16 * 1024),
            "assistant": _clip_utf8(turn.assistant_content, 64 * 1024),
            "reasoning": "",
            "activity": [],
            "tools": [],
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


class RuntimeServer:
    """Private, presentation-neutral owner of interactive application state."""

    def __init__(
        self,
        *,
        agent: Any,
        memory: Any,
        state_root: Path,
        config_path: Path,
        input_stream: TextIO,
        output_stream: TextIO,
    ) -> None:
        self.agent = agent
        self.memory = memory
        self.config_path = config_path
        self.emitter = RuntimeEmitter(output_stream)
        self.input_stream = input_stream
        self.store = SessionStore(state_root, storage_dir=state_root / "sessions")
        self.session = self._activate(self.store.bootstrap())
        self.pending_attachments: list[tuple[str, str]] = []
        self.stop_event = threading.Event()
        self.turn_thread: threading.Thread | None = None
        self.turn_request_id = ""
        self.turn_id = ""
        self.approvals: dict[str, tuple[threading.Event, dict[str, bool]]] = {}
        self._state_lock = threading.RLock()
        self._closed = False
        self._resources_closed = False

    def _activate(self, session: ChatSession) -> ChatSession:
        session.loaded_skill_ids = [skill.id for skill in self.agent.skill_runtime.skills_by_ids(session.loaded_skill_ids)]
        session.collaboration_mode = "plan" if session.collaboration_mode == "plan" else "execute"
        if session.tree.current_id == "root" and session.tree.nodes["root"].children:
            leaf = next(
                (node_id for node_id in reversed(session.tree.nodes) if node_id != "root" and not session.tree.nodes[node_id].children),
                None,
            )
            if leaf:
                session.tree.current_id = leaf
        return session

    def _save(self, *, title: str | None = None) -> None:
        self.session = self.store.save_tree(
            self.session.id,
            (title or self.session.title).strip() or "Untitled Session",
            self.session.tree,
            loaded_skill_ids=self.session.loaded_skill_ids,
            collaboration_mode=self.session.collaboration_mode,
            context_summary=self.session.tree.context_summary(),
            created_at=self.session.created_at,
            activate=True,
        )

    def _session_meta(self) -> dict[str, Any]:
        branch_name = (
            self.session.tree._pending_branch_label
            if self.session.tree._pending_branch
            else next(
                (turn.label for turn in reversed(self.session.tree.active_path) if turn.branch_root and turn.label),
                "root",
            )
        )
        return {
            "id": self.session.id,
            "title": self.session.title,
            "created_at": self.session.created_at,
            "updated_at": self.session.updated_at,
            "collaboration_mode": self.session.collaboration_mode,
            "loaded_skill_ids": list(self.session.loaded_skill_ids),
            "current_id": self.session.tree.current_id,
            "branch_name": str(branch_name),
            "pending_branch": self.session.tree._pending_branch,
            "pending_branch_label": self.session.tree._pending_branch_label,
        }

    def _snapshot(
        self,
        *,
        transcript_offset: int | None = None,
        tree_offset: int | None = None,
        streaming: bool | None = None,
    ) -> dict[str, Any]:
        path = [turn for turn in self.session.tree.active_path if turn.id != "root"]
        transcript_offset, transcript = _transcript_page(path, transcript_offset)
        nodes = [turn for turn in self.session.tree.nodes.values() if turn.id != "root"]
        tree_offset = max(0, tree_offset) if tree_offset is not None else max(0, len(nodes) - TREE_PAGE_SIZE)
        tree = nodes[tree_offset : tree_offset + TREE_PAGE_SIZE]
        status = self.agent.get_model_status()
        return {
            "session": self._session_meta(),
            "transcript": transcript,
            "transcript_offset": transcript_offset,
            "transcript_previous": _previous_transcript_offset(path, transcript_offset),
            "transcript_next": transcript_offset + len(transcript) if transcript_offset + len(transcript) < len(path) else None,
            "tree": [_tree_turn_view(turn) for turn in tree],
            "tree_offset": tree_offset,
            "tree_previous": max(0, tree_offset - TREE_PAGE_SIZE) if tree_offset > 0 else None,
            "tree_next": tree_offset + len(tree) if tree_offset + len(tree) < len(nodes) else None,
            "pending_attachments": [
                {"path": path, "kind": kind, "name": os.path.basename(path)} for path, kind in self.pending_attachments
            ],
            "model": {
                "state": str(status.state),
                "detail": str(status.last_error or ""),
                "model_name": str(status.model_name or ""),
                "context_window": status.context_window,
                "endpoint": str(status.endpoint or getattr(self.agent, "model_endpoint", "")),
            },
            # A worker remains alive for a few instructions after it emits
            # turn.completed.  Using Thread.is_alive() here lets a slash command
            # submitted in that window publish a stale streaming=true snapshot,
            # permanently locking the Rust composer.  turn_request_id is cleared
            # before the terminal event and therefore represents protocol-visible
            # activity rather than Python thread cleanup.
            "streaming": (bool(self.turn_request_id) if streaming is None else bool(streaming)),
        }

    def _emit_completed(self, request_id: str, data: dict[str, Any] | None = None) -> None:
        self.emitter.emit("request.completed", request_id=request_id, data=data or {})

    def _emit_error(self, request_id: str, message: str, *, category: str = "invalid_request") -> None:
        self.emitter.emit("request.error", request_id=request_id, data={"category": category, "message": message})
        self._emit_completed(request_id, {"status": "error"})

    def serve(self) -> int:
        try:
            while not self._closed:
                try:
                    raw_line = self._readline()
                except RuntimeProtocolError as exc:
                    self.emitter.emit("protocol.error", request_id="protocol", data={"message": str(exc)})
                    continue
                if raw_line is None:
                    break
                if self._closed:
                    break
                if not raw_line.strip():
                    continue
                try:
                    frame = decode_runtime_frame(raw_line)
                    self._dispatch(frame)
                except RuntimeProtocolError as exc:
                    self.emitter.emit("protocol.error", request_id="protocol", data={"message": str(exc)})
        finally:
            self.close()
        return 0

    def _readline(self) -> str | None:
        binary = getattr(self.input_stream, "buffer", None)
        if binary is not None:
            raw = binary.readline(MAX_RUNTIME_FRAME_BYTES + 1)
            if not raw:
                return None
            if len(raw) > MAX_RUNTIME_FRAME_BYTES:
                while raw and not raw.endswith(b"\n"):
                    raw = binary.readline(MAX_RUNTIME_FRAME_BYTES + 1)
                raise RuntimeProtocolError("runtime frame exceeds 1 MiB")
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RuntimeProtocolError("runtime frame is not valid UTF-8") from exc
        raw_text = self.input_stream.readline(MAX_RUNTIME_FRAME_BYTES + 1)
        if not raw_text:
            return None
        if len(raw_text.encode("utf-8")) > MAX_RUNTIME_FRAME_BYTES:
            while raw_text and not raw_text.endswith("\n"):
                raw_text = self.input_stream.readline(MAX_RUNTIME_FRAME_BYTES + 1)
            raise RuntimeProtocolError("runtime frame exceeds 1 MiB")
        return raw_text

    def _dispatch(self, frame: dict[str, Any]) -> None:
        message_type = str(frame["type"])
        request_id = str(frame["request_id"])
        data = frame.get("data", {})
        assert isinstance(data, dict)
        handlers = {
            "hello": self._hello,
            "state.get": self._state_get,
            "turn.start": self._turn_start,
            "turn.cancel": self._turn_cancel,
            "approval.resolve": self._approval_resolve,
            "session.list": self._session_list,
            "session.search": self._session_search,
            "session.create": self._session_create,
            "session.load": self._session_load,
            "session.rename": self._session_rename,
            "session.delete": self._session_delete,
            "branch.arm": self._branch_arm,
            "branch.unbranch": self._branch_unbranch,
            "branch.switch": self._branch_switch,
            "branch.open": self._branch_open,
            "attachment.add": self._attachment_add,
            "attachment.remove": self._attachment_remove,
            "status.refresh": self._status_refresh,
            "config.get": self._config_get,
            "config.apply": self._config_apply,
            "command.execute": self._command_execute,
            "theme.list": self._theme_list,
            "theme.apply": self._theme_apply,
            "palette.get": self._palette_get,
            "skill.toggle": self._skill_toggle,
            "shutdown": self._shutdown,
        }
        handler = handlers.get(message_type)
        if handler is None:
            self._emit_error(request_id, f"unknown runtime request: {message_type}")
            return
        try:
            handler(request_id, data)
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
            self._emit_error(request_id, str(exc))
        except Exception as exc:
            self._emit_error(request_id, f"{type(exc).__name__}: {exc}", category="internal")

    def _hello(self, request_id: str, data: dict[str, Any]) -> None:
        from core.ui_commands import palette_command_catalog, shortcut_catalog

        minimum = int(data.get("min_protocol", 1))
        maximum = int(data.get("max_protocol", 1))
        if not minimum <= 1 <= maximum:
            self._emit_error(request_id, "no compatible runtime protocol", category="protocol")
            return
        self.emitter.emit(
            "runtime.ready",
            request_id=request_id,
            data={
                "protocol_version": 1,
                "runtime_version": "0.2.0",
                "workspace": str(self.agent.skill_runtime.project.project_root),
                "commands": palette_command_catalog(),
                "shortcuts": shortcut_catalog(),
                "capabilities": sorted(
                    [
                        "approvals",
                        "attachments",
                        "branches",
                        "commands",
                        "config",
                        "palette",
                        "sessions",
                        "streaming",
                    ]
                ),
                "snapshot": self._snapshot(),
            },
        )
        self._emit_completed(request_id, {"status": "ok"})

    def _state_get(self, request_id: str, data: dict[str, Any]) -> None:
        transcript_offset = data.get("transcript_offset")
        tree_offset = data.get("tree_offset")
        self.emitter.emit(
            "state.snapshot",
            request_id=request_id,
            data=self._snapshot(
                transcript_offset=int(transcript_offset) if transcript_offset is not None else None,
                tree_offset=int(tree_offset) if tree_offset is not None else None,
            ),
        )
        self._emit_completed(request_id, {"status": "ok"})

    def _turn_start(self, request_id: str, data: dict[str, Any]) -> None:
        with self._state_lock:
            if self.turn_request_id:
                raise ValueError("a turn is already active")
            prompt = str(data.get("prompt") or "").strip()
            if not prompt:
                raise ValueError("turn.start requires a prompt")
            self.stop_event = threading.Event()
            content = build_content(prompt, self.pending_attachments)
            attachment_paths = [path for path, _kind in self.pending_attachments]
            self.pending_attachments.clear()
            turn = self.session.tree.add_turn(cast(JSONValue | list[MessageContentPart], content))
            self._save()
            self.turn_id = turn.id
            self.turn_request_id = request_id
            self.emitter.emit("turn.started", request_id=request_id, turn_id=turn.id, data={"turn": _turn_view(turn)})
            self.turn_thread = threading.Thread(
                target=self._run_turn,
                args=(request_id, turn, prompt, attachment_paths, bool(data.get("thinking", True))),
                daemon=True,
                name="alphanus-turn",
            )
            self.turn_thread.start()

    def _finish_turn_request(self, request_id: str) -> None:
        """Release protocol-visible turn state without clobbering a newer turn."""
        with self._state_lock:
            if self.turn_request_id == request_id:
                self.turn_request_id = ""
                self.turn_id = ""

    def _emit_turn_completion(
        self,
        request_id: str,
        turn_id: str,
        data: dict[str, Any],
        *,
        status: str,
    ) -> None:
        """Publish terminal events before allowing the next turn to start."""
        with self._state_lock:
            self.emitter.emit("turn.completed", request_id=request_id, turn_id=turn_id, data=data)
            self._emit_completed(request_id, {"status": status})
            self._finish_turn_request(request_id)

    def _run_turn(self, request_id: str, turn: Turn, prompt: str, attachment_paths: list[str], thinking: bool) -> None:
        reply_parts: list[str] = []
        reasoning_parts = _BoundedTextChunks(MAX_PERSISTED_REASONING_CHARS)
        activity_trace: list[dict[str, JSONValue]] = []
        current_reasoning_item: dict[str, JSONValue] | None = None
        current_reasoning_chunks: list[str] = []
        current_reasoning_truncated = False

        def flush_reasoning_activity() -> None:
            nonlocal current_reasoning_item, current_reasoning_chunks, current_reasoning_truncated
            if current_reasoning_item is not None:
                current_reasoning_item["text"] = "".join(current_reasoning_chunks) + (
                    "\n… [reasoning truncated]" if current_reasoning_truncated else ""
                )
            current_reasoning_item = None
            current_reasoning_chunks = []
            current_reasoning_truncated = False

        def append_reasoning_activity(text: str) -> None:
            nonlocal current_reasoning_item, current_reasoning_truncated
            if not text:
                return
            accepted = reasoning_parts.append(text)
            if current_reasoning_item is None:
                if len(activity_trace) >= MAX_ACTIVITY_ITEMS:
                    activity_trace.pop(0)
                current_reasoning_item = {"kind": "reasoning", "text": ""}
                activity_trace.append(current_reasoning_item)
            if accepted:
                current_reasoning_chunks.append(accepted)
            if len(accepted) < len(text):
                current_reasoning_truncated = True

        def append_tool_activity(payload: dict[str, Any]) -> None:
            flush_reasoning_activity()
            if len(activity_trace) >= MAX_ACTIVITY_ITEMS:
                activity_trace.pop(0)
            name = str(payload.get("name") or "tool")
            stream_id = str(payload.get("stream_id") or "")
            activity_trace.append(
                {
                    "kind": "tool",
                    "id": str(payload.get("id") or ""),
                    "name": name,
                    "completed": False,
                    **({"stream_id": stream_id} if stream_id else {}),
                    **_tool_preview_fields(name, payload),
                }
            )

        def complete_tool_activity(payload: dict[str, Any]) -> None:
            tool_id = str(payload.get("id") or "")
            name = str(payload.get("name") or "tool")
            result = payload.get("result")
            failed = isinstance(result, dict) and result.get("ok") is False
            for activity in reversed(activity_trace):
                if activity.get("kind") != "tool" or bool(activity.get("completed")):
                    continue
                if (tool_id and activity.get("id") == tool_id) or (not tool_id and activity.get("name") == name):
                    activity["completed"] = True
                    if failed:
                        activity["failed"] = True
                    activity.update(_tool_preview_fields(name, payload, completed=True))
                    return
            if len(activity_trace) >= MAX_ACTIVITY_ITEMS:
                activity_trace.pop(0)
            stream_id = str(payload.get("stream_id") or "")
            activity_trace.append(
                {
                    "kind": "tool",
                    "id": tool_id,
                    "name": name,
                    "completed": True,
                    **({"failed": True} if failed else {}),
                    **({"stream_id": stream_id} if stream_id else {}),
                    **_tool_preview_fields(name, payload, completed=True),
                }
            )

        def on_event(event: dict[str, Any]) -> None:
            event_type = str(event.get("type") or "agent.event")
            payload = {str(key): value for key, value in event.items() if key != "type"}
            if event_type == "content_token":
                reply_parts.append(str(payload.get("text") or ""))
                event_type = "assistant.delta"
            elif event_type == "reasoning_token":
                reasoning_text = str(payload.get("text") or "")
                append_reasoning_activity(reasoning_text)
                event_type = "reasoning.delta"
            elif event_type == "tool_call":
                append_tool_activity(payload)
                event_type = "tool.requested"
            elif event_type == "tool_result":
                complete_tool_activity(payload)
                event_type = "tool.completed"
            self.emitter.emit(event_type, request_id=request_id, turn_id=turn.id, data=_bounded_event_data(payload))

        def request_approval(request: dict[str, Any]) -> bool:
            approval_id = str(uuid.uuid4())
            event = threading.Event()
            holder = {"approved": False}
            self.approvals[approval_id] = (event, holder)
            self.emitter.emit(
                "approval.requested",
                request_id=request_id,
                turn_id=turn.id,
                approval_id=approval_id,
                data={"request": _bounded_event_data(request)},
            )
            timeout = float(
                self.agent.config.get("tui", {}).get("timing", {}).get("action_approval_timeout_s", 60.0)
                if isinstance(self.agent.config, dict)
                else 60.0
            )
            event.wait(max(1.0, timeout))
            self.approvals.pop(approval_id, None)
            return bool(holder["approved"]) and not self.stop_event.is_set()

        try:
            result = self.agent.run_turn(
                # The agent contract expects the active user turn in history.
                # `user_input` is also supplied for routing/classification, but is
                # not independently appended to the provider message payload.
                history_messages=self.session.tree.history_messages(),
                user_input=prompt,
                thinking=thinking,
                branch_labels=[item.label for item in self.session.tree.active_path if item.branch_root and item.label],
                attachments=attachment_paths,
                loaded_skill_ids=list(self.session.loaded_skill_ids),
                context_summary=self.session.tree.context_summary(),
                collaboration_mode=self.session.collaboration_mode,
                stop_event=self.stop_event,
                on_event=on_event,
                request_approval=request_approval,
            )
            reply = str(result.content or "") or "".join(reply_parts)
            flush_reasoning_activity()
            reasoning = _clip(
                str(getattr(result, "reasoning", "") or "") or reasoning_parts.value(),
                MAX_PERSISTED_REASONING_CHARS,
            )
            if reasoning and not any(item.get("kind") == "reasoning" for item in activity_trace):
                activity_trace.insert(0, {"kind": "reasoning", "text": reasoning})
            if self.stop_event.is_set() or result.status == "cancelled":
                self.session.tree.cancel_turn(turn.id, reply, reasoning, activity_trace)
                status = "cancelled"
            elif result.status not in {"ok", "done"}:
                self.session.tree.fail_turn(turn.id, reply, reasoning, activity_trace)
                status = "error"
            else:
                self.session.tree.complete_turn(turn.id, reply, reasoning, activity_trace)
                status = "success"
            for exchange in list(result.skill_exchanges or []):
                self.session.tree.append_skill_exchange(turn.id, exchange)
            self._save()
            self._emit_turn_completion(
                request_id,
                turn.id,
                {
                    "status": status,
                    "content": _clip(reply, COMPLETION_CONTENT_CHARS),
                    "error": str(result.error or ""),
                    "snapshot": self._snapshot(streaming=False),
                },
                status=status,
            )
        except Exception as exc:
            flush_reasoning_activity()
            self.session.tree.fail_turn(
                turn.id,
                "".join(reply_parts),
                reasoning_parts.value(),
                activity_trace,
            )
            self._save()
            self._emit_turn_completion(
                request_id,
                turn.id,
                {"status": "error", "content": "".join(reply_parts), "error": f"{type(exc).__name__}: {exc}"},
                status="error",
            )
        finally:
            self._finish_turn_request(request_id)

    def _turn_cancel(self, request_id: str, _data: dict[str, Any]) -> None:
        active = bool(self.turn_request_id)
        self.stop_event.set()
        for event, holder in list(self.approvals.values()):
            holder["approved"] = False
            event.set()
        self.emitter.emit("turn.cancellation_acknowledged", request_id=request_id, data={"active": active})
        self._emit_completed(request_id, {"status": "ok"})

    def _approval_resolve(self, request_id: str, data: dict[str, Any]) -> None:
        approval_id = str(data.get("approval_id") or "")
        pending = self.approvals.get(approval_id)
        if pending is None:
            raise ValueError("approval is no longer pending")
        event, holder = pending
        holder["approved"] = bool(data.get("approved", False))
        event.set()
        self.emitter.emit(
            "approval.resolved",
            request_id=request_id,
            approval_id=approval_id,
            data={"approved": holder["approved"]},
        )
        self._emit_completed(request_id, {"status": "ok"})

    def _require_idle(self) -> None:
        if self.turn_request_id:
            raise ValueError("stop the active response before changing sessions")

    def _session_list(self, request_id: str, data: dict[str, Any]) -> None:
        offset = max(0, int(data.get("offset", 0)))
        limit = max(1, min(int(data.get("limit", SESSION_PAGE_SIZE)), SESSION_PAGE_SIZE))
        sessions = self.store.list_sessions(limit=limit, offset=offset)
        self.emitter.emit(
            "session.list",
            request_id=request_id,
            data={
                "items": [asdict(item) for item in sessions],
                "offset": offset,
                "next": offset + len(sessions) if len(sessions) == limit else None,
            },
        )
        self._emit_completed(request_id, {"status": "ok"})

    def _session_search(self, request_id: str, data: dict[str, Any]) -> None:
        results = self.store.search_sessions(str(data.get("query") or ""), limit=min(int(data.get("limit", 80)), 200))
        self.emitter.emit("session.search", request_id=request_id, data={"items": [asdict(item) for item in results]})
        self._emit_completed(request_id, {"status": "ok"})

    def _session_create(self, request_id: str, data: dict[str, Any]) -> None:
        self._require_idle()
        self._save()
        title = str(data.get("title") or "").strip()
        if len(title) > 200:
            raise ValueError("session title must be at most 200 characters")
        self.session = self._activate(self.store.create_session(title))
        self.pending_attachments.clear()
        self.emitter.emit("state.snapshot", request_id=request_id, data=self._snapshot())
        self._emit_completed(request_id, {"status": "ok"})

    def _session_load(self, request_id: str, data: dict[str, Any]) -> None:
        self._require_idle()
        self._save()
        self.session = self._activate(self.store.load_session(str(data.get("session_id") or "")))
        turn_id = str(data.get("turn_id") or "")
        if turn_id and turn_id in self.session.tree.nodes:
            self.session.tree.current_id = turn_id
        self.pending_attachments.clear()
        self.emitter.emit("state.snapshot", request_id=request_id, data=self._snapshot())
        self._emit_completed(request_id, {"status": "ok"})

    def _session_rename(self, request_id: str, data: dict[str, Any]) -> None:
        title = str(data.get("title") or "").strip()
        if not title:
            raise ValueError("session title must not be empty")
        if len(title) > 200:
            raise ValueError("session title must be at most 200 characters")
        self._save(title=title)
        self.emitter.emit("session.changed", request_id=request_id, data={"session": self._session_meta()})
        self._emit_completed(request_id, {"status": "ok"})

    def _session_delete(self, request_id: str, data: dict[str, Any]) -> None:
        self._require_idle()
        session_id = str(data.get("session_id") or "")
        self.store.delete_session(session_id)
        if session_id == self.session.id:
            remaining = self.store.list_sessions(limit=1)
            self.session = self._activate(self.store.load_session(remaining[0].id) if remaining else self.store.create_session())
        self.emitter.emit("state.snapshot", request_id=request_id, data=self._snapshot())
        self._emit_completed(request_id, {"status": "ok"})

    def _branch_arm(self, request_id: str, data: dict[str, Any]) -> None:
        self.session.tree.arm_branch(str(data.get("label") or ""))
        self._save()
        self.emitter.emit("state.snapshot", request_id=request_id, data=self._snapshot())
        self._emit_completed(request_id, {"status": "ok"})

    def _branch_unbranch(self, request_id: str, _data: dict[str, Any]) -> None:
        if self.session.tree._pending_branch:
            self.session.tree.clear_pending_branch()
        elif self.session.tree.unbranch() is None:
            raise ValueError("no branch to leave")
        self._save()
        self.emitter.emit("state.snapshot", request_id=request_id, data=self._snapshot())
        self._emit_completed(request_id, {"status": "ok"})

    def _branch_switch(self, request_id: str, data: dict[str, Any]) -> None:
        index = int(data.get("index", 0))
        if self.session.tree.switch_child(index) is None:
            raise ValueError(f"no child branch at index {index}")
        self._save()
        self.emitter.emit("state.snapshot", request_id=request_id, data=self._snapshot())
        self._emit_completed(request_id, {"status": "ok"})

    def _branch_open(self, request_id: str, data: dict[str, Any]) -> None:
        turn_id = str(data.get("turn_id") or "")
        if turn_id not in self.session.tree.nodes:
            raise ValueError("unknown turn")
        self.session.tree.current_id = turn_id
        self._save()
        self.emitter.emit("state.snapshot", request_id=request_id, data=self._snapshot())
        self._emit_completed(request_id, {"status": "ok"})

    def _attachment_add(self, request_id: str, data: dict[str, Any]) -> None:
        path = Path(str(data.get("path") or "")).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        kind = classify_attachment(str(path))
        if kind not in {"image", "text"}:
            raise ValueError(f"unsupported attachment: {path.name}")
        item = (str(path), kind)
        if item not in self.pending_attachments:
            self.pending_attachments.append(item)
        self.emitter.emit("attachments.changed", request_id=request_id, data={"items": self._snapshot()["pending_attachments"]})
        self._emit_completed(request_id, {"status": "ok"})

    def _attachment_remove(self, request_id: str, data: dict[str, Any]) -> None:
        target = data.get("index", "last")
        if target == "all":
            self.pending_attachments.clear()
        elif target == "last":
            if self.pending_attachments:
                self.pending_attachments.pop()
        else:
            index = int(target)
            if index < 0 or index >= len(self.pending_attachments):
                raise ValueError("attachment index is out of range")
            self.pending_attachments.pop(index)
        self.emitter.emit("attachments.changed", request_id=request_id, data={"items": self._snapshot()["pending_attachments"]})
        self._emit_completed(request_id, {"status": "ok"})

    def _status_refresh(self, request_id: str, _data: dict[str, Any]) -> None:
        timeout = min(float(getattr(self.agent, "connect_timeout_s", 2.0)), 2.0)
        status = self.agent.refresh_model_status(timeout_s=timeout, force=True)
        self.emitter.emit(
            "status.changed",
            request_id=request_id,
            data={
                "state": str(status.state),
                "detail": str(status.last_error or ""),
                "model_name": str(status.model_name or ""),
                "context_window": status.context_window,
                "endpoint": str(status.endpoint or getattr(self.agent, "model_endpoint", "")),
            },
        )
        self._emit_completed(request_id, {"status": "ok"})

    def _config_get(self, request_id: str, _data: dict[str, Any]) -> None:
        config = load_global_config(self.config_path)
        self.emitter.emit(
            "config.value",
            request_id=request_id,
            data={"text": config_to_toml(config_for_editor_view(config)), "path": str(self.config_path)},
        )
        self._emit_completed(request_id, {"status": "ok"})

    def _config_apply(self, request_id: str, data: dict[str, Any]) -> None:
        text = str(data.get("text") or "")
        import tomllib

        parsed = tomllib.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("configuration must be a TOML table")
        save_global_config(self.config_path, parsed)
        self.emitter.emit("config.changed", request_id=request_id, data={"restart_required": True})
        self._emit_completed(request_id, {"status": "ok"})

    def _command_execute(self, request_id: str, data: dict[str, Any]) -> None:
        from core.ui_commands import execute_ui_command

        result = execute_ui_command(self, str(data.get("command") or ""))
        self.emitter.emit("command.result", request_id=request_id, data=result)
        if bool(result.get("state_changed")):
            self.emitter.emit("state.snapshot", request_id=request_id, data=self._snapshot())
        self._emit_completed(request_id, {"status": "ok" if result.get("ok", True) else "error"})

    def _theme_list(self, request_id: str, _data: dict[str, Any]) -> None:
        configured = str(self.agent.config.get("tui", {}).get("theme") or "")
        items = [theme_payload(theme_id) for theme_id in available_theme_ids()]
        self.emitter.emit(
            "theme.list",
            request_id=request_id,
            data={"items": items, "active": theme_payload(configured)},
        )
        self._emit_completed(request_id, {"status": "ok"})

    def _theme_apply(self, request_id: str, data: dict[str, Any]) -> None:
        theme_id = str(data.get("theme_id") or "").strip()
        if theme_id not in available_theme_ids():
            raise ValueError(f"unknown theme: {theme_id}")
        config = load_global_config(self.config_path)
        tui = config.setdefault("tui", {})
        if not isinstance(tui, dict):
            raise ValueError("config tui section must be a table")
        tui["theme"] = theme_id
        save_global_config(self.config_path, config)
        self.agent.config.setdefault("tui", {})["theme"] = theme_id
        reload_themes()
        self.emitter.emit("theme.changed", request_id=request_id, data={"theme": theme_payload(theme_id)})
        self._emit_completed(request_id, {"status": "ok"})

    def _palette_get(self, request_id: str, _data: dict[str, Any]) -> None:
        from core.ui_commands import palette_command_catalog

        items: list[dict[str, Any]] = list(palette_command_catalog())
        for summary in self.store.list_sessions(limit=20):
            items.append(
                {
                    "kind": "session",
                    "value": summary.id,
                    "prompt": summary.title,
                    "description": f"{summary.turn_count} turns",
                }
            )
        root = Path(self.agent.skill_runtime.project.project_root).resolve()
        skip = {".git", ".hg", ".svn", ".alphanus", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache"}
        file_count = 0
        for current, directories, names in os.walk(root):
            directories[:] = sorted(name for name in directories if name not in skip and not name.startswith("."))
            for name in sorted(names):
                if name.startswith("."):
                    continue
                path = (Path(current) / name).resolve()
                if classify_attachment(str(path)) not in {"image", "text"}:
                    continue
                items.append({"kind": "file", "value": str(path), "prompt": str(path.relative_to(root)), "description": "attach file"})
                file_count += 1
                if file_count >= 60:
                    break
            if file_count >= 60:
                break
        loaded = set(self.session.loaded_skill_ids)
        for skill in self.agent.skill_runtime.list_skills():
            if skill.id not in loaded and (not skill.enabled or not skill.available):
                continue
            items.append(
                {
                    "kind": "skill",
                    "value": skill.id,
                    "prompt": skill.id,
                    "description": "unload skill" if skill.id in loaded else "load skill",
                    "loaded": skill.id in loaded,
                }
            )
        self.emitter.emit("palette.items", request_id=request_id, data={"items": items})
        self._emit_completed(request_id, {"status": "ok"})

    def _skill_toggle(self, request_id: str, data: dict[str, Any]) -> None:
        skill_id = str(data.get("skill_id") or "").strip()
        skill = self.agent.skill_runtime.get_skill(skill_id)
        if skill is None or not skill.enabled or not skill.available:
            raise ValueError(f"skill is unavailable: {skill_id}")
        if skill_id in self.session.loaded_skill_ids:
            self.session.loaded_skill_ids = [item for item in self.session.loaded_skill_ids if item != skill_id]
            loaded = False
        else:
            self.session.loaded_skill_ids.append(skill_id)
            loaded = True
        self._save()
        self.emitter.emit("skill.changed", request_id=request_id, data={"skill_id": skill_id, "loaded": loaded})
        self._emit_completed(request_id, {"status": "ok"})

    def _shutdown(self, request_id: str, _data: dict[str, Any]) -> None:
        self.stop_event.set()
        for event, holder in list(self.approvals.values()):
            holder["approved"] = False
            event.set()
        self._emit_completed(request_id, {"status": "ok"})
        self._closed = True

    def close(self) -> None:
        if self._resources_closed:
            return
        self._closed = True
        self._resources_closed = True
        self.stop_event.set()
        if self.turn_thread and self.turn_thread.is_alive():
            self.turn_thread.join(timeout=5.0)
        try:
            self._save()
        except Exception:
            pass
        try:
            self.store.close()
        except Exception:
            pass
        try:
            self.memory.flush()
            close = getattr(self.memory, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
