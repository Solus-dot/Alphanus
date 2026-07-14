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
TREE_PAGE_SIZE = 250
SESSION_PAGE_SIZE = 100
TRANSCRIPT_FIELD_CHARS = 768
EVENT_DATA_BYTES = 256 * 1024


def _clip(value: Any, limit: int = TRANSCRIPT_FIELD_CHARS) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit} characters omitted]"


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


def _turn_view(turn: Turn, *, field_limit: int = TRANSCRIPT_FIELD_CHARS) -> dict[str, Any]:
    return {
        "id": turn.id,
        "user": _clip(turn.user_text(), field_limit),
        "attachments": _clip(turn.attachment_summary(), min(256, field_limit)),
        "assistant": _clip(turn.assistant_content, field_limit),
        "assistant_state": turn.assistant_state,
        "parent": turn.parent,
        "children": list(turn.children),
        "label": turn.label,
        "branch_root": turn.branch_root,
        "tool_exchange_count": len(turn.skill_exchanges),
    }


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
        return {
            "id": self.session.id,
            "title": self.session.title,
            "created_at": self.session.created_at,
            "updated_at": self.session.updated_at,
            "collaboration_mode": self.session.collaboration_mode,
            "loaded_skill_ids": list(self.session.loaded_skill_ids),
            "current_id": self.session.tree.current_id,
            "pending_branch": self.session.tree._pending_branch,
            "pending_branch_label": self.session.tree._pending_branch_label,
        }

    def _snapshot(self, *, transcript_offset: int | None = None, tree_offset: int | None = None) -> dict[str, Any]:
        path = [turn for turn in self.session.tree.active_path if turn.id != "root"]
        transcript_offset = max(0, transcript_offset) if transcript_offset is not None else max(0, len(path) - TRANSCRIPT_PAGE_SIZE)
        transcript = path[transcript_offset : transcript_offset + TRANSCRIPT_PAGE_SIZE]
        nodes = [turn for turn in self.session.tree.nodes.values() if turn.id != "root"]
        tree_offset = max(0, tree_offset) if tree_offset is not None else max(0, len(nodes) - TREE_PAGE_SIZE)
        tree = nodes[tree_offset : tree_offset + TREE_PAGE_SIZE]
        status = self.agent.get_model_status()
        return {
            "session": self._session_meta(),
            "transcript": [_turn_view(turn) for turn in transcript],
            "transcript_offset": transcript_offset,
            "transcript_previous": max(0, transcript_offset - TRANSCRIPT_PAGE_SIZE) if transcript_offset > 0 else None,
            "transcript_next": transcript_offset + len(transcript) if transcript_offset + len(transcript) < len(path) else None,
            "tree": [_turn_view(turn, field_limit=128) for turn in tree],
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
            },
            "streaming": bool(self.turn_thread and self.turn_thread.is_alive()),
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
            if self.turn_thread and self.turn_thread.is_alive():
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

    def _run_turn(self, request_id: str, turn: Turn, prompt: str, attachment_paths: list[str], thinking: bool) -> None:
        reply_parts: list[str] = []

        def on_event(event: dict[str, Any]) -> None:
            event_type = str(event.get("type") or "agent.event")
            payload = {str(key): value for key, value in event.items() if key != "type"}
            if event_type == "content_token":
                reply_parts.append(str(payload.get("text") or ""))
                event_type = "assistant.delta"
            elif event_type == "reasoning_token":
                event_type = "reasoning.delta"
            elif event_type == "tool_call":
                event_type = "tool.requested"
            elif event_type == "tool_result":
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
                history_messages=self.session.tree.history_messages()[:-1],
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
            if self.stop_event.is_set() or result.status == "cancelled":
                self.session.tree.cancel_turn(turn.id, reply)
                status = "cancelled"
            elif result.status not in {"ok", "done"}:
                self.session.tree.fail_turn(turn.id, reply)
                status = "error"
            else:
                self.session.tree.complete_turn(turn.id, reply)
                status = "success"
            for exchange in list(result.skill_exchanges or []):
                self.session.tree.append_skill_exchange(turn.id, exchange)
            self._save()
            self.emitter.emit(
                "turn.completed",
                request_id=request_id,
                turn_id=turn.id,
                data={"status": status, "content": reply, "error": str(result.error or ""), "snapshot": self._snapshot()},
            )
            self._emit_completed(request_id, {"status": status})
        except Exception as exc:
            self.session.tree.fail_turn(turn.id, "".join(reply_parts))
            self._save()
            self.emitter.emit(
                "turn.completed",
                request_id=request_id,
                turn_id=turn.id,
                data={"status": "error", "content": "".join(reply_parts), "error": f"{type(exc).__name__}: {exc}"},
            )
            self._emit_completed(request_id, {"status": "error"})
        finally:
            self.turn_id = ""
            self.turn_request_id = ""

    def _turn_cancel(self, request_id: str, _data: dict[str, Any]) -> None:
        active = bool(self.turn_thread and self.turn_thread.is_alive())
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
        if self.turn_thread and self.turn_thread.is_alive():
            raise ValueError("stop the active response before changing sessions")

    def _session_list(self, request_id: str, data: dict[str, Any]) -> None:
        offset = max(0, int(data.get("offset", 0)))
        limit = max(1, min(int(data.get("limit", SESSION_PAGE_SIZE)), SESSION_PAGE_SIZE))
        sessions = self.store.list_sessions(limit=limit, offset=offset)
        self.emitter.emit(
            "session.list",
            request_id=request_id,
            data={"items": [asdict(item) for item in sessions], "offset": offset, "next": offset + len(sessions) if len(sessions) == limit else None},
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
        self.emitter.emit(
            "attachments.changed", request_id=request_id, data={"items": self._snapshot()["pending_attachments"]}
        )
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
        self.emitter.emit(
            "attachments.changed", request_id=request_id, data={"items": self._snapshot()["pending_attachments"]}
        )
        self._emit_completed(request_id, {"status": "ok"})

    def _status_refresh(self, request_id: str, _data: dict[str, Any]) -> None:
        status = self.agent.refresh_model_status()
        self.emitter.emit(
            "status.changed",
            request_id=request_id,
            data={"state": str(status.state), "detail": str(status.last_error or ""), "model_name": str(status.model_name or "")},
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
        from core.ui_commands import command_catalog

        items: list[dict[str, Any]] = [
            {"kind": "command", "value": row["command"].split()[0], "prompt": row["command"], "description": row["description"]}
            for row in command_catalog()
        ]
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
                items.append(
                    {"kind": "file", "value": str(path), "prompt": str(path.relative_to(root)), "description": "attach file"}
                )
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
