from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent.types import AgentTurnResult
from tui.markdown_utils import render_md

ASSISTANT_MESSAGE_BAR_COLOR = "#6366f1"


@dataclass(slots=True)
class StreamTextState:
    reasoning_open: bool = False
    content_open: bool = False
    reasoning_buffer: str = ""
    content_buffer: str = ""
    in_fence: bool = False
    fence_lang: Optional[str] = None
    fence_lines: List[str] = field(default_factory=list)


@dataclass(slots=True)
class StreamRuntimeState:
    event_queue: queue.SimpleQueue[Dict[str, object]] = field(default_factory=queue.SimpleQueue)
    drain_active: bool = False
    partial_dirty: bool = False
    deferred_live_preview: Optional[Tuple[List[str], Optional[str]]] = None
    text: StreamTextState = field(default_factory=StreamTextState)


def _state(app: Any) -> StreamRuntimeState:
    state = getattr(app, "_stream_runtime", None)
    if state is None:
        state = StreamRuntimeState()
        app._stream_runtime = state
    return state


def start_turn_stream(app: Any, turn: Any, user_input: str, attachment_paths: List[str]) -> None:
    app.streaming = True
    app._auto_follow_stream = True
    app._active_turn_id = turn.id
    app._reply_acc = ""
    app._live_preview.reset()
    app._pending_tool_details = []
    app._last_stream_error_text = ""
    app._stream_runtime = StreamRuntimeState()
    app._reasoning_open = False
    app._content_open = False
    app._buf_r = ""
    app._buf_c = ""
    app._reset_fence_state()
    app._stop_event = threading.Event()
    app._partial().display = True
    app._write("")
    branch_labels = [t.label for t in app.conv_tree.active_path if t.branch_root and t.label]
    history_messages = app.conv_tree.history_messages()
    app._stream_worker(
        turn.id,
        history_messages,
        user_input,
        branch_labels,
        attachment_paths,
        list(app._loaded_skill_ids),
        app.thinking,
        app._stop_event,
    )


def enqueue_event(app: Any, event: Dict[str, Any]) -> None:
    _state(app).event_queue.put(event)
    if event.get("type") in {"tool_phase_started", "tool_call", "tool_result", "error", "info", "pass_end", "usage"}:
        app.call_from_thread(app._drain_stream_event_queue)


def visible_reasoning_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?tool_call>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?function(?:=[^>]+)?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?parameter(?:=[^>]+)?>", "", text, flags=re.IGNORECASE)
    return text


def flush_reasoning_buffer(app: Any) -> None:
    if not app._buf_r:
        app._set_partial_renderable(None)
        return
    text = app._buf_r
    app._buf_r = ""
    visible = "\n".join(line for line in visible_reasoning_text(text).splitlines() if not app._is_tool_trace_line(line)).strip()
    app._set_partial_renderable(None)
    if visible:
        app._write_assistant_bar_renderable(app._reasoning_panel_renderable(visible))


def close_reasoning_section(app: Any) -> bool:
    if not app._reasoning_open:
        return False
    flush_reasoning_buffer(app)
    app._reasoning_open = False
    return True


def refresh_deferred_partial(app: Any) -> None:
    stream_state = _state(app)
    if stream_state.deferred_live_preview is not None:
        lines, language = stream_state.deferred_live_preview
        stream_state.deferred_live_preview = None
        app._update_live_preview_partial(lines, language)
        return
    if not stream_state.partial_dirty:
        return
    stream_state.partial_dirty = False
    if app._reasoning_open and not app._content_open:
        display = "\n".join(
            line for line in visible_reasoning_text(app._buf_r).splitlines() if not app._is_tool_trace_line(line)
        ).strip()
        if display:
            app._set_partial_renderable(
                app._bar_renderable(app._reasoning_panel_renderable(display), ASSISTANT_MESSAGE_BAR_COLOR)
            )
        else:
            app._set_partial_renderable(None)
        return
    app._update_partial_content()


def flush_content_buffer(app: Any, include_partial: bool = False, *, update_partial: bool = True) -> None:
    while "\n" in app._buf_c:
        line, app._buf_c = app._buf_c.split("\n", 1)
        if app._is_tool_trace_line(line):
            continue
        app._render_content_line(line)

    if include_partial and app._buf_c:
        if not app._is_tool_trace_line(app._buf_c):
            if app._in_fence:
                if app._is_fence_line(app._buf_c):
                    app._flush_fence_block()
                else:
                    app._fence_lines.append(app._buf_c)
                    app._flush_fence_block()
            else:
                rendered, _ = render_md(app._buf_c, False)
                app._write_assistant_bar_wrapped_line(app._buf_c, rendered)
        app._buf_c = ""
        app._set_partial_renderable(None)
        return

    if update_partial:
        app._update_partial_content()
    else:
        _state(app).partial_dirty = True


def handle_content_token(app: Any, token: str, *, update_partial: bool = True) -> None:
    if not app._content_open:
        app._content_open = True
        if app._close_reasoning_section():
            app._write_assistant_bar_line()
    app._buf_c += token
    app._append_reply_token(token)
    app._flush_content_buffer(include_partial=False, update_partial=update_partial)


def on_agent_event(app: Any, event: Dict[str, Any]) -> None:
    etype = event.get("type")
    stream_drain_active = bool(_state(app).drain_active)

    stop_event = getattr(app, "_stop_event", None)
    if (
        stop_event is not None
        and stop_event.is_set()
        and etype in {"reasoning_token", "content_token", "tool_phase_started", "tool_call_delta"}
    ):
        return

    if etype == "reasoning_token":
        token = event.get("text", "")
        if not app._reasoning_open:
            app._reasoning_open = True
        app._buf_r += token
        if stream_drain_active:
            _state(app).partial_dirty = True
        else:
            display = "\n".join(
                line for line in visible_reasoning_text(app._buf_r).splitlines() if not app._is_tool_trace_line(line)
            ).strip()
            if display:
                app._set_partial_renderable(
                    app._bar_renderable(app._reasoning_panel_renderable(display), ASSISTANT_MESSAGE_BAR_COLOR)
                )
            else:
                app._set_partial_renderable(None)

    elif etype == "content_token":
        token = event.get("text", "")
        app._handle_content_token(token, update_partial=not stream_drain_active)

    elif etype == "tool_phase_started":
        app._close_reasoning_section()
        app._flush_content_buffer(include_partial=True, update_partial=True)
        app._content_open = False

    elif etype == "tool_call_delta":
        if not app._show_tool_details:
            return
        stream_id = str(event.get("stream_id") or "")
        name = str(event.get("name") or "")
        raw_arguments = str(event.get("raw_arguments") or "")
        if stream_id and name:
            rendered_preview = app._live_preview.update(
                stream_id,
                name,
                raw_arguments,
                app._write_assistant_bar_line,
                app._defer_live_preview_partial if stream_drain_active else app._update_live_preview_partial,
                lambda markup, _indent=0: app._write_assistant_bar_line(markup),
                app._write_code_block,
                app._clear_partial_preview,
            )
            if not rendered_preview:
                app._update_tool_call_partial(name, "streaming…")

    elif etype == "tool_call":
        close_reasoning_section(app)
        name = event.get("name", "tool")
        args = event.get("arguments", {})
        stream_id = str(event.get("stream_id") or "")
        detail = app._live_preview.compact_tool_args(name, args)
        if app._show_tool_details:
            app._pending_tool_details.append((name, detail))
            app._update_tool_call_partial(name, detail)
            streamed = (
                app._live_preview.close(
                    stream_id,
                    lambda markup, _indent=0: app._write_assistant_bar_line(markup),
                    app._write_code_block,
                    app._clear_partial_preview,
                    retain_partial=True,
                )
                if stream_id
                else False
            )
            if not streamed:
                app._live_preview.write_static_preview(
                    name,
                    args,
                    app._write_assistant_bar_line,
                    lambda markup, _indent=0: app._write_assistant_bar_line(markup),
                    app._write_code_block,
                )
        elif stream_id:
            app._live_preview.close(
                stream_id,
                lambda markup, _indent=0: app._write_assistant_bar_line(markup),
                app._write_code_block,
                app._clear_partial_preview,
            )

    elif etype == "tool_result":
        close_reasoning_section(app)
        name = event.get("name", "tool")
        result = event.get("result", {})
        if result.get("ok"):
            app._live_preview.write_result_preview(
                name,
                result,
                app._write_assistant_bar_line,
                lambda markup, _indent=0: app._write_assistant_bar_line(markup),
                app._write_code_block,
            )
        if name not in app._live_preview.draft_preview_tools or not result.get("ok"):
            app._clear_partial_preview()
        if not app._show_tool_result_line(name, bool(result.get("ok"))):
            return
        detail = app._take_pending_tool_detail(name)
        if result.get("ok"):
            app._write_tool_lifecycle_block(name, True, detail or "completed")
        else:
            msg = result.get("error", {}).get("message", "failed")
            app._write_tool_lifecycle_block(name, False, f"{detail}   {msg}".strip())

    elif etype == "error":
        app._last_stream_error_text = str(event.get("text", "Unknown error"))
        app._write_error(app._last_stream_error_text)

    elif etype == "info":
        app._write_info(str(event.get("text", "")))

    elif etype == "pass_end":
        finish_reason = str(event.get("finish_reason") or "")
        has_content = bool(event.get("has_content"))
        has_tool_calls = bool(event.get("has_tool_calls"))
        if finish_reason in {"stop", "length"} and not has_content and not has_tool_calls:
            app._buf_r = ""
            app._set_partial_renderable(None)

    elif etype == "usage":
        usage = event.get("usage") or {}
        if isinstance(usage, dict):
            app._update_context_usage_from_payload(usage)

    if stream_drain_active:
        return

    now = time.monotonic()
    if now - app._last_scroll >= app._scroll_interval:
        app._maybe_scroll_end()
        app._last_scroll = now


def drain_events(app: Any) -> None:
    stream_state = _state(app)
    if stream_state.drain_active:
        return

    queued: List[Dict[str, Any]] = []
    while True:
        try:
            queued.append(stream_state.event_queue.get_nowait())
        except queue.Empty:
            break

    if not queued:
        return

    stream_state.drain_active = True
    try:
        for event in queued:
            if event.get("type") in {"tool_phase_started", "tool_call", "tool_result", "error", "info", "pass_end", "usage"}:
                refresh_deferred_partial(app)
            on_agent_event(app, event)
        refresh_deferred_partial(app)
        now = time.monotonic()
        if now - app._last_scroll >= app._scroll_interval:
            app._maybe_scroll_end()
            app._last_scroll = now
    finally:
        stream_state.drain_active = False


def finish_turn_stream(app: Any, turn_id: str, result: AgentTurnResult) -> None:
    finalization_error = ""
    cleanup_errors: List[str] = []

    def run_cleanup_step(label: str, fn) -> None:
        try:
            fn()
        except Exception as exc:
            cleanup_errors.append(f"{label}: {exc}")

    try:
        drain_events(app)
        app._set_partial_renderable(None, visible=False)

        app._live_preview.close_all(
            lambda markup, _indent=0: app._write_assistant_bar_line(markup),
            app._write_code_block,
            app._clear_partial_preview,
        )

        if app._buf_r and not app._content_open:
            close_reasoning_section(app)
        elif app._reasoning_open and not app._content_open:
            close_reasoning_section(app)

        flush_content_buffer(app, include_partial=True)

        reply = result.content if result.content else app._reply_acc
        if result.status == "done" and not app._content_open and reply.strip():
            app._render_static_markdown(reply)

        if turn_id in app.conv_tree.nodes:
            for msg in result.skill_exchanges:
                app.conv_tree.append_skill_exchange(turn_id, msg)

        if result.status == "done":
            app.conv_tree.complete_turn(turn_id, reply)
        elif result.status == "cancelled":
            app.conv_tree.cancel_turn(turn_id, reply)
        else:
            app.conv_tree.fail_turn(turn_id, reply)
        usage = result.journal.get("model_usage", {}) if isinstance(result.journal, dict) else {}
        if isinstance(usage, dict):
            app._update_context_usage_from_payload(usage)
        loaded_skill_ids = result.journal.get("loaded_skill_ids", []) if isinstance(result.journal, dict) else []
        if isinstance(loaded_skill_ids, list):
            app._loaded_skill_ids = [
                skill.id
                for skill in app.agent.skill_runtime.skills_by_ids(
                    [str(item).strip() for item in loaded_skill_ids if str(item).strip()]
                )
            ]

        if result.status != "done":
            if result.error and result.error != app._last_stream_error_text:
                app._write_error(result.error)
            if result.status == "cancelled":
                app._write("[bold red]  ✖ interrupted[/bold red]")

        app._save_active_session()
        app._write("")
        app._maybe_scroll_end()
    except Exception as exc:
        finalization_error = str(exc).strip() or exc.__class__.__name__
    finally:
        run_cleanup_step("clear partial renderable", lambda: app._set_partial_renderable(None, visible=False))
        app.streaming = False
        app._live_preview.reset()
        app._active_turn_id = None
        app._esc_pending = False
        app._auto_follow_stream = True
        run_cleanup_step("update status1", app._update_status1)
        run_cleanup_step("update status2", app._update_status2)
        run_cleanup_step("update sidebar", app._update_sidebar)
        run_cleanup_step("update topbar", app._update_topbar)
        if finalization_error or cleanup_errors:
            parts: List[str] = []
            if finalization_error:
                parts.append(finalization_error)
            parts.extend(cleanup_errors)
            error_text = f"Stream finalization failed: {'; '.join(parts)}"
            try:
                app._write_error(error_text)
            except Exception:
                logging.error(error_text)
