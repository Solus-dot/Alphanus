from __future__ import annotations

import queue
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from core.types import AgentTurnResult
from tui.activity_runtime import ActivityState
from tui.stream_runtime import (
    StreamRuntimeState,
    drain_events,
    enqueue_event,
    finish_turn_stream,
    start_turn_stream,
    visible_reasoning_text,
)


class _Turn:
    def __init__(self, turn_id: str) -> None:
        self.id = turn_id


class _App:
    _stream_runtime: StreamRuntimeState | None
    _stop_event: Any
    _reasoning_open: bool
    _show_tool_details: bool
    _handle_content_token: Callable[..., Any]
    _write_assistant_bar_line: Callable[..., Any]
    _write_code_block: Callable[..., Any]
    _clear_partial_preview: Callable[..., Any]
    _show_tool_result_line: Callable[..., Any]
    _take_pending_tool_detail: Callable[..., Any]
    _write_tool_lifecycle_block: Callable[..., Any]
    _activity_state: ActivityState

    def __init__(self) -> None:
        self.streaming = False
        self._auto_follow_stream = False
        self._active_turn_id = None
        self._reply_acc = "stale"
        self._pending_tool_details = ["stale"]
        self._last_stream_error_text = "stale"
        self._stream_runtime = StreamRuntimeState()
        self._activity_state = ActivityState()
        self._stop_event = None
        self._loaded_skill_ids = ["a", "b"]
        self.thinking = True
        self.conv_tree = SimpleNamespace(active_path=[], history_messages=lambda: [], context_summary=lambda *_args: "branch summary")
        self._stream_worker_calls = []
        self._call_from_thread_calls = []
        self._usage_updates = 0
        self._sidebar_updates = 0
        self._last_scroll = 0.0
        self._scroll_interval = 999.0
        self._live_preview = SimpleNamespace(reset=lambda: None)
        self._reasoning_open = False
        self._show_tool_details = True
        self._handle_content_token = lambda *_args, **_kwargs: None
        self._write_assistant_bar_line = lambda *_args, **_kwargs: None
        self._write_code_block = lambda *_args, **_kwargs: None
        self._clear_partial_preview = lambda: None
        self._show_tool_result_line = lambda *_args, **_kwargs: True
        self._take_pending_tool_detail = lambda *_args, **_kwargs: ""
        self._write_tool_lifecycle_block = lambda *_args, **_kwargs: None

    def _write(self, _markup: str) -> None:
        return None

    def _partial(self):
        return SimpleNamespace(display=False)

    def _reset_fence_state(self) -> None:
        return None

    def _stream_worker(self, *args):
        self._stream_worker_calls.append(args)

    def call_from_thread(self, fn):
        self._call_from_thread_calls.append(fn)

    def _drain_stream_event_queue(self):
        return None

    def _update_context_usage_from_payload(self, _payload):
        self._usage_updates += 1

    def _maybe_scroll_end(self):
        return None

    def _update_sidebar(self):
        self._sidebar_updates += 1
        return None


def test_visible_reasoning_text_strips_internal_tags() -> None:
    text = "<think>r</think><tool_call><function=x></function></tool_call>answer"
    assert visible_reasoning_text(text) == "ranswer"


def test_start_turn_stream_resets_runtime_and_invokes_worker() -> None:
    app = _App()
    turn = _Turn("t1")

    start_turn_stream(app, turn, "hi", ["a.txt"])

    assert app.streaming is True
    assert app._active_turn_id == "t1"
    assert app._reply_acc == ""
    assert app._sidebar_updates == 1
    assert app._stream_worker_calls
    call = app._stream_worker_calls[0]
    assert call[0] == "t1"
    assert call[2] == "hi"
    assert call[4] == ["a.txt"]
    assert call[6] == "branch summary"


def test_enqueue_event_creates_queue_and_schedules_drain_for_notable_events() -> None:
    app = _App()
    app._stream_runtime = None

    enqueue_event(app, {"type": "tool_call", "name": "x"})

    assert isinstance(app._stream_runtime, StreamRuntimeState)
    queued = app._stream_runtime.event_queue.get_nowait()
    assert queued["type"] == "tool_call"
    assert app._call_from_thread_calls


def test_enqueue_event_coalesces_tool_call_delta_until_timer_tick() -> None:
    app = _App()

    enqueue_event(app, {"type": "tool_call_delta", "stream_id": "s1", "name": "create_file", "raw_arguments": "{}"})

    assert app._call_from_thread_calls == []
    assert app._stream_runtime is not None
    queued = app._stream_runtime.event_queue.get_nowait()
    assert queued["type"] == "tool_call_delta"


def test_drain_events_processes_usage_event() -> None:
    app = _App()
    assert app._stream_runtime is not None
    app._stream_runtime.event_queue = queue.SimpleQueue()
    app._stream_runtime.event_queue.put({"type": "usage", "usage": {"prompt_tokens": 5}})

    drain_events(app)

    assert app._usage_updates == 1
    assert app._stream_runtime is not None
    assert app._stream_runtime.drain_active is False


def test_tool_call_applies_final_arguments_before_closing_live_preview() -> None:
    app = _App()
    calls: list[tuple[Any, ...]] = []

    def apply_final_arguments(stream_id, name, args):
        calls.append(("apply", stream_id, name, args))

    def close(stream_id, *_args, **kwargs):
        calls.append(("close", stream_id, kwargs.get("retain_partial", False)))
        return True

    app._live_preview = SimpleNamespace(
        compact_tool_args=lambda name, args: f"filepath={args.get('filepath', '')}",
        apply_final_arguments=apply_final_arguments,
        close=close,
        write_static_preview=lambda *_args, **_kwargs: calls.append(("static",)),
    )
    assert app._stream_runtime is not None
    app._stream_runtime.event_queue = queue.SimpleQueue()
    app._stream_runtime.event_queue.put(
        {
            "type": "tool_call",
            "stream_id": "s1",
            "name": "create_file",
            "arguments": {"filepath": "RPS.html", "content": "<main>RPS</main>"},
        }
    )

    drain_events(app)

    assert calls[0] == ("apply", "s1", "create_file", {"filepath": "RPS.html", "content": "<main>RPS</main>"})
    assert calls[1] == ("close", "s1", False)
    assert ("static",) not in calls


def test_drain_events_ignores_content_events_after_stop_requested() -> None:
    app = _App()
    app._stop_event = SimpleNamespace(is_set=lambda: True)
    handled: list[str] = []
    app._handle_content_token = lambda token, update_partial=True: handled.append(token)
    assert app._stream_runtime is not None
    app._stream_runtime.event_queue = queue.SimpleQueue()
    app._stream_runtime.event_queue.put({"type": "content_token", "text": "still streaming"})

    drain_events(app)

    assert handled == []


def test_drain_events_keeps_tool_result_events_after_stop_requested() -> None:
    app = _App()
    app._stop_event = SimpleNamespace(is_set=lambda: True)
    app._reasoning_open = False
    app._show_tool_details = True
    events: list[str | tuple[str, int]] = []
    app._live_preview = SimpleNamespace(
        write_result_preview=lambda *_args, **_kwargs: events.append("preview"),
        draft_preview_tools=set(),
    )
    app._write_assistant_bar_line = lambda _markup="": None
    app._write_code_block = lambda _lines, _language, _indent=2: None
    app._clear_partial_preview = lambda: events.append("clear")
    app._show_tool_result_line = lambda name, ok: events.append(f"line:{name}:{ok}") or True
    app._take_pending_tool_detail = lambda _name: ""
    app._write_tool_lifecycle_block = lambda name, ok, detail="": events.append(f"lifecycle:{name}:{ok}")
    assert app._stream_runtime is not None
    app._stream_runtime.event_queue = queue.SimpleQueue()
    app._stream_runtime.event_queue.put(
        {"type": "tool_result", "name": "create_file", "result": {"ok": True, "data": {}, "error": None, "meta": {}}}
    )

    drain_events(app)

    assert "line:create_file:True" in events
    assert "lifecycle:create_file:True" in events


def test_drain_events_previews_failed_shell_output() -> None:
    app = _App()
    app._show_tool_details = False
    events: list[str | tuple[str, int]] = []
    code_blocks: list[tuple[list[str], str | None, int]] = []

    def supports_result_preview(name, result):
        return name == "shell_command" and isinstance(result.get("data"), dict)

    def write_result_preview(name, result, write, write_indented, write_code):
        data = result["data"]
        write(f"shell: {data['command']}")
        write_indented("stderr:", 2)
        write_code([data["stderr"]], "text", 2)

    app._live_preview = SimpleNamespace(
        draft_preview_tools=set(),
        supports_result_preview=supports_result_preview,
        write_result_preview=write_result_preview,
    )
    app._write_assistant_bar_line = lambda markup="", content_indent=0: events.append((markup, content_indent))
    app._write_code_block = lambda lines, language, indent=2: code_blocks.append((list(lines), language, indent))
    app._clear_partial_preview = lambda: events.append("clear")
    app._show_tool_result_line = lambda name, ok: events.append(f"line:{name}:{ok}") or True
    app._take_pending_tool_detail = lambda _name: ""
    app._write_tool_lifecycle_block = lambda name, ok, detail="": events.append(f"lifecycle:{name}:{ok}:{detail}")
    assert app._stream_runtime is not None
    app._stream_runtime.event_queue = queue.SimpleQueue()
    app._stream_runtime.event_queue.put(
        {
            "type": "tool_result",
            "name": "shell_command",
            "result": {
                "ok": False,
                "data": {"command": "pytest -q", "stderr": "failed", "returncode": 1},
                "error": {"message": "Command exited with code 1"},
                "meta": {"duration_ms": 7},
            },
        }
    )

    drain_events(app)

    assert ("shell: pytest -q", 0) in events
    assert ("stderr:", 2) in events
    assert ("line:shell_command:False") in events
    assert code_blocks == [(["failed"], "text", 2)]
    assert app._activity_state.rows[0].message == "Command exited with code 1"


def test_drain_events_updates_activity_for_tool_lifecycle() -> None:
    app = _App()
    app._show_tool_details = False
    app._live_preview = SimpleNamespace(
        compact_tool_args=lambda name, args: f"args={len(args)}",
        draft_preview_tools=set(),
        write_result_preview=lambda *_args, **_kwargs: None,
    )
    assert app._stream_runtime is not None
    app._stream_runtime.event_queue = queue.SimpleQueue()
    app._stream_runtime.event_queue.put({"type": "tool_call", "name": "read_file", "arguments": {"filepath": "README.md"}})
    app._stream_runtime.event_queue.put(
        {
            "type": "tool_result",
            "name": "read_file",
            "result": {"ok": True, "data": {}, "error": None, "meta": {"duration_ms": 3}},
        }
    )

    drain_events(app)

    assert app._activity_state.rows[0].name == "read_file"
    assert app._activity_state.rows[0].status == "done"
    assert app._activity_state.rows[0].duration_ms == 3
    assert app._sidebar_updates >= 2


def test_drain_events_writes_shell_running_preview_on_tool_call() -> None:
    app = _App()
    events: list[tuple[str, int]] = []
    app._write_assistant_bar_line = lambda markup="", content_indent=0: events.append((markup, content_indent))
    app._live_preview = SimpleNamespace(
        compact_tool_args=lambda _name, args: f"command={args['command']}, timeout_s={args['timeout_s']}",
        write_shell_running_preview=lambda name, args, write: write(f"running:{name}:{args['command']}:{args['timeout_s']}"),
        apply_final_arguments=lambda *_args, **_kwargs: None,
        close=lambda *_args, **_kwargs: False,
        write_static_preview=lambda *_args, **_kwargs: None,
    )
    assert app._stream_runtime is not None
    app._stream_runtime.event_queue = queue.SimpleQueue()
    app._stream_runtime.event_queue.put(
        {"type": "tool_call", "name": "shell_command", "arguments": {"command": "llama-update", "timeout_s": 1200}}
    )

    drain_events(app)

    assert ("running:shell_command:llama-update:1200", 0) in events
    assert app._activity_state.rows[0].name == "shell_command"
    assert app._activity_state.rows[0].status == "running"
    assert "timeout_s=1200" in app._activity_state.rows[0].detail


def test_finish_turn_stream_restores_ui_even_when_finalization_raises() -> None:
    events: list[str] = []

    class _FinishApp:
        def __init__(self) -> None:
            self.streaming = True
            self._active_turn_id = "t1"
            self._esc_pending = True
            self._auto_follow_stream = False
            self._reply_acc = "done"
            self._buf_r = ""
            self._buf_c = ""
            self._content_open = False
            self._reasoning_open = False
            self._last_stream_error_text = ""
            self._pending_tool_details = []
            self._stream_runtime = StreamRuntimeState()
            self._loaded_skill_ids = []
            self.conv_tree = SimpleNamespace(
                nodes={"t1": object()},
                append_skill_exchange=lambda turn_id, msg: events.append(f"skill:{turn_id}:{msg.get('role')}"),
                complete_turn=lambda turn_id, reply: events.append(f"complete:{turn_id}:{reply}"),
                cancel_turn=lambda turn_id, reply: events.append(f"cancel:{turn_id}:{reply}"),
                fail_turn=lambda turn_id, reply: events.append(f"fail:{turn_id}:{reply}"),
            )
            self.agent = SimpleNamespace(skill_runtime=SimpleNamespace(skills_by_ids=lambda ids: []))
            self._live_preview = SimpleNamespace(close_all=lambda *args, **kwargs: None, reset=lambda: events.append("reset"))

        def _set_partial_renderable(self, _renderable, visible=None):
            events.append(f"partial:{visible}")

        def _write_assistant_bar_line(self, _markup=""):
            return None

        def _write_code_block(self, _lines, _language, _indent=2):
            return None

        def _clear_partial_preview(self):
            events.append("clear-preview")

        def _render_static_markdown(self, reply: str):
            events.append(f"render:{reply}")

        def _update_partial_content(self):
            events.append("update-partial")

        def _update_context_usage_from_payload(self, usage):
            events.append(f"usage:{usage.get('prompt_tokens')}")

        def _write_error(self, text: str):
            events.append(f"error:{text}")

        def _write(self, markup: str):
            events.append(f"write:{markup}")

        def _maybe_scroll_end(self):
            events.append("scroll")

        def _update_status1(self):
            events.append("status1")

        def _update_status2(self):
            events.append("status2")

        def _update_sidebar(self):
            events.append("sidebar")

        def _update_metadata(self):
            events.append("metadata")

        def _save_active_session(self):
            raise RuntimeError("save exploded")

    app = _FinishApp()
    result = AgentTurnResult(status="done", content="done", reasoning="", skill_exchanges=[], journal={})

    finish_turn_stream(app, "t1", result)

    assert app.streaming is False
    assert app._active_turn_id is None
    assert app._esc_pending is False
    assert app._auto_follow_stream is True
    assert "reset" in events
    assert any(item.startswith("error:Stream finalization failed: save exploded") for item in events)


def test_finish_turn_stream_routes_error_content_to_error_channel() -> None:
    events: list[str] = []

    class _FinishApp:
        def __init__(self) -> None:
            self.streaming = True
            self._active_turn_id = "t1"
            self._esc_pending = False
            self._auto_follow_stream = True
            self._reply_acc = ""
            self._buf_r = ""
            self._buf_c = ""
            self._content_open = False
            self._reasoning_open = False
            self._last_stream_error_text = ""
            self._pending_tool_details = []
            self._stream_runtime = StreamRuntimeState()
            self._loaded_skill_ids = []
            self.conv_tree = SimpleNamespace(
                nodes={"t1": object()},
                append_skill_exchange=lambda turn_id, msg: events.append(f"skill:{turn_id}:{msg.get('role')}"),
                complete_turn=lambda turn_id, reply: events.append(f"complete:{turn_id}:{reply}"),
                cancel_turn=lambda turn_id, reply: events.append(f"cancel:{turn_id}:{reply}"),
                fail_turn=lambda turn_id, reply: events.append(f"fail:{turn_id}:{reply}"),
            )
            self.agent = SimpleNamespace(skill_runtime=SimpleNamespace(skills_by_ids=lambda ids: []))
            self._live_preview = SimpleNamespace(close_all=lambda *args, **kwargs: None, reset=lambda: None)

        def _set_partial_renderable(self, _renderable, visible=None):
            return None

        def _write_assistant_bar_line(self, _markup=""):
            return None

        def _write_code_block(self, _lines, _language, _indent=2):
            return None

        def _clear_partial_preview(self):
            return None

        def _render_static_markdown(self, reply: str):
            events.append(f"render:{reply}")

        def _update_partial_content(self):
            return None

        def _update_context_usage_from_payload(self, _usage):
            return None

        def _write_error(self, text: str):
            events.append(f"error:{text}")

        def _write(self, markup: str):
            events.append(f"write:{markup}")

        def _maybe_scroll_end(self):
            return None

        def _update_status1(self):
            return None

        def _update_status2(self):
            return None

        def _update_sidebar(self):
            return None

        def _update_metadata(self):
            return None

        def _save_active_session(self):
            return None

    app = _FinishApp()
    result = AgentTurnResult(
        status="error",
        content="[agent error] Finalization failed: the model repeatedly returned invalid final-answer output.",
        reasoning="",
        skill_exchanges=[],
        error="finalization_failed",
        journal={},
    )

    finish_turn_stream(app, "t1", result)

    assert not any(item.startswith("render:") for item in events)
    assert "error:[agent error] Finalization failed: the model repeatedly returned invalid final-answer output." in events
    assert "fail:t1:[agent error] Finalization failed: the model repeatedly returned invalid final-answer output." in events
