from __future__ import annotations

import queue
from types import SimpleNamespace

from core.types import AgentTurnResult
from tui.stream_runtime import StreamRuntimeState, drain_events, enqueue_event, finish_turn_stream, start_turn_stream, visible_reasoning_text


class _Turn:
    def __init__(self, turn_id: str) -> None:
        self.id = turn_id


class _App:
    def __init__(self) -> None:
        self.streaming = False
        self._auto_follow_stream = False
        self._active_turn_id = None
        self._reply_acc = "stale"
        self._pending_tool_details = ["stale"]
        self._last_stream_error_text = "stale"
        self._stream_runtime = StreamRuntimeState()
        self._stop_event = None
        self._loaded_skill_ids = ["a", "b"]
        self.thinking = True
        self.conv_tree = SimpleNamespace(active_path=[], history_messages=lambda: [])
        self._stream_worker_calls = []
        self._call_from_thread_calls = []
        self._usage_updates = 0
        self._last_scroll = 0.0
        self._scroll_interval = 999.0
        self._live_preview = SimpleNamespace(reset=lambda: None)

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
    assert app._stream_worker_calls
    call = app._stream_worker_calls[0]
    assert call[0] == "t1"
    assert call[2] == "hi"
    assert call[4] == ["a.txt"]


def test_enqueue_event_creates_queue_and_schedules_drain_for_notable_events() -> None:
    app = _App()
    app._stream_runtime = None

    enqueue_event(app, {"type": "tool_call", "name": "x"})

    assert isinstance(app._stream_runtime, StreamRuntimeState)
    queued = app._stream_runtime.event_queue.get_nowait()
    assert queued["type"] == "tool_call"
    assert app._call_from_thread_calls


def test_drain_events_processes_usage_event() -> None:
    app = _App()
    app._stream_runtime.event_queue = queue.SimpleQueue()
    app._stream_runtime.event_queue.put({"type": "usage", "usage": {"prompt_tokens": 5}})

    drain_events(app)

    assert app._usage_updates == 1
    assert app._stream_runtime.drain_active is False


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

        def _update_topbar(self):
            events.append("topbar")

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
