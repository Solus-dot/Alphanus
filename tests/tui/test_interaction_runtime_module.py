from __future__ import annotations

import threading
from types import SimpleNamespace

from tui.interaction_runtime import action_handle_esc, begin_action_approval, on_input_submitted


class _ChatInput:
    def __init__(self, expanded_text: str) -> None:
        self.value = ""
        self.expanded_text = expanded_text
        self.clear_calls = 0
        self.sync_calls: list[str] = []

    def sync_paste_placeholders(self, value: str) -> None:
        self.sync_calls.append(value)

    def expanded_value(self, _value: str) -> str:
        return self.expanded_text

    def clear_draft(self) -> None:
        self.clear_calls += 1
        self.value = ""


class _SubmitApp:
    def __init__(self, chat_input: _ChatInput) -> None:
        self.chat_input = chat_input
        self.streaming = False
        self.accept_popup = False
        self.should_accept_seen = ""
        self.sent: list[str] = []
        self.hidden_popup_calls = 0
        self.command_inputs: list[str] = []

    def query_one(self, _chat_input_cls):
        return self.chat_input

    def _should_accept_popup_on_enter(self, text: str) -> bool:
        self.should_accept_seen = text
        return self.accept_popup

    def _accept_command_selection(self) -> None:
        raise AssertionError("should not accept command popup")

    def _hide_command_popup(self) -> None:
        self.hidden_popup_calls += 1

    def _handle_command(self, text: str) -> bool:
        self.command_inputs.append(text)
        return False

    def _send(self, text: str) -> None:
        self.sent.append(text)


def test_on_input_submitted_expands_compacted_paste_before_send() -> None:
    expanded = "A" * 130 + " tail"
    chat_input = _ChatInput(expanded_text=expanded)
    app = _SubmitApp(chat_input)

    on_input_submitted(
        app,
        SimpleNamespace(value="[Pasted 130 chars] tail"),
        chat_input_cls=object,
    )

    assert app.should_accept_seen == expanded
    assert app.command_inputs == [expanded]
    assert app.sent == [expanded]
    assert app.hidden_popup_calls == 1
    assert chat_input.clear_calls == 1
    assert chat_input.sync_calls == ["[Pasted 130 chars] tail"]


def test_begin_action_approval_writes_prominent_transcript_prompt() -> None:
    writes: list[str] = []
    lines: list[str] = []
    event = threading.Event()
    holder = {"value": False}
    app = SimpleNamespace(
        _await_action_approval=False,
        _action_approval_command="",
        _action_approval_event=None,
        _action_approval_result=None,
        _write=lambda markup="": writes.append(markup),
        _write_assistant_bar_line=lambda markup, **_kwargs: lines.append(markup),
        _theme_color=lambda _name, fallback=None: fallback or "#ffffff",
        _update_status2=lambda: None,
    )

    begin_action_approval(
        app,
        {"kind": "shell_command", "command": "go version", "reason": "test approval", "cwd": "/tmp/project"},
        event,
        holder,
        esc=lambda value: value,
    )

    assert app._await_action_approval is True
    assert app._action_approval_command == "go version"
    assert app._action_approval_event is event
    assert app._action_approval_result is holder
    assert lines == [
        "[bold #f59e0b]Action Approval Required[/bold #f59e0b]",
        "[bold #f59e0b]![/bold #f59e0b] action waiting for approval",
        "[#a1a1aa]reason:[/#a1a1aa] test approval",
        "[#a1a1aa]cwd:[/#a1a1aa] /tmp/project",
        "[#a1a1aa]command:[/#a1a1aa] go version",
        "[#a1a1aa]press[/#a1a1aa] [bold #22c55e]Y[/bold #22c55e] [#a1a1aa]to run, or[/#a1a1aa] [bold #ef4444]N[/bold #ef4444] [#a1a1aa]to reject[/#a1a1aa]",
    ]
    assert writes == ["", ""]


def test_action_handle_esc_clears_chat_input_via_clear_draft() -> None:
    chat_input = _ChatInput(expanded_text="")
    chat_input.value = "[Pasted 130 chars]"
    app = SimpleNamespace(
        _await_action_approval=False,
        _command_popup_active=lambda: False,
        _hide_command_popup=lambda: None,
        streaming=False,
        query_one=lambda _chat_input_cls: chat_input,
    )

    action_handle_esc(app, chat_input_cls=object)

    assert chat_input.clear_calls == 1
    assert chat_input.value == ""


def test_action_handle_esc_streaming_second_press_sets_stop_event_and_emits_info() -> None:
    chat_input = _ChatInput(expanded_text="")
    infos: list[str] = []
    flushes: list[dict[str, object]] = []
    stop_event = threading.Event()
    app = SimpleNamespace(
        _await_action_approval=False,
        _command_popup_active=lambda: False,
        _hide_command_popup=lambda: None,
        streaming=True,
        _esc_pending=True,
        _esc_ts=0.0,
        _stop_event=stop_event,
        _write_info=infos.append,
        _flush_content_buffer=lambda **kwargs: flushes.append(kwargs),
        _update_status2=lambda: None,
        query_one=lambda _chat_input_cls: chat_input,
    )

    action_handle_esc(app, chat_input_cls=object)

    assert stop_event.is_set()
    assert app._esc_pending is False
    assert flushes == [{"include_partial": True, "update_partial": True}]
    assert infos == ["Interrupt requested. Stopping current turn..."]


def test_action_handle_esc_streaming_action_approval_first_press_only_arms_stop() -> None:
    chat_input = _ChatInput(expanded_text="")
    stop_event = threading.Event()
    finishes: list[bool] = []
    app = SimpleNamespace(
        _await_action_approval=True,
        _finish_action_approval=finishes.append,
        _command_popup_active=lambda: False,
        _hide_command_popup=lambda: None,
        streaming=True,
        _esc_pending=False,
        _esc_ts=0.0,
        _stop_event=stop_event,
        _write_info=lambda _text: None,
        _update_status2=lambda: None,
        query_one=lambda _chat_input_cls: chat_input,
    )

    action_handle_esc(app, chat_input_cls=object)

    assert app._esc_pending is True
    assert not stop_event.is_set()
    assert finishes == []


def test_action_handle_esc_streaming_action_approval_second_press_stops_and_rejects() -> None:
    chat_input = _ChatInput(expanded_text="")
    infos: list[str] = []
    finishes: list[bool] = []
    stop_event = threading.Event()
    app = SimpleNamespace(
        _await_action_approval=True,
        _finish_action_approval=finishes.append,
        _command_popup_active=lambda: False,
        _hide_command_popup=lambda: None,
        streaming=True,
        _esc_pending=True,
        _esc_ts=0.0,
        _stop_event=stop_event,
        _write_info=infos.append,
        _update_status2=lambda: None,
        query_one=lambda _chat_input_cls: chat_input,
    )

    action_handle_esc(app, chat_input_cls=object)

    assert stop_event.is_set()
    assert app._esc_pending is False
    assert finishes == [False]
    assert infos == ["Interrupt requested. Stopping current turn..."]
