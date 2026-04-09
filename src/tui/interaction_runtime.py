from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional


def on_session_manager_close(app: Any, result: Optional[Dict[str, str]]) -> None:
    action = str((result or {}).get("action") or "").strip()
    if not action:
        return
    if action == "open":
        session_id = str((result or {}).get("session_id") or "").strip()
        if not session_id:
            return
        try:
            app._load_session_from_manager(session_id)
        except Exception as exc:
            app._write_error(f"Load failed: {exc}")
        return
    if action == "create":
        session = app._open_new_session(str((result or {}).get("title") or ""))
        app._switch_to_session(session)
        return
    if action == "delete":
        session_id = str((result or {}).get("session_id") or "").strip()
        if not session_id:
            return
        app._delete_session_from_manager(session_id)


def show_keyboard_shortcuts(app: Any, *, sections) -> None:
    col = max((len(command) for _title, rows in sections for command, _desc in rows), default=20) + 4
    for title, rows in sections:
        app._write_section_heading(title)
        for command, desc in rows:
            app._write_command_row(command, desc, col=col)
        app._write("")


def confirm_shell_command(app: Any, command: str) -> bool:
    event = threading.Event()
    holder = {"value": False}
    app.call_from_thread(app._begin_shell_confirm, command, event, holder)
    if not event.wait(timeout=app._timing_config().shell_confirm_timeout_s):
        app.call_from_thread(app._expire_shell_confirm, event)
        return False
    return bool(holder.get("value", False))


def begin_shell_confirm(app: Any, command: str, event: threading.Event, holder: Dict[str, bool], *, esc) -> None:
    app._await_shell_confirm = True
    app._shell_confirm_command = command
    app._shell_confirm_event = event
    app._shell_confirm_result = holder
    app._write(f"[yellow]  ? Run shell command: {esc(command)}[/yellow]")
    app._update_status2()


def expire_shell_confirm(app: Any, event: threading.Event) -> None:
    if not app._await_shell_confirm:
        return
    if app._shell_confirm_event is not event:
        return
    if app._shell_confirm_result is not None:
        app._shell_confirm_result["value"] = False
    app._shell_confirm_event.set()
    app._write("[dim red]  · shell command approval timed out[/dim red]")
    app._await_shell_confirm = False
    app._shell_confirm_command = ""
    app._shell_confirm_event = None
    app._shell_confirm_result = None
    app._update_status2()


def finish_shell_confirm(app: Any, approved: bool) -> None:
    if not app._await_shell_confirm:
        return
    if app._shell_confirm_result is not None:
        app._shell_confirm_result["value"] = approved
    if app._shell_confirm_event is not None:
        app._shell_confirm_event.set()
    msg = "approved" if approved else "rejected"
    color = "green" if approved else "red"
    app._write(f"[dim {color}]  · shell command {msg}[/dim {color}]")
    app._await_shell_confirm = False
    app._shell_confirm_command = ""
    app._shell_confirm_event = None
    app._shell_confirm_result = None
    app._update_status2()


def on_input_submitted(app: Any, event: Any, *, chat_input_cls: Any) -> None:
    text = event.value.strip()
    if not text:
        app.query_one(chat_input_cls).value = ""
        return
    if app._should_accept_popup_on_enter(text):
        app._accept_command_selection()
        return
    app.query_one(chat_input_cls).value = ""
    app._hide_command_popup()
    handled = app._handle_command(text)
    if handled:
        cmd = text.split(None, 1)[0].lower()
        if cmd not in {"/quit", "/exit", "/q", "/config"}:
            app._ensure_command_gap()
        return
    if not app.streaming:
        app._send(text)


def action_handle_esc(app: Any, *, chat_input_cls: Any) -> None:
    if app._await_shell_confirm:
        app._finish_shell_confirm(False)
        return
    if app._command_popup_active():
        app._hide_command_popup()
        return
    if not app.streaming:
        app.query_one(chat_input_cls).value = ""
        return
    now_value = time.monotonic()
    if not app._esc_pending:
        app._esc_pending = True
        app._esc_ts = now_value
        app._update_status2()
    else:
        app._stop_event.set()
        app._esc_pending = False
        app._update_status2()
