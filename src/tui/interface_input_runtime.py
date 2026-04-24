from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from core.configuration import normalize_config, validate_endpoint_policy
from core.runtime_config import UiRuntimeConfig

KEYBOARD_SHORTCUT_SECTIONS = [
    (
        "Keymap",
        [
            ("F1 / ?", "Show keyboard shortcuts"),
            ("Ctrl+K", "Open quick palette"),
            ("Ctrl+P / /", "Open slash command palette"),
            ("Ctrl+F", "Open file picker"),
            ("Ctrl+G", "Focus composer"),
            ("Tab / Shift+Tab", "Cycle active panels"),
            ("Ctrl+H / Ctrl+L", "Focus transcript or tree"),
            ("F2", "Toggle live tool details"),
            ("F3", "Toggle thinking mode"),
            ("Ctrl+C / Ctrl+D", "Quit app"),
        ],
    ),
    ("Transcript", [("PgUp / PgDn", "Scroll transcript")]),
    (
        "Tree",
        [
            ("j / k", "Move selection"),
            ("Enter / o", "Open selected node"),
            ("[ / ]", "Jump sibling branches"),
            ("g / G", "Jump top or bottom"),
        ],
    ),
    (
        "Input",
        [
            ("Enter", "Send message"),
            ("Esc", "Clear input or stop stream"),
            ("Backspace (empty)", "Remove last attachment"),
            ("Ctrl+Backspace", "Remove last attachment"),
            ("Ctrl+Shift+Backspace", "Clear attachments"),
            ("Ctrl+U", "Clear the full draft"),
            ("Ctrl+Shift+K", "Delete to end of line"),
            ("/detach [n|last|all]", "Remove pending attachments"),
        ],
    ),
    (
        "Slash Palette",
        [
            ("Up / Down", "Move command selection"),
            ("Tab", "Insert highlighted command"),
            ("Esc", "Close command palette"),
        ],
    ),
]


def show_keyboard_shortcuts(app: Any, *, show_fn) -> None:
    show_fn(app, sections=KEYBOARD_SHORTCUT_SECTIONS)


def handle_key(app: Any, event: Any, *, chat_input_cls: Any) -> None:
    chat_input = app.query_one(chat_input_cls)
    if chat_input.has_focus and app._command_popup_active():
        key = event.key.lower()
        if key == "down":
            app._move_command_selection(1)
            event.stop()
            return
        if key == "up":
            app._move_command_selection(-1)
            event.stop()
            return
        if key == "tab":
            app._accept_command_selection()
            event.stop()
            return
    if (
        chat_input.has_focus
        and not app.streaming
        and not app._await_shell_confirm
        and not app._command_popup_active()
        and event.key.lower() == "backspace"
        and not chat_input.value
        and app.pending
    ):
        app.action_remove_last_attachment()
        event.stop()
        return
    if not app._await_shell_confirm:
        return
    key = event.key.lower()
    if key == "y":
        app._finish_shell_confirm(True)
        event.stop()
    elif key in {"n", "escape"}:
        app._finish_shell_confirm(False)
        event.stop()


def on_config_editor_close(app: Any, result: Optional[Dict[str, Any]], *, config_path: Path, status_runtime_state_cls: Any) -> None:
    if not result:
        return
    parsed = result.get("config")
    if not isinstance(parsed, dict):
        app._write_error("Config save failed: invalid config payload")
        return

    try:
        normalized, warnings = normalize_config(parsed)
        validate_endpoint_policy(normalized)
    except ValueError as exc:
        app._write_error(f"Config save failed: {exc}")
        return

    cleaned = app._config_for_editor(normalized)
    config_path.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
    merged = app._merge_live_config(app.agent.config, normalized)
    app.agent.reload_config(merged)
    app.thinking = bool(merged.get("agent", {}).get("enable_thinking", app.thinking))
    app._ui_config = UiRuntimeConfig.from_config(merged)
    app._ui_timing = app._ui_config.timing
    refreshed_status = app.agent.get_model_status()
    app._status_runtime = status_runtime_state_cls(
        model_status=refreshed_status,
        model_name=refreshed_status.model_name if refreshed_status.state != "offline" else None,
        model_context_window=(
            refreshed_status.context_window
            if isinstance(refreshed_status.context_window, int) and refreshed_status.context_window > 0
            else None
        ),
    )
    app._update_topbar()
    app._apply_tui_config()
    app._maybe_refresh_model_status(force=True)
    suffix = f" ({len(warnings)} normalization warning{'s' if len(warnings) != 1 else ''})." if warnings else "."
    app._write_info("Saved global config. Use environment variables for secrets like TAVILY_API_KEY or BRAVE_SEARCH_API_KEY" + suffix)
    for warning in warnings:
        app._write_info(f"Config warning: {warning}")
