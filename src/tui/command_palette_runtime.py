from __future__ import annotations

from typing import Any

from rich.markup import escape as esc
from textual.widgets import Static
from textual.widgets.option_list import Option

from tui.commands import (
    active_command_query,
    active_command_span,
    command_entries_for_query,
    command_label,
    exact_command_inputs,
    popup_command_query,
)


def refresh_command_popup(app: Any, value: str, *, chat_input_cls: Any) -> None:
    popup = app._command_popup()
    options = app._command_options()
    chat_input = app.query_one(chat_input_cls)
    query = popup_command_query(value, chat_input.cursor_position)
    next_matches = command_entries_for_query(query)
    if not next_matches or app.streaming or app._await_shell_confirm:
        next_matches = []

    if not next_matches:
        app._command_matches = []
        popup.display = False
        options.clear_options()
        options.styles.height = 0
        popup.styles.height = 0
        popup.offset = (0, 0)
        return

    app._command_matches = next_matches
    option_rows = min(len(app._command_matches), 8)
    option_height = option_rows + 1
    popup_height = option_height + 5
    separator = app.query_one("#footer-sep", Static)
    input_region = getattr(chat_input, "region", None)
    input_size = getattr(chat_input, "size", None)
    input_width = int(getattr(input_region, "width", 0) or 0)
    if input_width <= 0:
        input_width = int(getattr(input_size, "width", 0) or 0)
    input_x = int(getattr(input_region, "x", 0) or 0)
    if input_x <= 0:
        input_x = 1
    sep_region = getattr(separator, "region", None)
    separator_y = int(getattr(sep_region, "y", 0) or 0)
    if separator_y <= 0:
        input_y = int(getattr(input_region, "y", 0) or 0)
        input_height = int(getattr(input_region, "height", 0) or 0)
        separator_y = max(1, input_y + input_height)

    popup.display = True
    popup.styles.height = popup_height
    popup.styles.width = max(44, min(72, max(input_width, 44)))
    popup.offset = (max(1, input_x), max(1, separator_y - popup_height))
    options.styles.height = option_height
    rendered = [
        Option(
            f"[bold #6366f1]{esc(command_label(entry))}[/bold #6366f1] [dim]{esc(entry.description)}[/dim]",
            id=str(index),
        )
        for index, entry in enumerate(app._command_matches)
    ]
    options.clear_options()
    options.add_options(rendered)
    options.highlighted = 0


def hide_command_popup(app: Any) -> None:
    app._command_matches = []
    app._command_options().clear_options()
    app._command_popup().display = False
    app._command_popup().styles.height = 0
    app._command_popup().offset = (0, 0)


def command_popup_active(app: Any) -> bool:
    return bool(app._command_matches) and bool(app._command_popup().display)


def move_command_selection(app: Any, delta: int) -> None:
    if not app._command_popup_active():
        return
    options = app._command_options()
    current = 0 if options.highlighted is None else int(options.highlighted)
    count = len(app._command_matches)
    if count <= 0:
        return
    options.highlighted = (current + delta) % count
    options.scroll_to_highlight(top=False)


def accept_command_selection(app: Any, *, chat_input_cls: Any) -> bool:
    if not app._command_popup_active():
        return False
    options = app._command_options()
    highlighted = 0 if options.highlighted is None else int(options.highlighted)
    if highlighted < 0 or highlighted >= len(app._command_matches):
        return False
    return select_command_option(app, highlighted, chat_input_cls=chat_input_cls)


def should_accept_popup_on_enter(app: Any, *, chat_input_cls: Any) -> bool:
    if not app._command_popup_active():
        return False
    chat_input = app.query_one(chat_input_cls)
    query = active_command_query(chat_input.value, chat_input.cursor_position).strip()
    if not query or " " in query:
        return False
    base = query.lower()
    return base not in exact_command_inputs()


def select_command_option(app: Any, index: int, *, chat_input_cls: Any) -> bool:
    if index < 0 or index >= len(app._command_matches):
        return False
    entry = app._command_matches[index]
    chat_input = app.query_one(chat_input_cls)
    span = active_command_span(chat_input.value, chat_input.cursor_position)
    if span is None:
        chat_input.value = entry.insert_text
        chat_input.cursor_position = len(chat_input.value)
    else:
        start, end = span
        chat_input.value = f"{chat_input.value[:start]}{entry.insert_text}{chat_input.value[end:]}"
        chat_input.cursor_position = start + len(entry.insert_text)
    app._refresh_command_popup(chat_input.value)
    return True
