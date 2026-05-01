from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rich.highlighter import Highlighter
from textual import events
from textual.binding import Binding
from textual.widgets import Input

from tui.themes import fallback_color

DEFAULT_ACCENT_COLOR = fallback_color("accent")
DEFAULT_PANEL_BG = fallback_color("panel_bg")


@dataclass
class _CompactPasteChunk:
    start: int
    end: int
    marker: str
    text: str


class _PasteTokenHighlighter(Highlighter):
    STYLE = f"bold {DEFAULT_ACCENT_COLOR} on {DEFAULT_PANEL_BG}"

    def __init__(self, token_ranges_provider: Callable[[], list[tuple[int, int]]] | None = None) -> None:
        super().__init__()
        self._token_ranges_provider = token_ranges_provider or (lambda: [])

    def highlight(self, text) -> None:
        plain = text.plain
        for start, end in self._token_ranges_provider():
            if start < 0 or end <= start or end > len(plain):
                continue
            text.stylize(self.STYLE, start, end)


class ChatInput(Input):
    COMPACT_PASTE_THRESHOLD = 120
    PASTE_TOKEN_STYLE = _PasteTokenHighlighter.STYLE

    BINDINGS = [
        Binding("ctrl+h", "delete_left", show=False),
        Binding("ctrl+u", "clear_all", show=False),
        Binding("ctrl+k", "open_global_palette", show=False),
        Binding("ctrl+shift+k", "kill_to_end", show=False),
        Binding("ctrl+backspace", "remove_last_attachment", show=False),
        Binding("ctrl+shift+backspace", "clear_attachments", show=False),
        Binding("ctrl+f", "open_file_picker", show=False),
        Binding("ctrl+g", "focus_input", show=False),
        Binding("ctrl+p", "open_command_palette", show=False),
        Binding("f1", "show_keymap", show=False),
        Binding("f2", "toggle_details", show=False),
        Binding("f3", "toggle_thinking", show=False),
    ]

    def __init__(self, *args, **kwargs) -> None:
        if kwargs.get("highlighter") is None:
            kwargs["highlighter"] = _PasteTokenHighlighter()
        super().__init__(*args, **kwargs)
        self._compact_paste_chunks: list[_CompactPasteChunk] = []
        self._last_value = self.value
        if isinstance(self.highlighter, _PasteTokenHighlighter):
            self.highlighter._token_ranges_provider = self._highlighted_placeholder_ranges

    def _highlighted_placeholder_ranges(self) -> list[tuple[int, int]]:
        return [(chunk.start, chunk.end) for chunk in self._compact_paste_chunks]

    @staticmethod
    def _paste_marker(text: str) -> str:
        return f"[Pasted {len(text)} chars]"

    @staticmethod
    def _changed_span(old_value: str, new_value: str) -> tuple[int, int, int]:
        prefix_len = 0
        max_prefix = min(len(old_value), len(new_value))
        while prefix_len < max_prefix and old_value[prefix_len] == new_value[prefix_len]:
            prefix_len += 1

        old_suffix = len(old_value)
        new_suffix = len(new_value)
        while old_suffix > prefix_len and new_suffix > prefix_len and old_value[old_suffix - 1] == new_value[new_suffix - 1]:
            old_suffix -= 1
            new_suffix -= 1

        return prefix_len, old_suffix, new_suffix

    def _sync_chunk_ranges(self, value: str) -> None:
        old_value = self._last_value
        if old_value == value:
            self._last_value = value
            return

        prefix_len, old_suffix, new_suffix = self._changed_span(old_value, value)

        delta = (new_suffix - prefix_len) - (old_suffix - prefix_len)
        updated: list[_CompactPasteChunk] = []
        for chunk in self._compact_paste_chunks:
            start = chunk.start
            end = chunk.end

            if end <= prefix_len:
                pass
            elif start >= old_suffix:
                start += delta
                end += delta
            else:
                continue

            if start < 0 or end > len(value):
                continue
            if value[start:end] != chunk.marker:
                continue
            updated.append(_CompactPasteChunk(start=start, end=end, marker=chunk.marker, text=chunk.text))

        self._compact_paste_chunks = updated
        self._last_value = value

    def on_paste(self, event: events.Paste) -> None:
        text = event.text
        if len(text) < self.COMPACT_PASTE_THRESHOLD:
            return
        prevent_default = getattr(event, "prevent_default", None)
        if callable(prevent_default):
            prevent_default()
        self.sync_paste_placeholders(self.value)
        marker = self._paste_marker(text)
        before = self.value
        super().insert_text_at_cursor(marker)
        after = self.value
        start, _old_end, end = self._changed_span(before, after)
        if after[start:end] != marker:
            found = after.find(marker, start)
            if found < 0:
                self._last_value = after
                event.stop()
                return
            start = found
            end = found + len(marker)
        self._compact_paste_chunks.append(_CompactPasteChunk(start=start, end=end, marker=marker, text=text))
        self._last_value = self.value
        event.stop()

    def sync_paste_placeholders(self, value: str | None = None) -> None:
        self._sync_chunk_ranges(self.value if value is None else value)

    def expanded_value(self, value: str | None = None) -> str:
        self.sync_paste_placeholders(self.value if value is None else value)
        expanded = self.value if value is None else value
        for chunk in sorted(self._compact_paste_chunks, key=lambda item: item.start, reverse=True):
            if chunk.start < 0 or chunk.end > len(expanded):
                continue
            if expanded[chunk.start : chunk.end] != chunk.marker:
                continue
            expanded = f"{expanded[: chunk.start]}{chunk.text}{expanded[chunk.end :]}"
        return expanded

    def clear_draft(self) -> None:
        self._compact_paste_chunks.clear()
        self.value = ""
        self._last_value = self.value

    def action_clear_all(self) -> None:
        self.clear_draft()

    def action_kill_to_end(self) -> None:
        self.value = self.value[: self.cursor_position]
        self.sync_paste_placeholders()

    def action_delete_left(self) -> None:
        self.sync_paste_placeholders(self.value)
        if not self.selection.is_empty:
            super().action_delete_left()
            self.sync_paste_placeholders(self.value)
            return

        cursor = self.cursor_position
        if cursor <= 0:
            return

        for chunk in self._compact_paste_chunks:
            if chunk.start < cursor <= chunk.end:
                self.delete(chunk.start, chunk.end)
                self.sync_paste_placeholders(self.value)
                return

        super().action_delete_left()
        self.sync_paste_placeholders(self.value)

    def _invoke_app_action(self, action_name: str) -> None:
        action = getattr(self.app, action_name, None)
        if callable(action):
            action()

    def action_focus_input(self) -> None:
        self._invoke_app_action("action_focus_input")

    def action_open_command_palette(self) -> None:
        self._invoke_app_action("action_open_command_palette")

    def action_open_global_palette(self) -> None:
        self._invoke_app_action("action_open_global_palette")

    def action_open_file_picker(self) -> None:
        self._invoke_app_action("action_open_file_picker")

    def action_remove_last_attachment(self) -> None:
        self._invoke_app_action("action_remove_last_attachment")

    def action_clear_attachments(self) -> None:
        self._invoke_app_action("action_clear_attachments")

    def action_show_keymap(self) -> None:
        self._invoke_app_action("action_show_keymap")

    def action_toggle_details(self) -> None:
        self._invoke_app_action("action_toggle_details")

    def action_toggle_thinking(self) -> None:
        self._invoke_app_action("action_toggle_thinking")
