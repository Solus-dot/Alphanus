from __future__ import annotations

import io
from dataclasses import dataclass

from rich.console import Console, Group, RenderableType
from rich.text import Text
from textual.widgets import Static


@dataclass(slots=True)
class TranscriptEntry:
    kind: str
    renderable: RenderableType


@dataclass(slots=True)
class ScrollAnchor:
    near_bottom: bool
    entry_index: int
    line_offset: int
    partial_line_offset: int = -1


def count_renderable_lines(renderable: RenderableType, width: int) -> int:
    width = max(1, int(width))
    console = _line_count_console(width)
    options = console.options.update(width=width)
    return _entry_line_count(TranscriptEntry("renderable", renderable), console, options)


class TranscriptView(Static):
    def __init__(self, *args, max_lines: int | None = None, **kwargs) -> None:
        super().__init__("", *args, markup=False, **kwargs)
        self._entries: list[TranscriptEntry] = []
        self._max_lines = max_lines
        self._last_render_width: int = 1
        self._last_line_counts: list[int] = []
        self._last_line_total: int = 0
        self._has_rendered = False

    @property
    def max_lines(self) -> int | None:
        return self._max_lines

    @max_lines.setter
    def max_lines(self, value: int | None) -> None:
        self._max_lines = value

    def render(self) -> RenderableType:
        self._last_render_width = self._available_width()
        self._has_rendered = True
        if not self._entries:
            return Text("")
        return Group(*(entry.renderable for entry in self._entries))

    def set_entries(self, entries: list[TranscriptEntry]) -> None:
        self._entries = list(entries)
        self._recalculate_line_cache(self._measurement_width())
        self._trim_entries_to_max_lines()
        self.refresh(layout=True)

    def append_entry(self, entry: TranscriptEntry, *, refresh: bool = True) -> None:
        width = self._measurement_width()
        self._entries.append(entry)
        count = self._entry_line_count_for_width(entry, width)
        self._last_line_counts.append(count)
        self._last_line_total += count
        self._trim_entries_to_max_lines()
        if refresh:
            self.refresh(layout=True)

    def clear_entries(self) -> None:
        self._entries = []
        self._last_line_counts = []
        self._last_line_total = 0
        self.refresh(layout=True)

    def refresh_for_width_change(self) -> None:
        width = self._available_width()
        self._last_render_width = width
        self._recalculate_line_cache(width)
        self.refresh(layout=True)

    def capture_anchor(self, scroll_y: float, *, partial_line_count: int = 0) -> ScrollAnchor | None:
        counts = self._cached_line_counts()
        if not counts and partial_line_count <= 0:
            return None
        line_y = max(0, int(scroll_y))
        transcript_total = sum(counts)
        total = 0
        index = len(counts) - 1
        offset = 0
        partial_offset = -1
        if partial_line_count > 0 and line_y >= transcript_total:
            partial_offset = min(line_y - transcript_total, max(0, partial_line_count - 1))
            index = max(0, len(counts) - 1)
            offset = 0
        else:
            if not counts:
                return None
            for current, count in enumerate(counts):
                if total + count > line_y:
                    index = current
                    offset = line_y - total
                    break
                total += count
        parent = self.parent
        max_scroll_y = float(getattr(parent, "max_scroll_y", 0) or 0)
        current_scroll_y = float(getattr(parent, "scroll_y", scroll_y) or 0)
        near_bottom = (max_scroll_y - current_scroll_y) <= 1.0
        return ScrollAnchor(near_bottom=near_bottom, entry_index=index, line_offset=offset, partial_line_offset=partial_offset)

    def restore_anchor(self, anchor: ScrollAnchor | None, *, partial_line_count: int = 0) -> float:
        if not anchor:
            return 0.0
        counts = self._entry_line_counts_for_width(self._available_width())
        transcript_total = sum(counts)
        if anchor.partial_line_offset >= 0 and partial_line_count > 0:
            return float(transcript_total + min(anchor.partial_line_offset, max(0, partial_line_count - 1)))
        if not counts:
            return 0.0
        index = max(0, min(anchor.entry_index, len(counts) - 1))
        offset = max(0, min(anchor.line_offset, counts[index] - 1))
        return float(sum(counts[:index]) + offset)

    def _available_width(self) -> int:
        width = int(getattr(self.content_region, "width", 0) or 0)
        if width <= 0:
            width = int(getattr(self.region, "width", 0) or 0)
        if width <= 0:
            width = int(getattr(self.size, "width", 0) or 0)
        return max(1, width)

    def _cached_line_counts(self) -> list[int]:
        if self._has_rendered and len(self._last_line_counts) == len(self._entries):
            return list(self._last_line_counts)
        self._recalculate_line_cache(self._available_width())
        return list(self._last_line_counts)

    def _measurement_width(self) -> int:
        return self._last_render_width if self._has_rendered else self._available_width()

    def _entry_line_counts_for_width(self, width: int) -> list[int]:
        width = max(1, int(width))
        console = _line_count_console(width)
        options = console.options.update(width=width)
        return [_entry_line_count(entry, console, options) for entry in self._entries]

    def _entry_line_count_for_width(self, entry: TranscriptEntry, width: int) -> int:
        width = max(1, int(width))
        console = _line_count_console(width)
        options = console.options.update(width=width)
        return _entry_line_count(entry, console, options)

    def _recalculate_line_cache(self, width: int) -> None:
        self._last_line_counts = self._entry_line_counts_for_width(width)
        self._last_line_total = sum(self._last_line_counts)

    def _trim_entries_to_max_lines(self) -> None:
        if not isinstance(self._max_lines, int) or self._max_lines <= 0 or not self._entries:
            return
        while len(self._entries) > 1 and self._last_line_total > self._max_lines:
            removed = self._last_line_counts.pop(0)
            self._entries.pop(0)
            self._last_line_total -= removed
        if not self._entries:
            self._last_line_counts = []
            self._last_line_total = 0
        elif not self._last_line_counts or len(self._last_line_counts) != len(self._entries):
            self._recalculate_line_cache(self._measurement_width())


def _line_count_console(width: int) -> Console:
    console = Console(
        width=width,
        file=io.StringIO(),
        force_terminal=True,
        color_system="truecolor",
        record=False,
    )
    return console


def _entry_line_count(entry: TranscriptEntry, console: Console, options) -> int:
    if entry.kind == "blank":
        return 1
    lines = console.render_lines(entry.renderable, options, pad=False, new_lines=False)
    return max(1, len(lines))
