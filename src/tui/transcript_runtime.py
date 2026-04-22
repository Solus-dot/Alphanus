from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Tuple

from rich import box
from rich.console import Console, ConsoleOptions, RenderResult, RenderableType
from rich.markup import escape as esc
from rich.padding import Padding
from rich.panel import Panel
from rich.segment import Segment
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text

from tui.markdown_utils import fence_language, hanging_indent, render_md
from tui.themes import fallback_color
from tui.transcript import ScrollAnchor, TranscriptEntry, count_renderable_lines

DEFAULT_USER_BAR_COLOR = fallback_color("user_bar")
DEFAULT_ASSISTANT_BAR_COLOR = fallback_color("assistant_bar")
DEFAULT_PANEL_BG = fallback_color("panel_bg")
DEFAULT_PANEL_BORDER = fallback_color("panel_border")
DEFAULT_TEXT_COLOR = fallback_color("text")
DEFAULT_MUTED_COLOR = fallback_color("muted")
DEFAULT_SUCCESS_COLOR = fallback_color("success")
DEFAULT_ERROR_COLOR = fallback_color("error")
DEFAULT_CHIP_BG = fallback_color("chip_bg")
DEFAULT_CHIP_TEXT = fallback_color("chip_text")


def _theme_color(app: Any, key: str, default: str) -> str:
    picker = getattr(app, "_theme_color", None)
    if callable(picker):
        try:
            return str(picker(key, default))
        except Exception:
            return default
    return default


class EdgeBar:
    def __init__(
        self,
        renderable: RenderableType,
        color: str,
        first_indent: int = 0,
        continuation_indent: Optional[int] = None,
    ) -> None:
        self.renderable = renderable
        self.color = color
        self.first_indent = max(0, int(first_indent))
        self.continuation_indent = (
            self.first_indent if continuation_indent is None else max(0, int(continuation_indent))
        )

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        inner_width = max(1, options.max_width - 2)
        wrap_indent = max(self.first_indent, self.continuation_indent)
        style = Style.parse(f"bold {self.color}")
        if isinstance(self.renderable, Text):
            wrap_width = max(1, inner_width - wrap_indent)
            lines = [text.render(console) for text in self.renderable.wrap(console, wrap_width)] or [[]]
        else:
            inner_options = options.update(width=max(1, inner_width - wrap_indent))
            lines = console.render_lines(self.renderable, inner_options, pad=False, new_lines=False)
        for index, line in enumerate(lines or [[]]):
            indent = self.first_indent if index == 0 else self.continuation_indent
            yield Segment("┃", style)
            yield Segment(" ")
            if indent:
                yield Segment(" " * indent)
            yield from line
            yield Segment.line()


def partial_measurement_width(app: Any) -> int:
    partial = app._partial()
    for region_name in ("content_region", "region", "size"):
        region = getattr(partial, region_name, None)
        width = int(getattr(region, "width", 0) or 0)
        if width > 0:
            return width
    return 1


def set_partial_renderable(
    app: Any,
    renderable: Optional[RenderableType],
    *,
    visible: Optional[bool] = None,
) -> None:
    partial = app._partial()
    if visible is not None:
        partial.display = visible
    if renderable is None:
        partial.update("")
        app._partial_renderable = None
        app._last_partial_render_width = partial_measurement_width(app)
        app._last_partial_line_count = 0
        app._partial_line_count_dirty = False
        return
    partial.update(renderable)
    app._partial_renderable = renderable
    app._last_partial_render_width = partial_measurement_width(app)
    app._partial_line_count_dirty = True


def remeasure_partial_line_count(
    app: Any,
    width: Optional[int] = None,
    *,
    count_lines: Callable[[RenderableType, int], int] = count_renderable_lines,
) -> int:
    partial = app._partial()
    renderable = getattr(app, "_partial_renderable", None)
    if not getattr(partial, "display", True) or renderable is None:
        app._last_partial_line_count = 0
        app._partial_line_count_dirty = False
        return 0
    measured_width = max(1, int(width if width is not None else partial_measurement_width(app)))
    app._last_partial_render_width = measured_width
    app._last_partial_line_count = count_lines(renderable, measured_width)
    app._partial_line_count_dirty = False
    return app._last_partial_line_count


def cached_partial_line_count(
    app: Any,
    *,
    count_lines: Callable[[RenderableType, int], int] = count_renderable_lines,
) -> int:
    partial = app._partial()
    if not getattr(partial, "display", True):
        return 0
    if getattr(app, "_partial_renderable", None) is None:
        return 0
    width = partial_measurement_width(app)
    if getattr(app, "_partial_line_count_dirty", False) or width != int(getattr(app, "_last_partial_render_width", 0) or 0):
        return remeasure_partial_line_count(app, width, count_lines=count_lines)
    return int(getattr(app, "_last_partial_line_count", 0) or 0)


def current_partial_line_count(
    app: Any,
    *,
    count_lines: Callable[[RenderableType, int], int] = count_renderable_lines,
) -> int:
    return remeasure_partial_line_count(app, partial_measurement_width(app), count_lines=count_lines)


def append_transcript_entry(app: Any, entry: TranscriptEntry) -> None:
    app._log().append_entry(entry)
    app._maybe_scroll_end()


def write_markup(app: Any, markup: str) -> None:
    entry = TranscriptEntry("blank", Text("")) if markup == "" else TranscriptEntry("markup_line", Text.from_markup(markup))
    append_transcript_entry(app, entry)
    app._last_log_was_blank = markup == ""


def bar_renderable(
    renderable: RenderableType,
    color: str,
    *,
    content_indent: int = 0,
    continuation_indent: Optional[int] = None,
) -> EdgeBar:
    return EdgeBar(
        renderable,
        color,
        first_indent=content_indent,
        continuation_indent=continuation_indent,
    )


def line_indents(line: str, *, base_indent: int = 2) -> Tuple[int, int]:
    lead = len(line) - len(line.lstrip(" "))
    return max(base_indent, lead), max(base_indent, hanging_indent(line))


def write_renderable(app: Any, renderable: RenderableType, indent: int = 2) -> None:
    append_transcript_entry(app, TranscriptEntry("renderable", Padding(renderable, pad=(0, 0, 0, indent))))
    app._last_log_was_blank = False


def write_user_bar_line(app: Any, markup: str = "", *, content_indent: int = 0) -> None:
    app._write_renderable(
        app._bar_renderable(
            Text.from_markup(markup),
            _theme_color(app, "user_bar", DEFAULT_USER_BAR_COLOR),
            content_indent=content_indent,
        ),
        indent=0,
    )


def write_assistant_bar_line(app: Any, markup: str = "", *, content_indent: int = 0) -> None:
    app._write_renderable(
        app._bar_renderable(
            Text.from_markup(markup),
            _theme_color(app, "assistant_bar", DEFAULT_ASSISTANT_BAR_COLOR),
            content_indent=content_indent,
        ),
        indent=0,
    )


def write_assistant_bar_renderable(app: Any, renderable: RenderableType, *, content_indent: int = 0) -> None:
    app._write_renderable(
        app._bar_renderable(
            renderable,
            _theme_color(app, "assistant_bar", DEFAULT_ASSISTANT_BAR_COLOR),
            content_indent=content_indent,
        ),
        indent=0,
    )


def write_user_bar_wrapped_line(app: Any, line: str) -> None:
    first_indent, continuation_indent = app._line_indents(line)
    app._write_renderable(
        app._bar_renderable(
            Text.from_markup(esc(line.lstrip(" "))),
            _theme_color(app, "user_bar", DEFAULT_USER_BAR_COLOR),
            content_indent=first_indent,
            continuation_indent=continuation_indent,
        ),
        indent=0,
    )


def write_assistant_bar_wrapped_line(app: Any, line: str, markup: str) -> None:
    first_indent, continuation_indent = app._line_indents(line)
    app._write_renderable(
        app._bar_renderable(
            Text.from_markup(markup.lstrip(" ")),
            _theme_color(app, "assistant_bar", DEFAULT_ASSISTANT_BAR_COLOR),
            content_indent=first_indent,
            continuation_indent=continuation_indent,
        ),
        indent=0,
    )


def syntax_renderable(
    app: Any,
    code: str,
    language: Optional[str],
    *,
    syntax_theme: str = "github-dark",
    background_color: str = DEFAULT_PANEL_BG,
) -> Syntax:
    return Syntax(
        code,
        language or "text",
        theme=syntax_theme,
        word_wrap=True,
        background_color=background_color,
        line_numbers=False,
    )


def code_panel_renderable(app: Any, code: str, language: Optional[str]) -> Panel:
    border = _theme_color(app, "panel_border", DEFAULT_PANEL_BORDER)
    panel_bg = _theme_color(app, "panel_bg", DEFAULT_PANEL_BG)
    return Panel(
        app._syntax_renderable(code, language),
        expand=True,
        padding=(0, 1),
        border_style=border,
        style=f"on {panel_bg}",
    )


def reasoning_panel_renderable(app: Any, text: str) -> Panel:
    accent = _theme_color(app, "accent", DEFAULT_ASSISTANT_BAR_COLOR)
    border = _theme_color(app, "panel_border", DEFAULT_PANEL_BORDER)
    panel_bg = _theme_color(app, "panel_bg", DEFAULT_PANEL_BG)
    rendered, _ = render_md(text, False)
    return Panel(
        Text.from_markup(f"[dim]{rendered}[/dim]"),
        title=f"[dim {accent}]thinking[/dim {accent}]",
        title_align="left",
        expand=True,
        padding=(0, 1),
        border_style=border,
        style=f"on {panel_bg}",
        box=box.SQUARE,
    )


def tool_event_panel(
    app: Any,
    title: str,
    title_color: str,
    border_color: str,
    name: str,
    detail: str = "",
) -> Panel:
    text_color = _theme_color(app, "text", DEFAULT_TEXT_COLOR)
    muted = _theme_color(app, "muted", DEFAULT_MUTED_COLOR)
    panel_bg = _theme_color(app, "panel_bg", DEFAULT_PANEL_BG)
    text = Text()
    text.append(name, style=f"bold {text_color}")
    if detail:
        text.append("   ")
        text.append(detail, style=muted)
    return Panel(
        text,
        title=f"[bold {title_color}]{title}[/bold {title_color}]",
        title_align="left",
        expand=True,
        padding=(0, 1),
        border_style=border_color,
        style=f"on {panel_bg}",
        box=box.SQUARE,
    )


def tool_lifecycle_panel(app: Any, name: str, detail: str, *, ok: bool) -> Panel:
    success = _theme_color(app, "success", DEFAULT_SUCCESS_COLOR)
    error = _theme_color(app, "error", DEFAULT_ERROR_COLOR)
    return app._tool_event_panel(
        "tool → done" if ok else "tool → fail",
        success if ok else error,
        success if ok else error,
        name,
        detail,
    )


def update_tool_call_partial(app: Any, name: str, detail: str = "") -> None:
    assistant = _theme_color(app, "assistant_bar", DEFAULT_ASSISTANT_BAR_COLOR)
    app._set_partial_renderable(
        app._bar_renderable(app._tool_event_panel("tool", assistant, assistant, name, detail), assistant),
        visible=True,
    )


def write_tool_lifecycle_block(app: Any, name: str, ok: bool, detail: str = "") -> None:
    app._write_assistant_bar_renderable(
        app._tool_lifecycle_panel(name, detail or ("completed" if ok else "failed"), ok=ok),
    )


def show_tool_result_line(app: Any, _name: str, ok: bool) -> bool:
    if not ok:
        return True
    return app._show_tool_details


def take_pending_tool_detail(app: Any, name: str) -> str:
    for idx, (pending_name, pending_detail) in enumerate(app._pending_tool_details):
        if pending_name == name:
            app._pending_tool_details.pop(idx)
            return pending_detail
    return ""


def remember_code_block(app: Any, code: str, language: Optional[str]) -> int:
    app._code_blocks.append((code, language))
    if len(app._code_blocks) > 64:
        app._code_blocks = app._code_blocks[-64:]
    return len(app._code_blocks)


def write_code_block(
    app: Any,
    lines: List[str],
    language: Optional[str],
    content_indent: int = 2,
) -> None:
    code = "\n".join(lines)
    block_index = app._remember_code_block(code, language)
    app._write_assistant_bar_renderable(
        app._code_panel_renderable(code, language),
        content_indent=max(0, int(content_indent)),
    )
    app._write_assistant_bar_line(
        f"[dim]code block {block_index} · /code {block_index} to open copyable view[/dim]",
        content_indent=max(0, int(content_indent)),
    )


def render_static_markdown(app: Any, text: str) -> None:
    in_fence = False
    fence_lang: Optional[str] = None
    fence_lines: List[str] = []
    for line in text.splitlines() or [""]:
        if app._is_fence_line(line):
            if in_fence:
                if fence_lines:
                    app._write_code_block(fence_lines, fence_lang)
                in_fence = False
                fence_lang = None
                fence_lines = []
            else:
                in_fence = True
                fence_lang = fence_language(line)
                fence_lines = []
            continue
        if in_fence:
            fence_lines.append(line)
            continue
        rendered, _ = render_md(line, False)
        app._write_assistant_bar_wrapped_line(line, rendered)

    if in_fence and fence_lines:
        app._write_code_block(fence_lines, fence_lang)


def reset_fence_state(app: Any) -> None:
    app._in_fence = False
    app._fence_lang = None
    app._fence_lines = []


def is_fence_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("```") or stripped.startswith("~~~")


def flush_fence_block(app: Any) -> None:
    if app._fence_lines:
        app._write_code_block(app._fence_lines, app._fence_lang)
    app._reset_fence_state()


def render_content_line(app: Any, line: str) -> None:
    if app._is_fence_line(line):
        if app._in_fence:
            app._flush_fence_block()
        else:
            app._in_fence = True
            app._fence_lang = fence_language(line)
            app._fence_lines = []
        return

    if app._in_fence:
        app._fence_lines.append(line)
        return

    rendered, _ = render_md(line, False)
    app._write_assistant_bar_wrapped_line(line, rendered)


def update_partial_content(app: Any) -> None:
    assistant = _theme_color(app, "assistant_bar", DEFAULT_ASSISTANT_BAR_COLOR)
    if app._in_fence:
        lines = list(app._fence_lines)
        if app._buf_c and not app._is_fence_line(app._buf_c):
            lines.append(app._buf_c)
        if lines:
            app._set_partial_renderable(
                app._bar_renderable(app._code_panel_renderable("\n".join(lines), app._fence_lang), assistant)
            )
        else:
            app._set_partial_renderable(None)
        return
    if not app._buf_c:
        app._set_partial_renderable(None)
        return
    rendered, _ = render_md(app._buf_c, False)
    first_indent, continuation_indent = app._line_indents(app._buf_c)
    app._set_partial_renderable(
        app._bar_renderable(
            Text.from_markup(rendered.lstrip(" ")),
            assistant,
            content_indent=first_indent,
            continuation_indent=continuation_indent,
        )
    )


def update_live_preview_partial(app: Any, lines: List[str], language: Optional[str]) -> None:
    assistant = _theme_color(app, "assistant_bar", DEFAULT_ASSISTANT_BAR_COLOR)
    app._set_partial_renderable(
        app._bar_renderable(app._code_panel_renderable("\n".join(lines), language), assistant),
        visible=True,
    )


def defer_live_preview_partial(app: Any, lines: List[str], language: Optional[str]) -> None:
    state = app._stream_runtime
    state.deferred_live_preview = (list(lines), language)
    state.partial_dirty = False


def clear_partial_preview(app: Any) -> None:
    state = app._stream_runtime
    state.deferred_live_preview = None
    state.partial_dirty = False
    app._set_partial_renderable(None)
    partial = app._partial()
    if not app.streaming:
        partial.display = False


def is_near_bottom(app: Any, threshold: float = 1.0) -> bool:
    scroll = app._scroll()
    return (scroll.max_scroll_y - scroll.scroll_y) <= threshold


def capture_scroll_anchor(app: Any) -> Optional[ScrollAnchor]:
    scroll = app._scroll()
    return app._log().capture_anchor(
        float(scroll.scroll_y or 0),
        partial_line_count=app._cached_partial_line_count(),
    )


def restore_scroll_anchor(app: Any, anchor: Optional[ScrollAnchor]) -> None:
    if not anchor:
        return
    scroll = app._scroll()
    if anchor.near_bottom:
        scroll.scroll_end(animate=False)
        return
    target_y = app._log().restore_anchor(anchor, partial_line_count=app._current_partial_line_count())
    scroll.scroll_to(y=target_y, animate=False, immediate=True, force=True)


def maybe_scroll_end(app: Any, force: bool = False) -> None:
    if force:
        app._scroll().scroll_end(animate=False)
        return
    if app.streaming and app._auto_follow_stream and not app._is_near_bottom():
        app._auto_follow_stream = False
        app._update_status2()
    if not app.streaming or app._auto_follow_stream:
        app._scroll().scroll_end(animate=False)


def write_info(app: Any, text: str, *, accent_color: str) -> None:
    text_color = _theme_color(app, "text", DEFAULT_TEXT_COLOR)
    app._write(f"  [bold {accent_color}]›[/bold {accent_color}] [{text_color}]{esc(text)}[/{text_color}]")


def write_error(app: Any, text: str) -> None:
    app._write(f"[bold red]  ✖ {esc(text)}[/bold red]")


def write_section_heading(app: Any, title: str, *, color: str) -> None:
    app._write("")
    app._write(f"[bold {color}]  {esc(title)}[/bold {color}]")


def write_detail_line(app: Any, label: str, value: str, *, accent_color: str, value_markup: bool = False) -> None:
    text_color = _theme_color(app, "text", DEFAULT_TEXT_COLOR)
    rendered = value if value_markup else esc(value)
    app._write(
        f"  [bold {accent_color}]{esc(label)}:[/bold {accent_color}] [{text_color}]{rendered}[/{text_color}]"
        if not value_markup
        else f"  [bold {accent_color}]{esc(label)}:[/bold {accent_color}] {rendered}"
    )


def write_indexed_dim_lines(
    app: Any,
    rows: List[str],
    *,
    color: str,
    allow_markup: bool = False,
) -> None:
    text_color = _theme_color(app, "text", DEFAULT_TEXT_COLOR)
    for index, row in enumerate(rows):
        if allow_markup:
            app._write(f"  [{color}]{index}.[/{color}] {row}")
        else:
            app._write(f"  [{color}]{index}.[/{color}] [{text_color}]{esc(row)}[/{text_color}]")


def write_command_action(app: Any, text: str, *, color: str, icon: str = "•") -> None:
    text_color = _theme_color(app, "text", DEFAULT_TEXT_COLOR)
    app._write(f"  [bold {color}]{esc(icon)}[/bold {color}] [{text_color}]{esc(text)}[/{text_color}]")


def write_command_row(app: Any, command: str, desc: str, *, col: int, accent_color: str) -> None:
    muted = _theme_color(app, "muted", DEFAULT_MUTED_COLOR)
    gap = max(1, col - len(command))
    app._write(
        f"  [bold {accent_color}]{esc(command)}[/bold {accent_color}]{' ' * gap}[{muted}]{esc(desc)}[/{muted}]"
    )


def write_muted_lines(app: Any, rows: List[str]) -> None:
    muted = _theme_color(app, "muted", DEFAULT_MUTED_COLOR)
    for row in rows:
        app._write(f"  [{muted}]{esc(row)}[/{muted}]")


def write_usage(app: Any, usage: str) -> bool:
    app._write_error(f"Usage: {usage}")
    return True


def ensure_command_gap(app: Any) -> None:
    if not app._last_log_was_blank:
        app._write("")


def pending_attachment_markup(app: Any) -> str:
    if not app.pending:
        return ""
    chip_bg = _theme_color(app, "chip_bg", DEFAULT_CHIP_BG)
    chip_text = _theme_color(app, "chip_text", DEFAULT_CHIP_TEXT)
    muted = _theme_color(app, "muted", DEFAULT_MUTED_COLOR)
    chips: List[str] = []
    visible = app.pending[:3]
    for index, (path, _kind) in enumerate(visible, start=1):
        chips.append(f"[{chip_text} on {chip_bg}] {index}. {esc(os.path.basename(path))} [/{chip_text} on {chip_bg}]")
    overflow = len(app.pending) - len(visible)
    if overflow > 0:
        chips.append(f"[{muted} on {chip_bg}] +{overflow} more [/{muted} on {chip_bg}]")
    return " ".join(chips)
