from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.markup import escape as esc
from rich.padding import Padding
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.geometry import Offset
from textual.reactive import reactive
from textual.widgets import Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from agent.core import Agent, AgentTurnResult
from core.attachments import build_content, classify_attachment
from core.conv_tree import ConvTree, Turn
from tui.live_tool_preview import LiveToolPreviewManager
from tui.markdown_utils import fence_language, hanging_indent, render_md
from tui.popups import CodeViewerModal, ConfigEditorModal

DEFAULT_SAVE = "llamachat_tree.json"
MAX_REPLY_ACC_CHARS = 24000
SHELL_CONFIRM_TIMEOUT_S = 60
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_CONFIG_PATH = PROJECT_ROOT / "config" / "global_config.json"


@dataclass(frozen=True)
class CommandEntry:
    prompt: str
    insert_text: str
    description: str

HELP_SECTIONS = [
    (
        "CONVERSATION",
        [
            ("/help", "Show this help"),
            ("/think", "Toggle thinking mode"),
            ("/clear", "Clear tree and chat log"),
            ("/file <path>", "Attach image/text file to next message"),
            ("/quit /exit /q", "Exit app"),
        ],
    ),
    (
        "BRANCHING",
        [
            ("/branch [label]", "Arm next user message as a branch"),
            ("/unbranch", "Return to nearest branch fork parent"),
            ("/branches", "List child turns of current turn"),
            ("/switch <n>", "Switch to child branch by index"),
            ("/tree", "Render full conversation tree"),
        ],
    ),
    (
        "SKILLS",
        [
            ("/skills", "List installed skills"),
            ("/skill on <id>", "Enable skill"),
            ("/skill off <id>", "Disable skill"),
            ("/skill reload", "Reload skills from disk"),
            ("/skill info <id>", "Show skill details"),
        ],
    ),
    (
        "UTILITIES",
        [
            ("/memory stats", "Show memory stats"),
            ("/workspace tree", "Render workspace tree"),
            ("/config", "Edit global config in a popup"),
            ("/code [n|last]", "Open a copyable code block viewer"),
            ("/save [file]", f"Save tree JSON (default {DEFAULT_SAVE})"),
            ("/load [file]", f"Load tree JSON (default {DEFAULT_SAVE})"),
        ],
    ),
]

COMMAND_ENTRIES = [
    CommandEntry("/help", "/help", "Show this help"),
    CommandEntry("/think", "/think", "Toggle thinking mode"),
    CommandEntry("/clear", "/clear", "Clear tree and chat log"),
    CommandEntry("/file <path>", "/file ", "Attach a file to the next message"),
    CommandEntry("/branch [label]", "/branch ", "Arm the next user message as a branch"),
    CommandEntry("/unbranch", "/unbranch", "Return to the nearest branch fork"),
    CommandEntry("/branches", "/branches", "List branches from the current turn"),
    CommandEntry("/switch <n>", "/switch ", "Switch to a child branch"),
    CommandEntry("/tree", "/tree", "Render the full conversation tree"),
    CommandEntry("/skills", "/skills", "List installed skills"),
    CommandEntry("/skill on <id>", "/skill on ", "Enable a skill"),
    CommandEntry("/skill off <id>", "/skill off ", "Disable a skill"),
    CommandEntry("/skill reload", "/skill reload", "Reload skills from disk"),
    CommandEntry("/skill info <id>", "/skill info ", "Show skill details"),
    CommandEntry("/memory stats", "/memory stats", "Show memory stats"),
    CommandEntry("/workspace tree", "/workspace tree", "Render the workspace tree"),
    CommandEntry("/config", "/config", "Edit the global config in a popup"),
    CommandEntry("/code [n|last]", "/code ", "Open a copyable code block viewer"),
    CommandEntry("/save [file]", "/save ", f"Save tree JSON (default {DEFAULT_SAVE})"),
    CommandEntry("/load [file]", "/load ", f"Load tree JSON (default {DEFAULT_SAVE})"),
    CommandEntry("/quit", "/quit", "Exit app"),
]
class ChatInput(Input):
    BINDINGS = [
        Binding("ctrl+u", "clear_all", show=False),
        Binding("ctrl+k", "kill_to_end", show=False),
    ]

    def action_clear_all(self) -> None:
        self.value = ""

    def action_kill_to_end(self) -> None:
        self.value = self.value[: self.cursor_position]


class AlphanusTUI(App):
    TITLE = "Alphanus"

    CSS = """
    Screen {
        layout: vertical;
        background: #1c1c1c;
        color: #e0e0e0;
    }

    #main-area {
        height: 1fr;
        layout: horizontal;
    }

    #chat-scroll {
        width: 1fr;
        height: 1fr;
        background: #1c1c1c;
        scrollbar-size: 1 1;
        scrollbar-color: #606060 #2a2a2a;
    }

    #chat-log {
        width: 1fr;
        height: auto;
        background: #1c1c1c;
        padding: 0 2;
        scrollbar-size: 0 0;
    }

    #partial {
        width: 1fr;
        height: auto;
        background: #1c1c1c;
        display: none;
        padding: 0 2;
    }

    #sidebar {
        width: 34;
        border-left: solid #2e2e2e;
        background: #1c1c1c;
        display: none;
        padding: 0 1;
    }

    #sidebar-content {
        width: 1fr;
        height: auto;
        background: #1c1c1c;
    }

    #footer {
        height: auto;
        background: #1c1c1c;
        layout: vertical;
        dock: bottom;
    }

    #command-popup {
        width: 64;
        max-height: 12;
        background: #101418;
        border: round #5f87d7;
        display: none;
        overlay: screen;
        padding: 0 1;
    }

    #command-popup-title {
        height: auto;
        color: #8fb7ff;
        padding: 0 1;
        text-style: bold;
    }

    #command-popup-hint {
        height: auto;
        color: #a0a0a0;
        padding: 0 1;
    }

    #command-options {
        width: 1fr;
        height: auto;
        max-height: 8;
        background: #101418;
        border: none;
        padding: 0 0 1 0;
        scrollbar-background: #0d1319;
        scrollbar-background-hover: #111921;
        scrollbar-background-active: #111921;
        scrollbar-color: #3f5468;
        scrollbar-color-hover: #56718a;
        scrollbar-color-active: #6e90b0;
        scrollbar-corner-color: #0d1319;
    }

    #command-options > .option-list--option-highlighted {
        color: #f3f6fb;
        background: #1f3550;
        text-style: bold;
    }

    #command-options:focus > .option-list--option-highlighted {
        color: #ffffff;
        background: #29496d;
        text-style: bold;
    }

    #footer-sep {
        height: 1;
        background: #2a2a2a;
    }

    #status1, #status2 {
        height: 1;
        padding: 0 2;
        background: #1c1c1c;
    }

    #input-row {
        height: 1;
        layout: horizontal;
        background: #242424;
        padding: 0 2;
    }

    ChatInput {
        width: 1fr;
        height: 1;
        border: none;
        background: #242424;
        color: #e0e0e0;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", show=False),
        Binding("ctrl+d", "quit", show=False),
        Binding("escape", "handle_esc", show=False),
        Binding("pageup", "scroll_up", show=False),
        Binding("pagedown", "scroll_down", show=False),
    ]

    thinking: reactive[bool] = reactive(True)
    streaming: reactive[bool] = reactive(False)

    def __init__(self, agent: Agent, debug: bool = False):
        super().__init__()
        self.agent = agent
        self._debug_mode = debug

        tui_cfg = self.agent.config.get("tui", {})
        tree_cfg = tui_cfg.get("tree_compaction", {})
        chat_log_max_lines = int(tui_cfg.get("chat_log_max_lines", 5000))
        self._chat_log_max_lines = chat_log_max_lines if chat_log_max_lines > 0 else None
        self._tree_compaction_enabled = bool(tree_cfg.get("enabled", True))
        self._inactive_assistant_char_limit = int(tree_cfg.get("inactive_assistant_char_limit", 12000))
        self._inactive_tool_argument_char_limit = int(tree_cfg.get("inactive_tool_argument_char_limit", 5000))
        self._inactive_tool_content_char_limit = int(tree_cfg.get("inactive_tool_content_char_limit", 8000))

        self.conv_tree = self._new_conv_tree()
        self.pending: List[Tuple[str, str]] = []

        self._stop_event = threading.Event()
        self._active_turn_id: Optional[str] = None
        self._reply_acc = ""
        self._tool_activity_seen = False
        self._live_preview = LiveToolPreviewManager()

        self._reasoning_open = False
        self._content_open = False
        self._done_thinking_rendered = False
        self._buf_r = ""
        self._buf_c = ""
        self._in_fence = False
        self._fence_lang: Optional[str] = None
        self._fence_lines: List[str] = []

        self._last_scroll = 0.0
        self._scroll_interval = 0.05
        self._last_status2 = ""
        self._auto_follow_stream = True

        self._esc_pending = False
        self._esc_ts = 0.0
        self._spin_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spin_i = 0

        self._await_shell_confirm = False
        self._shell_confirm_command = ""
        self._shell_confirm_event: Optional[threading.Event] = None
        self._shell_confirm_result: Optional[Dict[str, bool]] = None
        self._command_matches: List[CommandEntry] = []
        self._command_anchor_region = None
        self._command_match_keys: List[str] = []
        self._code_blocks: List[Tuple[str, Optional[str]]] = []

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-area"):
            with ScrollableContainer(id="chat-scroll"):
                yield RichLog(
                    id="chat-log",
                    markup=True,
                    highlight=False,
                    wrap=True,
                    max_lines=self._chat_log_max_lines,
                )
                yield Static("", id="partial", markup=True)
            with ScrollableContainer(id="sidebar"):
                yield Static("", id="sidebar-content", markup=True)

        with Vertical(id="footer"):
            yield Static("", id="footer-sep")
            yield Static("", id="status1")
            yield Static("", id="status2")
            with Horizontal(id="input-row"):
                yield ChatInput(id="chat-input", placeholder="Type a message…")
            with Vertical(id="command-popup"):
                yield Static("commands", id="command-popup-title")
                yield Static("type to filter · tab to insert", id="command-popup-hint")
                yield OptionList(id="command-options")

    def on_mount(self) -> None:
        self.thinking = bool(self.agent.config.get("agent", {}).get("enable_thinking", True))
        self.set_interval(0.1, self._tick)
        self._show_banner()
        self._update_status1()
        self._update_status2()
        self._update_sidebar()
        self.query_one(ChatInput).focus()

    def on_resize(self, event) -> None:
        sidebar = self.query_one("#sidebar", ScrollableContainer)
        sidebar.display = event.size.width >= 120
        self._update_sidebar()
        if self._command_popup_active():
            self._command_anchor_region = self.query_one(ChatInput).region
            self._position_command_popup()

    def _new_conv_tree(self) -> ConvTree:
        return ConvTree(
            compact_inactive_branches=self._tree_compaction_enabled,
            inactive_assistant_char_limit=self._inactive_assistant_char_limit,
            inactive_tool_argument_char_limit=self._inactive_tool_argument_char_limit,
            inactive_tool_content_char_limit=self._inactive_tool_content_char_limit,
        )

    def _apply_tree_compaction_policy(self, tree: ConvTree) -> ConvTree:
        tree.set_compaction_policy(
            enabled=self._tree_compaction_enabled,
            inactive_assistant_char_limit=self._inactive_assistant_char_limit,
            inactive_tool_argument_char_limit=self._inactive_tool_argument_char_limit,
            inactive_tool_content_char_limit=self._inactive_tool_content_char_limit,
        )
        return tree

    def on_key(self, event) -> None:
        chat_input = self.query_one(ChatInput)
        if chat_input.has_focus and self._command_popup_active():
            key = event.key.lower()
            if key == "down":
                self._move_command_selection(1)
                event.stop()
                return
            if key == "up":
                self._move_command_selection(-1)
                event.stop()
                return
            if key == "tab":
                self._accept_command_selection()
                event.stop()
                return
        if not self._await_shell_confirm:
            return
        key = event.key.lower()
        if key == "y":
            self._finish_shell_confirm(True)
            event.stop()
        elif key in {"n", "escape"}:
            self._finish_shell_confirm(False)
            event.stop()

    def _tick(self) -> None:
        if self._esc_pending and time.monotonic() - self._esc_ts > 3.0:
            self._esc_pending = False
            self._update_status2()
            return
        if self.streaming:
            if self._auto_follow_stream and not self._is_near_bottom():
                self._auto_follow_stream = False
                self._update_status2()
            elif not self._auto_follow_stream and self._is_near_bottom():
                self._auto_follow_stream = True
                self._update_status2()
        if self.streaming and not self._esc_pending:
            self._spin_i += 1
            self._update_status2()

    def watch_streaming(self, value: bool) -> None:
        self.query_one(ChatInput).disabled = value
        if value:
            self._hide_command_popup()
        self._update_status2()

    def watch_thinking(self, value: bool) -> None:
        self._update_status1()
        self._update_status2()

    def _log(self) -> RichLog:
        return self.query_one("#chat-log", RichLog)

    def _scroll(self) -> ScrollableContainer:
        return self.query_one("#chat-scroll", ScrollableContainer)

    def _partial(self) -> Static:
        return self.query_one("#partial", Static)

    def _command_popup(self) -> Vertical:
        return self.query_one("#command-popup", Vertical)

    def _command_options(self) -> OptionList:
        return self.query_one("#command-options", OptionList)

    def _write(self, markup: str) -> None:
        self._log().write(Text.from_markup(markup))
        self._maybe_scroll_end()

    def _write_indented(self, markup: str, indent: int = 2) -> None:
        self._log().write(Padding(Text.from_markup(markup), pad=(0, 0, 0, indent)))
        self._maybe_scroll_end()

    def _write_renderable(self, renderable, indent: int = 2) -> None:
        self._log().write(Padding(renderable, pad=(0, 0, 0, indent)))
        self._maybe_scroll_end()

    def _syntax_renderable(self, code: str, language: Optional[str]) -> Syntax:
        return Syntax(
            code,
            language or "text",
            theme="monokai",
            word_wrap=True,
            background_color="#0b0b0b",
            line_numbers=False,
        )

    def _code_panel_renderable(self, code: str, language: Optional[str]) -> Panel:
        return Panel(
            self._syntax_renderable(code, language),
            expand=True,
            padding=(0, 1),
            border_style="#2e2e2e",
            style="on #0b0b0b",
        )

    def _remember_code_block(self, code: str, language: Optional[str]) -> int:
        self._code_blocks.append((code, language))
        if len(self._code_blocks) > 64:
            self._code_blocks = self._code_blocks[-64:]
        return len(self._code_blocks)

    def _write_code_block(self, lines: List[str], language: Optional[str], indent: int = 2) -> None:
        code = "\n".join(lines)
        block_index = self._remember_code_block(code, language)
        self._write_renderable(self._code_panel_renderable(code, language), indent=indent)
        self._write_indented(
            f"[dim]code block {block_index} · /code {block_index} to open copyable view[/dim]",
            indent=indent + 2,
        )

    def _reset_fence_state(self) -> None:
        self._in_fence = False
        self._fence_lang = None
        self._fence_lines = []

    def _flush_fence_block(self) -> None:
        if self._fence_lines:
            self._write_code_block(self._fence_lines, self._fence_lang, indent=2)
        self._reset_fence_state()

    def _render_content_line(self, line: str) -> None:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            if self._in_fence:
                self._flush_fence_block()
            else:
                self._in_fence = True
                self._fence_lang = fence_language(line)
                self._fence_lines = []
            return

        if self._in_fence:
            self._fence_lines.append(line)
            return

        rendered, _ = render_md(line, False)
        self._write_indented(rendered, indent=max(2, hanging_indent(line)))

    def _update_partial_content(self) -> None:
        partial = self._partial()
        if self._in_fence:
            lines = list(self._fence_lines)
            if self._buf_c:
                lines.append(self._buf_c)
            if lines:
                partial.update(Padding(self._code_panel_renderable("\n".join(lines), self._fence_lang), (0, 0, 0, 2)))
            else:
                partial.update("")
            return
        partial.update(f"  {esc(self._buf_c)}" if self._buf_c else "")

    def _update_live_preview_partial(self, lines: List[str], language: Optional[str]) -> None:
        partial = self._partial()
        partial.display = True
        partial.update(Padding(self._code_panel_renderable("\n".join(lines), language), (0, 0, 0, 2)))

    def _clear_partial_preview(self) -> None:
        partial = self._partial()
        partial.update("")
        if not self.streaming:
            partial.display = False

    def _is_near_bottom(self, threshold: float = 1.0) -> bool:
        scroll = self._scroll()
        try:
            return (scroll.max_scroll_y - scroll.scroll_y) <= threshold
        except Exception:
            return True

    def _maybe_scroll_end(self, force: bool = False) -> None:
        if force:
            self._scroll().scroll_end(animate=False)
            return
        if not self.streaming or self._auto_follow_stream:
            self._scroll().scroll_end(animate=False)

    def _write_info(self, text: str) -> None:
        self._write(f"[dim]  {esc(text)}[/dim]")

    def _write_error(self, text: str) -> None:
        self._write(f"[bold red]  ✖ {esc(text)}[/bold red]")

    def _write_section_heading(self, title: str, color: str = "cyan") -> None:
        self._write("")
        self._write(f"[bold {color}]  {esc(title)}[/bold {color}]")

    def _write_detail_line(self, label: str, value: str, *, value_markup: bool = False) -> None:
        rendered = value if value_markup else esc(value)
        self._write(f"  [dim]{esc(label)}:[/dim] {rendered}")

    def _append_reply_token(self, token: str) -> None:
        if not token:
            return
        if len(self._reply_acc) >= MAX_REPLY_ACC_CHARS:
            return
        remaining = MAX_REPLY_ACC_CHARS - len(self._reply_acc)
        self._reply_acc += token[:remaining]

    def _is_tool_trace_line(self, line: str) -> bool:
        s = line.strip().lower()
        return "tool call:" in s

    def _update_status1(self) -> None:
        parts = []
        if self.thinking:
            parts.append("[#5f87d7]thinking[/#5f87d7]")
        else:
            parts.append("[dim]thinking off[/dim]")

        if self.conv_tree._pending_branch:
            label = self.conv_tree._pending_branch_label
            suffix = f" '{esc(label)}'" if label else ""
            parts.append(f"[yellow]⎇ armed{suffix}[/yellow]")

        if self.pending:
            badges = " ".join(
                f"[black on {'cyan' if kind == 'image' else 'green'}] {esc(os.path.basename(path))} [/black on {'cyan' if kind == 'image' else 'green'}]"
                for path, kind in self.pending
            )
            parts.append(badges)

        self.query_one("#status1", Static).update(" " + "  [dim]·[/dim]  ".join(parts))

    def _status2(self, text: str) -> None:
        if text == self._last_status2:
            return
        self._last_status2 = text
        self.query_one("#status2", Static).update(text)

    def _update_status2(self) -> None:
        turns = self.conv_tree.turn_count()
        right = f"[dim]{turns} turn{'s' if turns != 1 else ''}[/dim]"

        if self._await_shell_confirm:
            left = f"[bold yellow]approve shell command?[/bold yellow] [dim][y/n][/dim]"
        elif self.streaming:
            frame = self._spin_frames[self._spin_i % len(self._spin_frames)]
            if self._stop_event.is_set():
                left = f"[dim]{frame}[/dim] [yellow]stopping...[/yellow]"
            elif self._esc_pending:
                left = f"[dim]{frame}[/dim] [dim]generating[/dim] [bold red]esc again to stop[/bold red]"
            elif not self._auto_follow_stream:
                left = f"[dim]{frame}[/dim] [dim]generating[/dim] [yellow]free scroll[/yellow] [dim]pgdn to resume follow[/dim]"
            else:
                left = f"[dim]{frame}[/dim] [dim]generating[/dim] [dim]esc · stop[/dim]"
        else:
            left = "[dim]esc · clear   pgup/dn · scroll[/dim]"

        self._status2(f" {left}  {right}")

    def _show_banner(self) -> None:
        self._write("")
        self._write("[bold #5f87d7]Alphanus[/bold #5f87d7] [dim]conversation tree · skills · streaming[/dim]")
        self._write(f"[dim]{esc(self.agent.model_endpoint)}[/dim]")
        self._write("[dim]/help for commands[/dim]")
        self._write("")

    def _update_input_placeholder(self) -> None:
        self.query_one(ChatInput).placeholder = (
            "Type to start branch…" if self.conv_tree._pending_branch else "Type a message…"
        )

    def _command_entries_for_query(self, value: str) -> List[CommandEntry]:
        query = value.strip().lower()
        if not query.startswith("/"):
            return []
        needle = query[1:]
        if not needle:
            return COMMAND_ENTRIES[:10]

        def sort_key(entry: CommandEntry) -> Tuple[int, int, str]:
            haystack = f"{entry.prompt} {entry.description}".lower()
            starts = 0 if entry.prompt.lower().startswith(f"/{needle}") else 1
            pos = haystack.find(needle)
            return (starts, pos if pos >= 0 else 9999, entry.prompt)

        matches = [
            entry
            for entry in COMMAND_ENTRIES
            if needle in entry.prompt.lower() or needle in entry.description.lower()
        ]
        matches.sort(key=sort_key)
        return matches[:8]

    def _refresh_command_popup(self, value: str) -> None:
        popup = self._command_popup()
        options = self._command_options()
        next_matches = self._command_entries_for_query(value)
        next_keys = [entry.prompt for entry in next_matches]
        if not next_matches or self.streaming or self._await_shell_confirm:
            next_matches = []
            next_keys = []

        if not next_matches:
            self._command_matches = []
            self._command_match_keys = []
            popup.display = False
            options.clear_options()
            popup.absolute_offset = None
            return

        was_hidden = not bool(popup.display)
        geometry_changed = self._command_anchor_region != self.query_one(ChatInput).region
        self._command_anchor_region = self.query_one(ChatInput).region
        self._command_matches = next_matches

        if next_keys != self._command_match_keys:
            self._command_match_keys = next_keys
            popup.display = True
            rendered = [
                Option(
                    f"[bold #8fb7ff]{esc(entry.prompt)}[/bold #8fb7ff] [dim]{esc(entry.description)}[/dim]",
                    id=str(index),
                )
                for index, entry in enumerate(self._command_matches)
            ]
            options.clear_options()
            options.add_options(rendered)
            options.highlighted = 0
            self.call_after_refresh(self._position_command_popup)
            if was_hidden:
                self.set_timer(0.02, self._position_command_popup)
            return

        popup.display = True
        if was_hidden or geometry_changed:
            self.call_after_refresh(self._position_command_popup)
            if was_hidden:
                self.set_timer(0.02, self._position_command_popup)

    def _position_command_popup(self) -> None:
        if not self._command_matches:
            return
        popup = self._command_popup()
        chat_input = self.query_one(ChatInput)
        option_rows = min(len(self._command_matches), 8)
        popup_height = option_rows + 4
        popup_width = max(44, min(72, max(chat_input.region.width, 44)))
        popup.styles.height = popup_height
        popup.styles.width = popup_width

        input_region = self._command_anchor_region or chat_input.region
        x = max(1, int(input_region.x))
        above_y = int(input_region.y) - popup_height - 1
        below_y = int(input_region.y + input_region.height)
        y = above_y if above_y >= 1 else below_y
        popup.absolute_offset = Offset(x, y)

    def _hide_command_popup(self) -> None:
        self._command_matches = []
        self._command_anchor_region = None
        self._command_match_keys = []
        self._command_options().clear_options()
        self._command_popup().display = False
        self._command_popup().absolute_offset = None

    def _command_popup_active(self) -> bool:
        return bool(self._command_matches) and bool(self._command_popup().display)

    def _move_command_selection(self, delta: int) -> None:
        if not self._command_popup_active():
            return
        options = self._command_options()
        current = 0 if options.highlighted is None else int(options.highlighted)
        count = len(self._command_matches)
        if count <= 0:
            return
        options.highlighted = (current + delta) % count
        options.scroll_to_highlight(top=False)

    def _accept_command_selection(self) -> bool:
        if not self._command_popup_active():
            return False
        options = self._command_options()
        highlighted = 0 if options.highlighted is None else int(options.highlighted)
        if highlighted < 0 or highlighted >= len(self._command_matches):
            return False
        entry = self._command_matches[highlighted]
        chat_input = self.query_one(ChatInput)
        chat_input.value = entry.insert_text
        chat_input.cursor_position = len(chat_input.value)
        self._refresh_command_popup(chat_input.value)
        return True

    def _should_accept_popup_on_enter(self, text: str) -> bool:
        stripped = text.strip()
        if not self._command_popup_active() or not stripped.startswith("/"):
            return False
        if " " in stripped:
            return False
        base = stripped.lower()
        exact = {entry.insert_text.strip().lower() for entry in COMMAND_ENTRIES}
        return base not in exact

    def _open_config_editor(self) -> None:
        raw = GLOBAL_CONFIG_PATH.read_text(encoding="utf-8")
        self.push_screen(ConfigEditorModal(GLOBAL_CONFIG_PATH, raw), self._on_config_editor_close)

    def _merge_live_config(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = dict(base)
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merged[key] = self._merge_live_config(base[key], value)
            else:
                merged[key] = value
        return merged

    def _apply_tui_config(self) -> None:
        tui_cfg = self.agent.config.get("tui", {})
        tree_cfg = tui_cfg.get("tree_compaction", {})
        chat_log_max_lines = int(tui_cfg.get("chat_log_max_lines", 5000))
        self._chat_log_max_lines = chat_log_max_lines if chat_log_max_lines > 0 else None
        self._tree_compaction_enabled = bool(tree_cfg.get("enabled", True))
        self._inactive_assistant_char_limit = int(tree_cfg.get("inactive_assistant_char_limit", 12000))
        self._inactive_tool_argument_char_limit = int(tree_cfg.get("inactive_tool_argument_char_limit", 5000))
        self._inactive_tool_content_char_limit = int(tree_cfg.get("inactive_tool_content_char_limit", 8000))
        self.conv_tree = self._apply_tree_compaction_policy(self.conv_tree)
        try:
            self._log().max_lines = self._chat_log_max_lines
        except Exception:
            pass
        self._update_status1()
        self._update_status2()
        self._update_sidebar()

    def _on_config_editor_close(self, result: Optional[Dict[str, Any]]) -> None:
        if not result:
            return
        text = str(result.get("text", ""))
        parsed = result.get("config")
        if not isinstance(parsed, dict):
            self._write_error("Config save failed: invalid config payload")
            return

        GLOBAL_CONFIG_PATH.write_text(text, encoding="utf-8")
        merged = self._merge_live_config(self.agent.config, parsed)
        self.agent.config = merged
        self.agent.skill_runtime.config = merged
        self.thinking = bool(merged.get("agent", {}).get("enable_thinking", self.thinking))
        self._apply_tui_config()
        self._write_info("Saved global config. Restart to apply endpoint, workspace, or memory changes.")

    def _open_code_block(self, index: int) -> None:
        if index < 1 or index > len(self._code_blocks):
            self._write_error(f"No code block {index}")
            return
        code, language = self._code_blocks[index - 1]
        self.push_screen(CodeViewerModal(code, language, title=f"Code Block {index}"))

    def _update_sidebar(self) -> None:
        sidebar = self.query_one("#sidebar", ScrollableContainer)
        if not sidebar.display:
            return
        lines = [f"[dim]conversation tree · {self.conv_tree.turn_count()} turns[/dim]", ""]
        cur = self.conv_tree.current_id
        for text, tag, active in self.conv_tree.render_tree(width=30):
            line = esc(text)
            if tag == "root":
                lines.append(f"[dim]{line}[/dim]")
            elif tag == cur:
                lines.append(f"[bold #5f87d7]{line}[/bold #5f87d7]")
            elif active:
                lines.append(f"[#5f87d7]{line}[/#5f87d7]")
            else:
                lines.append(f"[dim]{line}[/dim]")
        self.query_one("#sidebar-content", Static).update("\n".join(lines))

    def _write_turn_user(self, turn: Turn) -> None:
        self._write("")
        if turn.branch_root:
            label = f" ⎇  {esc(turn.label)}" if turn.label else " ⎇  new branch"
            self._write(f"[dim yellow]{label}[/dim yellow]")
        self._write("[bold #5faf5f]You[/bold #5faf5f]")
        body = turn.user_text()
        for line in body.splitlines() or [""]:
            self._write_indented(esc(line), indent=2)

    def _write_skill_exchanges(self, turn: Turn) -> None:
        for msg in turn.skill_exchanges:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for call in msg["tool_calls"]:
                    name = call.get("function", {}).get("name", "unknown")
                    raw_args = call.get("function", {}).get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args = raw_args
                    self._write(
                        f"[dim]  · tool call: {esc(name)}({esc(self._live_preview.compact_tool_args(name, args))})[/dim]"
                    )
                    self._live_preview.write_static_preview(
                        name, args, self._write, self._write_indented, self._write_code_block
                    )
            elif msg.get("role") == "tool":
                name = msg.get("name", "tool")
                try:
                    payload = json.loads(msg.get("content", "{}"))
                except Exception:
                    payload = {"ok": False, "error": {"message": "invalid tool response"}}
                if payload.get("ok"):
                    self._write(f"[dim green]  · {esc(name)} ✓[/dim green]")
                else:
                    em = payload.get("error", {}).get("message", "failed")
                    self._write(f"[dim red]  · {esc(name)} ✗ {esc(em)}[/dim red]")

    def _write_completed_turn_asst(self, turn: Turn) -> None:
        self._write("")
        self._write("[bold #5f87d7]Assistant[/bold #5f87d7]")
        self._write_skill_exchanges(turn)

        content = turn.assistant_content or ""
        interrupted = "[interrupted]" in content
        display = content.replace("\n[interrupted]", "").rstrip()

        in_fence = False
        fence_lang: Optional[str] = None
        fence_lines: List[str] = []
        for line in display.splitlines() or [""]:
            stripped = line.strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                if in_fence:
                    if fence_lines:
                        self._write_code_block(fence_lines, fence_lang, indent=2)
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
            self._write_indented(rendered, indent=max(2, hanging_indent(line)))

        if in_fence and fence_lines:
            self._write_code_block(fence_lines, fence_lang, indent=2)

        if interrupted:
            self._write("[dim red]  ✖ interrupted[/dim red]")
        self._write("")

    def _rebuild_viewport(self) -> None:
        self._log().clear()
        for turn in self.conv_tree.active_path:
            if turn.id == "root":
                continue
            self._write_turn_user(turn)
            if turn.assistant_content:
                self._write_completed_turn_asst(turn)
        self._maybe_scroll_end(force=True)

    def _start_stream(self, turn: Turn, user_input: str, attachment_paths: List[str]) -> None:
        self.streaming = True
        self._auto_follow_stream = True
        self._active_turn_id = turn.id
        self._reply_acc = ""
        self._tool_activity_seen = False
        self._live_preview.reset()
        self._reasoning_open = False
        self._content_open = False
        self._done_thinking_rendered = False
        self._buf_r = ""
        self._buf_c = ""
        self._reset_fence_state()
        self._stop_event = threading.Event()
        self._partial().display = True

        self._write("")
        self._write("[bold #5f87d7]Assistant[/bold #5f87d7]")

        branch_labels = [t.label for t in self.conv_tree.active_path if t.branch_root and t.label]
        history_messages = self.conv_tree.history_messages()
        self._stream_worker(turn.id, history_messages, user_input, branch_labels, attachment_paths, self.thinking, self._stop_event)

    @work(thread=True, exclusive=True)
    def _stream_worker(
        self,
        turn_id: str,
        history_messages: List[Dict[str, Any]],
        user_input: str,
        branch_labels: List[str],
        attachment_paths: List[str],
        thinking: bool,
        stop_event: threading.Event,
    ) -> None:
        def on_event(event: Dict[str, Any]) -> None:
            self.call_from_thread(self._on_agent_event, event)

        result = self.agent.run_turn(
            history_messages=history_messages,
            user_input=user_input,
            thinking=thinking,
            branch_labels=branch_labels,
            attachments=attachment_paths,
            stop_event=stop_event,
            on_event=on_event,
            confirm_shell=self._confirm_shell_command,
        )
        self.call_from_thread(self._on_stream_end, turn_id, result)

    def _flush_reasoning_buffer(self) -> None:
        if not self._buf_r:
            self._partial().update("")
            return
        text = self._buf_r
        self._buf_r = ""
        self._partial().update("")
        for line in text.splitlines():
            if self._is_tool_trace_line(line):
                continue
            rendered, _ = render_md(line, False)
            self._write_indented(f"[dim]{rendered}[/dim]", indent=4)

    def _flush_content_buffer(self, include_partial: bool = False) -> None:
        while "\n" in self._buf_c:
            line, self._buf_c = self._buf_c.split("\n", 1)
            if self._is_tool_trace_line(line):
                continue
            self._render_content_line(line)

        if include_partial and self._buf_c:
            if not self._is_tool_trace_line(self._buf_c):
                if self._in_fence:
                    self._fence_lines.append(self._buf_c)
                    self._flush_fence_block()
                else:
                    rendered, _ = render_md(self._buf_c, False)
                    self._write_indented(rendered, indent=max(2, hanging_indent(self._buf_c)))
            self._buf_c = ""
            self._partial().update("")
            return

        self._update_partial_content()

    def _handle_content_token(self, token: str) -> None:
        if not self._content_open:
            self._content_open = True
            if self._reasoning_open:
                self._flush_reasoning_buffer()
                if not self._done_thinking_rendered:
                    self._write("[bold #5f87d7]· done thinking[/bold #5f87d7]")
                    self._write("")
                    self._done_thinking_rendered = True
        self._buf_c += token
        self._append_reply_token(token)
        self._flush_content_buffer(include_partial=False)

    def _on_agent_event(self, event: Dict[str, Any]) -> None:
        partial = self._partial()
        etype = event.get("type")

        if etype == "reasoning_token":
            token = event.get("text", "")
            if not self._reasoning_open:
                self._reasoning_open = True
                self._write("[bold #5f87d7]· thinking[/bold #5f87d7]")
            self._buf_r += token
            display = self._buf_r
            if "\n" in display:
                prefix, last = display.rsplit("\n", 1)
                if self._is_tool_trace_line(last):
                    display = prefix + ("\n" if prefix else "")
            elif self._is_tool_trace_line(display):
                display = ""
            partial.update(f"[dim]  {esc(display)}[/dim]" if display else "")

        elif etype == "content_token":
            token = event.get("text", "")
            self._handle_content_token(token)

        elif etype == "tool_phase_started":
            self._tool_activity_seen = True
            # Preserve in-progress text before tool call deltas start.
            if self._reasoning_open:
                self._flush_reasoning_buffer()
            self._flush_content_buffer(include_partial=True)
            self._content_open = False

        elif etype == "tool_call_delta":
            self._tool_activity_seen = True
            stream_id = str(event.get("stream_id") or "")
            name = str(event.get("name") or "")
            raw_arguments = str(event.get("raw_arguments") or "")
            if stream_id and name:
                self._live_preview.update(
                    stream_id, name, raw_arguments, self._write, self._update_live_preview_partial
                )

        elif etype == "tool_call":
            self._tool_activity_seen = True
            self._flush_reasoning_buffer()
            name = event.get("name", "tool")
            args = event.get("arguments", {})
            stream_id = str(event.get("stream_id") or "")
            self._write(
                f"[dim]  · tool call: {esc(name)}({esc(self._live_preview.compact_tool_args(name, args))})[/dim]"
            )
            streamed = (
                self._live_preview.close(
                    stream_id, self._write_indented, self._write_code_block, self._clear_partial_preview
                )
                if stream_id
                else False
            )
            if not streamed:
                self._live_preview.write_static_preview(
                    name, args, self._write, self._write_indented, self._write_code_block
                )

        elif etype == "tool_result":
            self._tool_activity_seen = True
            self._flush_reasoning_buffer()
            name = event.get("name", "tool")
            result = event.get("result", {})
            if result.get("ok"):
                self._write(f"[dim green]  · {esc(name)} ✓[/dim green]")
            else:
                msg = result.get("error", {}).get("message", "failed")
                self._write(f"[dim red]  · {esc(name)} ✗ {esc(msg)}[/dim red]")

        elif etype == "error":
            self._write_error(str(event.get("text", "Unknown error")))

        elif etype == "info":
            self._write_info(str(event.get("text", "")))

        elif etype == "pass_end":
            finish_reason = str(event.get("finish_reason") or "")
            has_content = bool(event.get("has_content"))
            has_tool_calls = bool(event.get("has_tool_calls"))
            # Some llama-server/model passes end with reasoning-only stop and no
            # visible output. Drop that provisional reasoning so it doesn't leak.
            if finish_reason in {"stop", "length"} and not has_content and not has_tool_calls:
                self._buf_r = ""
                partial.update("")

        now = time.monotonic()
        if now - self._last_scroll >= self._scroll_interval:
            self._maybe_scroll_end()
            self._last_scroll = now

    def _on_stream_end(self, turn_id: str, result: AgentTurnResult) -> None:
        partial = self._partial()
        partial.update("")
        partial.display = False

        self._live_preview.close_all(self._write_indented, self._write_code_block, self._clear_partial_preview)

        if self._buf_r and not self._content_open:
            if not self._is_tool_trace_line(self._buf_r):
                rendered, _ = render_md(self._buf_r, False)
                self._write_indented(f"[dim]{rendered}[/dim]", indent=4)
            self._buf_r = ""

        if self._reasoning_open and not self._content_open:
            if not self._done_thinking_rendered:
                self._write("[bold #5f87d7]· done thinking[/bold #5f87d7]")
                self._write("")
                self._done_thinking_rendered = True

        self._flush_content_buffer(include_partial=True)

        reply = result.content if result.content else self._reply_acc
        if result.status == "done" and not self._content_open and reply.strip():
            in_fence = False
            fence_lang: Optional[str] = None
            fence_lines: List[str] = []
            for line in reply.splitlines() or [""]:
                stripped = line.strip()
                if stripped.startswith("```") or stripped.startswith("~~~"):
                    if in_fence:
                        if fence_lines:
                            self._write_code_block(fence_lines, fence_lang, indent=2)
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
                self._write_indented(rendered, indent=max(2, hanging_indent(line)))

            if in_fence and fence_lines:
                self._write_code_block(fence_lines, fence_lang, indent=2)

        if turn_id in self.conv_tree.nodes:
            for msg in result.skill_exchanges:
                self.conv_tree.append_skill_exchange(turn_id, msg)

        if result.status == "done":
            self.conv_tree.complete_turn(turn_id, reply)
        else:
            self.conv_tree.cancel_turn(turn_id, reply)
            if result.error:
                self._write_error(result.error)
            if result.status == "cancelled":
                self._write("[bold red]  ✖ interrupted[/bold red]")

        self._write("")
        self._maybe_scroll_end()
        self.streaming = False
        self._live_preview.reset()
        self._active_turn_id = None
        self._esc_pending = False
        self._auto_follow_stream = True
        self._update_status1()
        self._update_status2()
        self._update_sidebar()

    def _confirm_shell_command(self, command: str) -> bool:
        event = threading.Event()
        holder = {"value": False}
        self.call_from_thread(self._begin_shell_confirm, command, event, holder)
        if not event.wait(timeout=SHELL_CONFIRM_TIMEOUT_S):
            self.call_from_thread(self._expire_shell_confirm, event)
            return False
        return bool(holder.get("value", False))

    def _begin_shell_confirm(self, command: str, event: threading.Event, holder: Dict[str, bool]) -> None:
        self._await_shell_confirm = True
        self._shell_confirm_command = command
        self._shell_confirm_event = event
        self._shell_confirm_result = holder
        self._write(f"[yellow]  ? Run shell command: {esc(command)}[/yellow]")
        self._update_status2()

    def _expire_shell_confirm(self, event: threading.Event) -> None:
        if not self._await_shell_confirm:
            return
        if self._shell_confirm_event is not event:
            return
        if self._shell_confirm_result is not None:
            self._shell_confirm_result["value"] = False
        self._shell_confirm_event.set()
        self._write("[dim red]  · shell command approval timed out[/dim red]")
        self._await_shell_confirm = False
        self._shell_confirm_command = ""
        self._shell_confirm_event = None
        self._shell_confirm_result = None
        self._update_status2()

    def _finish_shell_confirm(self, approved: bool) -> None:
        if not self._await_shell_confirm:
            return
        if self._shell_confirm_result is not None:
            self._shell_confirm_result["value"] = approved
        if self._shell_confirm_event is not None:
            self._shell_confirm_event.set()
        msg = "approved" if approved else "rejected"
        color = "green" if approved else "red"
        self._write(f"[dim {color}]  · shell command {msg}[/dim {color}]")

        self._await_shell_confirm = False
        self._shell_confirm_command = ""
        self._shell_confirm_event = None
        self._shell_confirm_result = None
        self._update_status2()

    def _send(self, text: str) -> None:
        attachments = list(self.pending)
        self.pending.clear()
        content = build_content(text, attachments)
        turn = self.conv_tree.add_turn(content)
        self._write_turn_user(turn)
        self._update_status1()
        self._update_status2()
        self._update_input_placeholder()
        self._update_sidebar()
        self._start_stream(turn, text, [p for p, _ in attachments])

    @on(Input.Submitted, "#chat-input")
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            self.query_one(ChatInput).value = ""
            return
        if self._should_accept_popup_on_enter(text):
            self._accept_command_selection()
            return
        self.query_one(ChatInput).value = ""
        self._hide_command_popup()
        if not self._handle_command(text):
            if not self.streaming:
                self._send(text)

    @on(Input.Changed, "#chat-input")
    def on_input_changed(self, event: Input.Changed) -> None:
        self._refresh_command_popup(event.value)

    @on(OptionList.OptionSelected, "#command-options")
    def on_command_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = event.option.id
        if option_id is None:
            return
        try:
            index = int(option_id)
        except ValueError:
            return
        if index < 0 or index >= len(self._command_matches):
            return
        entry = self._command_matches[index]
        chat_input = self.query_one(ChatInput)
        chat_input.value = entry.insert_text
        chat_input.cursor_position = len(chat_input.value)
        chat_input.focus()
        self._refresh_command_popup(chat_input.value)

    def action_handle_esc(self) -> None:
        if self._await_shell_confirm:
            self._finish_shell_confirm(False)
            return
        if self._command_popup_active():
            self._hide_command_popup()
            return
        if not self.streaming:
            self.query_one(ChatInput).value = ""
            return
        now = time.monotonic()
        if not self._esc_pending:
            self._esc_pending = True
            self._esc_ts = now
            self._update_status2()
        else:
            self._stop_event.set()
            self._esc_pending = False
            self._update_status2()

    def action_scroll_up(self) -> None:
        self._scroll().scroll_page_up()
        if self.streaming:
            self._auto_follow_stream = False
            self._update_status2()

    def action_scroll_down(self) -> None:
        self._scroll().scroll_page_down()
        if self.streaming and self._is_near_bottom():
            self._auto_follow_stream = True
            self._update_status2()

    def _handle_command(self, text: str) -> bool:
        parts = text.strip().split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in {"/quit", "/exit", "/q"}:
            self.exit()
            return True

        if cmd == "/help":
            self._cmd_help()
            return True

        if cmd == "/think":
            self.thinking = not self.thinking
            self._write_info(f"Thinking {'enabled' if self.thinking else 'disabled'}")
            return True

        if cmd == "/branch":
            self.conv_tree.arm_branch(arg)
            suffix = f" '{arg}'" if arg else ""
            self._write(f"[yellow]  ⎇ Branch armed{esc(suffix)}[/yellow]")
            self._update_status1()
            self._update_input_placeholder()
            return True

        if cmd == "/unbranch":
            moved = self.conv_tree.unbranch()
            if moved is None:
                self._write_error("No branch to leave.")
            else:
                self._write("[yellow]  ↩ Returned to fork point[/yellow]")
                self._rebuild_viewport()
                self._update_sidebar()
            self._update_status1()
            return True

        if cmd == "/branches":
            children = self.conv_tree.current.children
            if not children:
                self._write_info("No child branches from current turn.")
            else:
                self._write_section_heading("Children")
                for i, cid in enumerate(children):
                    t = self.conv_tree.nodes[cid]
                    self._write(f"  [yellow]{i}.[/yellow] [dim]{esc(t.short(60))}[/dim]")
            return True

        if cmd == "/switch":
            try:
                idx = int(arg)
            except ValueError:
                self._write_error("/switch requires an integer index")
                return True
            turn = self.conv_tree.switch_child(idx)
            if not turn:
                self._write_error(f"No child {idx} at current node")
            else:
                self._write(f"[yellow]  ↪ Switched to branch {idx}[/yellow]")
                self._rebuild_viewport()
                self._update_sidebar()
            return True

        if cmd == "/tree":
            self._cmd_tree()
            return True

        if cmd == "/save":
            path = arg or DEFAULT_SAVE
            try:
                self.conv_tree.save(path)
                self._write(f"[yellow]  ✓ Saved tree to {esc(path)}[/yellow]")
            except Exception as exc:
                self._write_error(f"Save failed: {exc}")
            return True

        if cmd == "/load":
            path = arg or DEFAULT_SAVE
            if not os.path.isfile(path):
                self._write_error(f"File not found: {path}")
                return True
            try:
                loaded = ConvTree.load(path)
                self.conv_tree = self._apply_tree_compaction_policy(loaded)
                self._rebuild_viewport()
                self._update_sidebar()
                self._update_status1()
                self._write(f"[yellow]  ✓ Loaded tree from {esc(path)}[/yellow]")
            except Exception as exc:
                self._write_error(f"Load failed: {exc}")
            return True

        if cmd == "/clear":
            self.conv_tree = self._new_conv_tree()
            self.pending.clear()
            self._log().clear()
            self._partial().update("")
            self._show_banner()
            self._update_status1()
            self._update_status2()
            self._update_sidebar()
            self._update_input_placeholder()
            return True

        if cmd in {"/file", "/image"}:
            if not arg:
                self._write_error(f"{cmd} requires a path")
                return True
            path = os.path.expanduser(arg)
            if not os.path.isfile(path):
                self._write_error(f"File not found: {path}")
                return True
            kind = classify_attachment(path)
            if kind == "unknown":
                self._write_error("Unsupported file type")
                return True
            self.pending.append((path, kind))
            self._write(f"[dim]  attached {kind}: {esc(os.path.basename(path))}[/dim]")
            self._update_status1()
            return True

        if cmd == "/skills":
            self._cmd_skills()
            return True

        if cmd == "/skill":
            return self._cmd_skill(arg)

        if cmd == "/memory":
            return self._cmd_memory(arg)

        if cmd == "/workspace":
            return self._cmd_workspace(arg)

        if cmd == "/config":
            self._open_config_editor()
            return True

        if cmd == "/code":
            return self._cmd_code(arg)

        if cmd.startswith("/"):
            self._write_error(f"Unknown command: {cmd}")
            return True

        return False

    def _cmd_help(self) -> None:
        self._write("")
        col = 22
        for section, rows in HELP_SECTIONS:
            self._write(f"[bold yellow]  {section}[/bold yellow]")
            for c, desc in rows:
                self._write(f"  [yellow]{esc(c):<{col}}[/yellow] [dim]{esc(desc)}[/dim]")
            self._write("")

    def _cmd_tree(self) -> None:
        self._write("")
        for text, tag, active in self.conv_tree.render_tree(width=80):
            if tag == self.conv_tree.current_id:
                self._write(f"[bold yellow]  {esc(text)}[/bold yellow]")
            elif active:
                self._write(f"[yellow]  {esc(text)}[/yellow]")
            else:
                self._write(f"[dim]  {esc(text)}[/dim]")

    def _cmd_skills(self) -> None:
        skills = self.agent.skill_runtime.list_skills()
        self._write_section_heading("Skills")
        name_col = max((len(skill.id) for skill in skills), default=0) + 2
        for skill in skills:
            state = "on" if skill.enabled else "off"
            color = "green" if skill.enabled else "red"
            self._write(
                f"  [bold]{esc(skill.id):<{name_col}}[/bold][dim]({esc(skill.version)})[/dim] "
                f"[{color}]{state}[/{color}] [dim]{esc(skill.description)}[/dim]"
            )

    def _cmd_skill(self, arg: str) -> bool:
        parts = arg.split()
        if not parts:
            self._write_error("Usage: /skill on|off|reload|info <id>")
            return True

        sub = parts[0].lower()
        if sub == "reload":
            self.agent.skill_runtime.load_skills()
            self._write_info("Reloaded skills")
            return True

        if sub in {"on", "off"}:
            if len(parts) < 2:
                self._write_error("/skill on|off requires a skill id")
                return True
            skill_id = parts[1]
            ok = self.agent.skill_runtime.set_enabled(skill_id, sub == "on")
            if not ok:
                self._write_error(f"Skill not found: {skill_id}")
            else:
                self._write_info(f"Skill {skill_id} {'enabled' if sub == 'on' else 'disabled'}")
            return True

        if sub == "info":
            if len(parts) < 2:
                self._write_error("/skill info requires a skill id")
                return True
            skill = self.agent.skill_runtime.get_skill(parts[1])
            if not skill:
                self._write_error(f"Skill not found: {parts[1]}")
                return True
            self._write_section_heading(skill.name)
            self._write(f"  [dim]{esc(skill.description)}[/dim]")
            keywords = ", ".join(skill.triggers.get("keywords", [])) or "none"
            file_ext = ", ".join(skill.triggers.get("file_ext", [])) or "none"
            tools = ", ".join(skill.allowed_tools) or "all"
            enabled = "on" if skill.enabled else "off"
            color = "green" if skill.enabled else "red"
            self._write_detail_line("id", skill.id)
            self._write_detail_line("version", skill.version)
            self._write_detail_line("status", f"[{color}]{enabled}[/{color}]", value_markup=True)
            self._write_detail_line("keywords", keywords)
            self._write_detail_line("file_ext", file_ext)
            self._write_detail_line("tools", tools)
            return True

        self._write_error("Usage: /skill on|off|reload|info <id>")
        return True

    def _cmd_memory(self, arg: str) -> bool:
        sub = arg.strip().lower()
        if sub == "stats":
            stats = self.agent.skill_runtime.memory.stats()
            self._write_section_heading("Memory Stats")
            self._write_detail_line("count", str(stats["count"]))
            self._write_detail_line("model", str(stats["model_name"]))
            self._write_detail_line("dimension", str(stats["dimension"]))
            self._write_detail_line("by_type", json.dumps(stats["by_type"]))
            return True
        self._write_error("Usage: /memory stats")
        return True

    def _cmd_workspace(self, arg: str) -> bool:
        sub = arg.strip().lower()
        if sub == "tree":
            tree = self.agent.skill_runtime.workspace.workspace_tree()
            self._write_section_heading("Workspace Tree")
            for line in tree.splitlines():
                self._write(f"[dim]  {esc(line)}[/dim]")
            return True
        self._write_error("Usage: /workspace tree")
        return True

    def _cmd_code(self, arg: str) -> bool:
        target = arg.strip().lower() or "last"
        if not self._code_blocks:
            self._write_error("No code blocks available yet")
            return True
        if target == "last":
            self._open_code_block(len(self._code_blocks))
            return True
        try:
            index = int(target)
        except ValueError:
            self._write_error("Usage: /code [n|last]")
            return True
        self._open_code_block(index)
        return True
