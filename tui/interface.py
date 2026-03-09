from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from rich.markup import escape as esc
from rich.padding import Padding
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widgets import Input, RichLog, Static

from agent.core import Agent, AgentTurnResult
from core.attachments import build_content, classify_attachment
from core.conv_tree import ConvTree, Turn

DEFAULT_SAVE = "llamachat_tree.json"
STREAMED_FILE_TOOLS = {"create_file", "edit_file"}
MAX_LIVE_PREVIEW_CHARS = 12000
MAX_LIVE_PREVIEW_LINES = 180
MAX_REPLY_ACC_CHARS = 24000

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
            ("/save [file]", f"Save tree JSON (default {DEFAULT_SAVE})"),
            ("/load [file]", f"Load tree JSON (default {DEFAULT_SAVE})"),
        ],
    ),
]


def _hanging_indent(line: str) -> int:
    stripped = line.lstrip(" ")
    lead = len(line) - len(stripped)
    if stripped.startswith(("- ", "* ", "+ ")):
        return lead + 2
    if ". " in stripped[:5] and stripped[0].isdigit():
        idx = stripped.find(". ")
        if idx > 0:
            return lead + idx + 2
    return lead


def _render_md(line: str, in_fence: bool) -> Tuple[str, bool]:
    stripped = line.strip()
    if stripped.startswith("```") or stripped.startswith("~~~"):
        return f"[dim]{esc(line)}[/dim]", not in_fence
    if in_fence:
        return esc(line), in_fence

    out = []
    i = 0
    n = len(line)
    while i < n:
        if line[i] == "`":
            j = i + 1
            while j < n and line[j] != "`":
                j += 1
            if j < n:
                out.append(f"[bold yellow]{esc(line[i:j+1])}[/bold yellow]")
                i = j + 1
            else:
                out.append(esc(line[i:]))
                break
            continue
        if i + 1 < n and line[i : i + 2] == "**":
            j = line.find("**", i + 2)
            if j != -1:
                out.append(f"[bold]{esc(line[i+2:j])}[/bold]")
                i = j + 2
                continue
        out.append(esc(line[i]))
        i += 1
    return "".join(out), in_fence


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

        self.conv_tree = ConvTree()
        self.pending: List[Tuple[str, str]] = []

        self._stop_event = threading.Event()
        self._active_turn_id: Optional[str] = None
        self._reply_acc = ""
        self._tool_activity_seen = False
        self._live_tool_streams: Dict[str, Dict[str, Any]] = {}

        self._reasoning_open = False
        self._content_open = False
        self._done_thinking_rendered = False
        self._buf_r = ""
        self._buf_c = ""
        self._in_fence = False

        self._last_scroll = 0.0
        self._scroll_interval = 0.05
        self._last_status2 = ""

        self._esc_pending = False
        self._esc_ts = 0.0
        self._spin_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spin_i = 0

        self._await_shell_confirm = False
        self._shell_confirm_command = ""
        self._shell_confirm_event: Optional[threading.Event] = None
        self._shell_confirm_result: Optional[Dict[str, bool]] = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-area"):
            with ScrollableContainer(id="chat-scroll"):
                yield RichLog(id="chat-log", markup=True, highlight=False, wrap=True)
                yield Static("", id="partial", markup=True)
            with ScrollableContainer(id="sidebar"):
                yield Static("", id="sidebar-content", markup=True)

        with Vertical(id="footer"):
            yield Static("", id="footer-sep")
            yield Static("", id="status1")
            yield Static("", id="status2")
            with Horizontal(id="input-row"):
                yield ChatInput(id="chat-input", placeholder="Type a message…")

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

    def on_key(self, event) -> None:
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
        if self.streaming and not self._esc_pending:
            self._spin_i += 1
            self._update_status2()

    def watch_streaming(self, value: bool) -> None:
        self.query_one(ChatInput).disabled = value
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

    def _write(self, markup: str) -> None:
        self._log().write(Text.from_markup(markup))
        self._scroll().scroll_end(animate=False)

    def _write_indented(self, markup: str, indent: int = 2) -> None:
        self._log().write(Padding(Text.from_markup(markup), pad=(0, 0, 0, indent)))
        self._scroll().scroll_end(animate=False)

    def _write_info(self, text: str) -> None:
        self._write(f"[dim]  {esc(text)}[/dim]")

    def _write_error(self, text: str) -> None:
        self._write(f"[bold red]  ✖ {esc(text)}[/bold red]")

    def _append_reply_token(self, token: str) -> None:
        if not token:
            return
        if len(self._reply_acc) >= MAX_REPLY_ACC_CHARS:
            return
        remaining = MAX_REPLY_ACC_CHARS - len(self._reply_acc)
        self._reply_acc += token[:remaining]

    def _compact_tool_args(self, tool_name: str, args: Any) -> str:
        if not isinstance(args, dict):
            return str(args)

        if tool_name in STREAMED_FILE_TOOLS:
            filepath = str(args.get("filepath", ""))
            content = args.get("content", "")
            n_chars = len(content) if isinstance(content, str) else 0
            return f'filepath="{filepath}", content={n_chars} chars'

        parts = []
        for key in sorted(args.keys()):
            value = args[key]
            text = str(value)
            if len(text) > 60:
                text = text[:57] + "..."
            parts.append(f"{key}={text}")
        return ", ".join(parts)

    def _is_tool_trace_line(self, line: str) -> bool:
        s = line.strip().lower()
        return "tool call:" in s

    def _extract_partial_json_string_field(self, raw: str, key: str) -> Tuple[Optional[str], bool]:
        marker = f'"{key}"'
        start = raw.find(marker)
        if start < 0:
            return None, False
        colon = raw.find(":", start + len(marker))
        if colon < 0:
            return None, False
        i = colon + 1
        while i < len(raw) and raw[i].isspace():
            i += 1
        if i >= len(raw) or raw[i] != '"':
            return None, False
        i += 1
        out: List[str] = []
        escaped = False
        while i < len(raw):
            ch = raw[i]
            if escaped:
                if ch == "n":
                    out.append("\n")
                elif ch == "r":
                    out.append("\r")
                elif ch == "t":
                    out.append("\t")
                elif ch in {'"', "\\", "/"}:
                    out.append(ch)
                elif ch == "b":
                    out.append("\b")
                elif ch == "f":
                    out.append("\f")
                elif ch == "u":
                    if i + 4 < len(raw):
                        hex_part = raw[i + 1 : i + 5]
                        try:
                            out.append(chr(int(hex_part, 16)))
                            i += 4
                        except ValueError:
                            out.append("u")
                    else:
                        return "".join(out), False
                else:
                    out.append(ch)
                escaped = False
            else:
                if ch == "\\":
                    escaped = True
                elif ch == '"':
                    return "".join(out), True
                else:
                    out.append(ch)
            i += 1
        return "".join(out), False

    def _update_live_tool_preview(self, stream_id: str, name: str, raw_arguments: str) -> None:
        if name not in STREAMED_FILE_TOOLS:
            return
        if not raw_arguments:
            return

        state = self._live_tool_streams.setdefault(
            stream_id,
            {
                "name": name,
                "filepath": "",
                "opened": False,
                "closed": False,
                "printed_len": 0,
                "line_buf": "",
                "rendered_chars": 0,
                "rendered_lines": 0,
                "truncated": False,
            },
        )
        if state.get("name") != name:
            state["name"] = name
        filepath, _ = self._extract_partial_json_string_field(raw_arguments, "filepath")
        if filepath:
            state["filepath"] = filepath

        content, complete = self._extract_partial_json_string_field(raw_arguments, "content")
        if content is None:
            return

        if not state["opened"]:
            state["opened"] = True
            path_label = state.get("filepath") or "(pending filepath)"
            self._write(f"[dim]  · file draft: {esc(path_label)}[/dim]")
            self._write_indented("[dim]```[/dim]", indent=2)

        prev_len = int(state.get("printed_len", 0))
        if len(content) < prev_len:
            prev_len = 0
            state["line_buf"] = ""
        delta = content[prev_len:]
        state["printed_len"] = len(content)

        if delta and not state.get("truncated"):
            state["line_buf"] = str(state.get("line_buf", "")) + delta
            while "\n" in state["line_buf"]:
                line, state["line_buf"] = state["line_buf"].split("\n", 1)
                rendered_lines = int(state.get("rendered_lines", 0))
                rendered_chars = int(state.get("rendered_chars", 0))
                if rendered_lines >= MAX_LIVE_PREVIEW_LINES or rendered_chars >= MAX_LIVE_PREVIEW_CHARS:
                    state["truncated"] = True
                    state["line_buf"] = ""
                    break
                remaining_chars = MAX_LIVE_PREVIEW_CHARS - rendered_chars
                out_line = line[:remaining_chars]
                self._write_indented(esc(out_line), indent=2)
                state["rendered_chars"] = rendered_chars + len(out_line) + 1
                state["rendered_lines"] = rendered_lines + 1
                if len(line) > remaining_chars:
                    state["truncated"] = True
                    state["line_buf"] = ""
                    break

        if complete and not state["closed"]:
            tail = str(state.get("line_buf", ""))
            if tail and not state.get("truncated"):
                remaining_chars = MAX_LIVE_PREVIEW_CHARS - int(state.get("rendered_chars", 0))
                if remaining_chars > 0:
                    self._write_indented(esc(tail[:remaining_chars]), indent=2)
                    state["rendered_chars"] = int(state.get("rendered_chars", 0)) + min(len(tail), remaining_chars)
                if len(tail) > remaining_chars:
                    state["truncated"] = True
                state["line_buf"] = ""
            if state.get("truncated"):
                self._write_indented("[dim]... (live preview truncated) ...[/dim]", indent=2)
            self._write_indented("[dim]```[/dim]", indent=2)
            state["closed"] = True

    def _close_live_tool_preview(self, stream_id: str) -> bool:
        state = self._live_tool_streams.get(stream_id)
        if not state or not state.get("opened"):
            return False
        if not state.get("closed"):
            tail = str(state.get("line_buf", ""))
            if tail and not state.get("truncated"):
                remaining_chars = MAX_LIVE_PREVIEW_CHARS - int(state.get("rendered_chars", 0))
                if remaining_chars > 0:
                    self._write_indented(esc(tail[:remaining_chars]), indent=2)
                if len(tail) > remaining_chars:
                    state["truncated"] = True
                state["line_buf"] = ""
            if state.get("truncated"):
                self._write_indented("[dim]... (live preview truncated) ...[/dim]", indent=2)
            self._write_indented("[dim]```[/dim]", indent=2)
            state["closed"] = True
        self._live_tool_streams.pop(stream_id, None)
        return True

    def _close_all_live_tool_previews(self) -> None:
        for stream_id in list(self._live_tool_streams.keys()):
            self._close_live_tool_preview(stream_id)

    def _write_tool_content_preview(self, tool_name: str, args: Any) -> None:
        if tool_name not in STREAMED_FILE_TOOLS:
            return
        if not isinstance(args, dict):
            return
        content = args.get("content")
        filepath = str(args.get("filepath", ""))
        if not isinstance(content, str) or not content.strip():
            return

        max_chars = 8000
        max_lines = 140
        clipped = content[:max_chars]
        lines = clipped.splitlines()
        truncated = len(content) > max_chars or len(lines) > max_lines
        lines = lines[:max_lines]

        self._write(f"[dim]  · file draft: {esc(filepath)}[/dim]")
        self._write_indented("[dim]```[/dim]", indent=2)
        for line in lines:
            self._write_indented(esc(line), indent=2)
        if truncated:
            self._write_indented("[dim]... (preview truncated) ...[/dim]", indent=2)
        self._write_indented("[dim]```[/dim]", indent=2)

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
            left = f"[bold yellow]approve shell command?[/bold yellow] [dim]y=yes n=no[/dim]"
        elif self.streaming:
            frame = self._spin_frames[self._spin_i % len(self._spin_frames)]
            if self._esc_pending:
                left = f"[dim]{frame}[/dim] [dim]generating[/dim] [bold red]esc again to stop[/bold red]"
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
                        f"[dim]  · tool call: {esc(name)}({esc(self._compact_tool_args(name, args))})[/dim]"
                    )
                    self._write_tool_content_preview(name, args)
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
        for line in display.splitlines() or [""]:
            rendered, in_fence = _render_md(line, in_fence)
            indent = 2 if in_fence else max(2, _hanging_indent(line))
            self._write_indented(rendered, indent=indent)

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
        self._scroll().scroll_end(animate=False)

    def _start_stream(self, turn: Turn, user_input: str, attachment_paths: List[str]) -> None:
        self.streaming = True
        self._active_turn_id = turn.id
        self._reply_acc = ""
        self._tool_activity_seen = False
        self._live_tool_streams = {}
        self._reasoning_open = False
        self._content_open = False
        self._done_thinking_rendered = False
        self._buf_r = ""
        self._buf_c = ""
        self._in_fence = False
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
            rendered, _ = _render_md(line, False)
            self._write_indented(f"[dim]{rendered}[/dim]", indent=4)

    def _flush_content_buffer(self, include_partial: bool = False) -> None:
        while "\n" in self._buf_c:
            line, self._buf_c = self._buf_c.split("\n", 1)
            if self._is_tool_trace_line(line):
                continue
            rendered, self._in_fence = _render_md(line, self._in_fence)
            indent = 2 if self._in_fence else max(2, _hanging_indent(line))
            self._write_indented(rendered, indent=indent)

        if include_partial and self._buf_c:
            if not self._is_tool_trace_line(self._buf_c):
                rendered, self._in_fence = _render_md(self._buf_c, self._in_fence)
                indent = 2 if self._in_fence else max(2, _hanging_indent(self._buf_c))
                self._write_indented(rendered, indent=indent)
            self._buf_c = ""
            self._partial().update("")
            return

        self._partial().update(f"  {esc(self._buf_c)}" if self._buf_c else "")

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
                self._update_live_tool_preview(stream_id, name, raw_arguments)

        elif etype == "tool_call":
            self._tool_activity_seen = True
            self._flush_reasoning_buffer()
            name = event.get("name", "tool")
            args = event.get("arguments", {})
            stream_id = str(event.get("stream_id") or "")
            self._write(
                f"[dim]  · tool call: {esc(name)}({esc(self._compact_tool_args(name, args))})[/dim]"
            )
            streamed = self._close_live_tool_preview(stream_id) if stream_id else False
            if not streamed:
                self._write_tool_content_preview(name, args)

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
            self._scroll().scroll_end(animate=False)
            self._last_scroll = now

    def _on_stream_end(self, turn_id: str, result: AgentTurnResult) -> None:
        partial = self._partial()
        partial.update("")
        partial.display = False

        self._close_all_live_tool_previews()

        if self._buf_r and not self._content_open:
            if not self._is_tool_trace_line(self._buf_r):
                rendered, _ = _render_md(self._buf_r, False)
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
            for line in reply.splitlines() or [""]:
                rendered, in_fence = _render_md(line, in_fence)
                indent = 2 if in_fence else max(2, _hanging_indent(line))
                self._write_indented(rendered, indent=indent)

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
        self._scroll().scroll_end(animate=False)
        self.streaming = False
        self._live_tool_streams = {}
        self._active_turn_id = None
        self._esc_pending = False
        self._update_status1()
        self._update_status2()
        self._update_sidebar()

    def _confirm_shell_command(self, command: str) -> bool:
        event = threading.Event()
        holder = {"value": False}
        self.call_from_thread(self._begin_shell_confirm, command, event, holder)
        event.wait(timeout=60)
        return bool(holder.get("value", False))

    def _begin_shell_confirm(self, command: str, event: threading.Event, holder: Dict[str, bool]) -> None:
        self._await_shell_confirm = True
        self._shell_confirm_command = command
        self._shell_confirm_event = event
        self._shell_confirm_result = holder
        self._write(f"[yellow]  ? Run shell command: {esc(command)}[/yellow]")
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
        self.query_one(ChatInput).value = ""
        if not text:
            return
        if not self._handle_command(text):
            if not self.streaming:
                self._send(text)

    def action_handle_esc(self) -> None:
        if self._await_shell_confirm:
            self._finish_shell_confirm(False)
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

    def action_scroll_up(self) -> None:
        self._scroll().scroll_page_up()

    def action_scroll_down(self) -> None:
        self._scroll().scroll_page_down()

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
                self._write("")
                self._write("[bold cyan]  Children[/bold cyan]")
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
                self.conv_tree = ConvTree.load(path)
                self._rebuild_viewport()
                self._update_sidebar()
                self._update_status1()
                self._write(f"[yellow]  ✓ Loaded tree from {esc(path)}[/yellow]")
            except Exception as exc:
                self._write_error(f"Load failed: {exc}")
            return True

        if cmd == "/clear":
            self.conv_tree = ConvTree()
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
        self._write("")
        self._write("[bold cyan]  Skills[/bold cyan]")
        for skill in self.agent.skill_runtime.list_skills():
            state = "on" if skill.enabled else "off"
            color = "green" if skill.enabled else "red"
            self._write(
                f"  [bold]{esc(skill.id)}[/bold] [dim]({esc(skill.version)})[/dim] "
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
            self._write("")
            self._write(f"[bold]{esc(skill.name)}[/bold] [dim]({esc(skill.id)})[/dim]")
            self._write(f"[dim]{esc(skill.description)}[/dim]")
            self._write(f"[dim]enabled={skill.enabled}[/dim]")
            keywords = ", ".join(skill.triggers.get("keywords", [])) or "none"
            file_ext = ", ".join(skill.triggers.get("file_ext", [])) or "none"
            tools = ", ".join(skill.allowed_tools) or "all"
            self._write(f"[dim]keywords: {esc(keywords)}[/dim]")
            self._write(f"[dim]file_ext: {esc(file_ext)}[/dim]")
            self._write(f"[dim]tools: {esc(tools)}[/dim]")
            return True

        self._write_error("Usage: /skill on|off|reload|info <id>")
        return True

    def _cmd_memory(self, arg: str) -> bool:
        sub = arg.strip().lower()
        if sub == "stats":
            stats = self.agent.skill_runtime.memory.stats()
            self._write("")
            self._write("[bold cyan]  Memory Stats[/bold cyan]")
            self._write(f"  [dim]count: {stats['count']}[/dim]")
            self._write(f"  [dim]model: {esc(str(stats['model_name']))}[/dim]")
            self._write(f"  [dim]dimension: {stats['dimension']}[/dim]")
            self._write(f"  [dim]by_type: {esc(json.dumps(stats['by_type']))}[/dim]")
            return True
        self._write_error("Usage: /memory stats")
        return True

    def _cmd_workspace(self, arg: str) -> bool:
        sub = arg.strip().lower()
        if sub == "tree":
            tree = self.agent.skill_runtime.workspace.workspace_tree()
            self._write("")
            self._write("[bold cyan]  Workspace Tree[/bold cyan]")
            for line in tree.splitlines():
                self._write(f"[dim]  {esc(line)}[/dim]")
            return True
        self._write_error("Usage: /workspace tree")
        return True
