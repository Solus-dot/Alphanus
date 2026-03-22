from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich import box
from rich.markup import escape as esc
from rich.padding import Padding
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.geometry import Offset
from textual.reactive import reactive
from textual.widgets import Button, Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from agent.core import Agent, AgentTurnResult
from core.attachments import build_content, classify_attachment
from core.configuration import (
    config_for_editor_view,
    load_or_create_global_config,
    normalize_config,
    validate_endpoint_policy,
)
from core.conv_tree import ConvTree, Turn
from core.sessions import ChatSession, SessionStore, SessionSummary
from tui.commands import (
    HELP_SECTIONS,
    CommandEntry,
    command_entries_for_query,
    command_label,
    exact_command_inputs,
)
from tui.live_tool_preview import LiveToolPreviewManager
from tui.markdown_utils import fence_language, hanging_indent, render_md
from tui.popups import (
    CodeViewerModal,
    ConfigEditorModal,
    PickerItem,
    SelectionPickerModal,
    SessionPickerModal,
    export_picker_items,
    session_picker_items,
)
from tui.sidebar import render_sidebar_inspector_markup, render_sidebar_tree_markup
from tui.status import status_left_markup, status_right_markup, topbar_center, topbar_left, topbar_right

MAX_REPLY_ACC_CHARS = 24000
SHELL_CONFIRM_TIMEOUT_S = 60
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_CONFIG_PATH = PROJECT_ROOT / "config" / "global_config.json"
ACCENT_COLOR = "#6366f1"


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
        background: #09090b;
        color: #e4e4e7;
    }

    #topbar {
        height: 3;
        layout: horizontal;
        background: #121214;
        border-bottom: solid #27272a;
        padding: 0 2;
    }

    #topbar-left {
        width: 1fr;
        height: 3;
        content-align: left middle;
    }

    #topbar-center {
        width: auto;
        min-width: 40;
        height: 3;
        content-align: center middle;
    }

    #topbar-right {
        width: auto;
        height: 3;
        content-align: right middle;
    }

    #main-area {
        height: 1fr;
        layout: horizontal;
        background: #09090b;
    }

    #chat-scroll {
        width: 1fr;
        height: 1fr;
        background: #09090b;
        overflow-x: hidden;
        scrollbar-size: 1 1;
        scrollbar-color: #3f3f46 #121214;
        scrollbar-background: #121214;
        scrollbar-background-hover: #18181b;
        scrollbar-background-active: #18181b;
        scrollbar-color-hover: #52525b;
        scrollbar-color-active: #6366f1;
        scrollbar-corner-color: #121214;
    }

    #chat-scroll.-active-panel {
        border: round #6366f1;
    }

    #chat-log {
        width: 1fr;
        height: auto;
        background: #09090b;
        padding: 0 3;
        overflow-x: hidden;
        scrollbar-size: 0 0;
    }

    #partial {
        width: 1fr;
        height: auto;
        background: #09090b;
        display: none;
        padding: 0 3;
        overflow-x: hidden;
    }

    #sidebar {
        width: 38;
        border-left: solid #27272a;
        background: #121214;
        display: none;
        padding: 0;
        layout: vertical;
    }

    #sidebar-tree-section {
        width: 1fr;
        height: 1fr;
        layout: vertical;
        min-height: 5;
    }

    #sidebar-tree-header,
    #sidebar-inspector-header {
        width: 1fr;
        height: auto;
        color: #a1a1aa;
        text-style: bold;
        padding: 1 2 0 2;
    }

    #sidebar-tree-meta {
        width: 1fr;
        height: auto;
        color: #71717a;
        padding: 0 2 1 2;
    }

    #sidebar-tree-scroll {
        width: 1fr;
        height: 1fr;
        background: #121214;
        padding: 0 2 1 2;
        scrollbar-background: #121214;
        scrollbar-background-hover: #18181b;
        scrollbar-background-active: #18181b;
        scrollbar-color: #3f3f46;
        scrollbar-color-hover: #52525b;
        scrollbar-color-active: #6366f1;
        scrollbar-corner-color: #121214;
    }

    #sidebar.-active-panel {
        border-left: solid #6366f1;
    }

    #sidebar-tree-content {
        width: 1fr;
        height: auto;
        background: #121214;
    }

    #sidebar-inspector-section {
        width: 1fr;
        height: auto;
        border-top: solid #27272a;
        background: #121214;
        layout: vertical;
    }

    #sidebar-inspector-scroll {
        width: 1fr;
        height: auto;
        max-height: 12;
        background: #121214;
        padding: 0 2 1 2;
        scrollbar-background: #121214;
        scrollbar-background-hover: #18181b;
        scrollbar-background-active: #18181b;
        scrollbar-color: #3f3f46;
        scrollbar-color-hover: #52525b;
        scrollbar-color-active: #6366f1;
        scrollbar-corner-color: #121214;
    }

    #sidebar-inspector-content {
        width: 1fr;
        height: auto;
        background: #121214;
    }

    #footer {
        height: auto;
        background: #09090b;
        layout: vertical;
        dock: bottom;
    }

    #command-popup {
        width: 64;
        max-height: 12;
        background: #121214;
        border: round #27272a;
        display: none;
        overlay: screen;
        padding: 0 1;
    }

    #command-popup-title {
        height: auto;
        color: #6366f1;
        padding: 1 1 0 1;
        text-style: bold;
    }

    #command-popup-hint {
        height: auto;
        color: #a1a1aa;
        padding: 0 1 1 1;
    }

    #command-options {
        width: 1fr;
        height: auto;
        max-height: 8;
        background: #121214;
        border: none;
        padding: 0 1 1 1;
        scrollbar-background: #121214;
        scrollbar-background-hover: #18181b;
        scrollbar-background-active: #18181b;
        scrollbar-color: #3f3f46;
        scrollbar-color-hover: #52525b;
        scrollbar-color-active: #6366f1;
        scrollbar-corner-color: #121214;
    }

    #command-options > .option-list--option-highlighted {
        color: #e4e4e7;
        background: #18181b;
        text-style: none;
    }

    #command-options:focus > .option-list--option-highlighted {
        color: #ffffff;
        background: #1a1730;
        text-style: bold;
    }

    #footer-sep {
        height: 1;
        background: #27272a;
    }

    #status-bar {
        height: 1;
        layout: horizontal;
        padding: 0 0;
        background: #09090b;
    }

    #status-left {
        width: 1fr;
        height: 1;
        content-align: left middle;
    }

    #status-right {
        width: auto;
        height: 1;
        content-align: right middle;
    }

    #input-row {
        height: auto;
        layout: vertical;
        background: #09090b;
        padding: 0 0 1 0;
        min-height: 3;
    }

    #composer-shell {
        width: 1fr;
        height: 3;
        layout: horizontal;
        background: #18181b;
        border: round #3f3f46;
        padding: 0 1;
        align: left middle;
    }

    ChatInput {
        width: 1fr;
        height: 3;
        border: none;
        background: transparent;
        color: #e4e4e7;
    }

    ChatInput:focus {
        border: none;
        background: transparent;
    }

    #input-row.-active-panel #composer-shell {
        border: round #6366f1;
    }

    #input-accessories {
        width: auto;
        height: 1;
        layout: horizontal;
        align: right middle;
        padding-left: 1;
    }

    #pending-attachments {
        width: auto;
        max-width: 44;
        height: 1;
        content-align: right middle;
        padding-right: 1;
    }

    #attach-file {
        width: auto;
        min-width: 8;
        height: 1;
        background: #1a1730;
        color: #6366f1;
        border: none;
        text-style: bold;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", show=False),
        Binding("ctrl+d", "quit", show=False),
        Binding("escape", "handle_esc", show=False),
        Binding("pageup", "scroll_up", show=False),
        Binding("pagedown", "scroll_down", show=False),
        Binding("tab", "focus_next_panel", show=False),
        Binding("shift+tab", "focus_prev_panel", show=False),
        Binding("ctrl+h", "focus_chat", show=False),
        Binding("ctrl+l", "focus_tree", show=False),
        Binding("question_mark", "show_keymap", show=False),
        Binding("enter", "tree_open", show=False),
        Binding("g", "tree_top", show=False),
        Binding("shift+g", "tree_bottom", show=False),
        Binding("[", "tree_prev_sibling", show=False),
        Binding("]", "tree_next_sibling", show=False),
        Binding("j", "tree_down", show=False),
        Binding("k", "tree_up", show=False),
        Binding("o", "tree_open", show=False),
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

        self._session_store = SessionStore(self.agent.skill_runtime.workspace.workspace_root)
        self._session_id = ""
        self._session_title = ""
        self._session_created_at = ""
        self.conv_tree = self._new_conv_tree()
        self._activate_session_state(self._session_store.bootstrap())
        self.pending: List[Tuple[str, str]] = []

        self._stop_event = threading.Event()
        self._active_turn_id: Optional[str] = None
        self._reply_acc = ""
        self._live_preview = LiveToolPreviewManager()

        self._reasoning_open = False
        self._content_open = False
        self._buf_r = ""
        self._buf_c = ""
        self._in_fence = False
        self._fence_lang: Optional[str] = None
        self._fence_lines: List[str] = []

        self._last_scroll = 0.0
        self._scroll_interval = 0.05
        self._last_status_left = ""
        self._last_status_right = ""
        self._auto_follow_stream = True
        self._model_name: Optional[str] = None
        self._model_refresh_inflight = False
        self._last_model_refresh = 0.0
        self._model_refresh_interval_s = 5.0

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
        self._code_blocks: List[Tuple[str, Optional[str]]] = []
        self._show_tool_details = True
        self._show_historical_tool_details = False
        self._pending_tool_details: List[Tuple[str, str]] = []
        self._focused_panel = "input"
        self._tree_cursor_id = "root"
        self._last_log_was_blank = False
        self._last_model_context_tokens: Optional[int] = None
        self._startup_session_prompt_opened = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Static("", id="topbar-left")
            yield Static("", id="topbar-center")
            yield Static("", id="topbar-right")
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
            with Vertical(id="sidebar"):
                with Vertical(id="sidebar-tree-section"):
                    yield Static("Conversation Tree", id="sidebar-tree-header")
                    yield Static("0 turns", id="sidebar-tree-meta")
                    with ScrollableContainer(id="sidebar-tree-scroll"):
                        yield Static("", id="sidebar-tree-content", markup=True)
                with Vertical(id="sidebar-inspector-section"):
                    yield Static("Inspector", id="sidebar-inspector-header")
                    with ScrollableContainer(id="sidebar-inspector-scroll"):
                        yield Static("", id="sidebar-inspector-content", markup=True)

        with Vertical(id="footer"):
            yield Static("", id="footer-sep")
            with Vertical(id="command-popup"):
                yield Static("commands", id="command-popup-title")
                yield Static("type to filter · tab to insert", id="command-popup-hint")
                yield OptionList(id="command-options")
            with Horizontal(id="input-row"):
                with Horizontal(id="composer-shell"):
                    yield ChatInput(id="chat-input", placeholder="Type a message…")
                    with Horizontal(id="input-accessories"):
                        yield Static("", id="pending-attachments", markup=True)
                        yield Button("+ File", id="attach-file")
            with Horizontal(id="status-bar"):
                yield Static("", id="status-left")
                yield Static("", id="status-right")

    def on_mount(self) -> None:
        self.thinking = bool(self.agent.config.get("agent", {}).get("enable_thinking", True))
        self.set_interval(0.1, self._tick)
        self._sync_tree_cursor()
        self._apply_focus_classes()
        self._update_topbar()
        self._update_status1()
        self._update_status2()
        self._update_sidebar()
        self._update_pending_attachments()
        self._maybe_refresh_model_name(force=True)
        self.query_one(ChatInput).focus()
        self.call_after_refresh(self._open_startup_session_picker)

    def on_resize(self, event) -> None:
        sidebar = self.query_one("#sidebar", Vertical)
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

    def _activate_session_state(self, session: ChatSession) -> None:
        self._session_id = session.id
        self._session_title = session.title
        self._session_created_at = session.created_at
        tree = self._apply_tree_compaction_policy(session.tree)
        if tree.current_id == "root" and tree.nodes["root"].children and not tree._pending_branch:
            for node_id in reversed(list(tree.nodes.keys())):
                if node_id != "root" and not tree.nodes[node_id].children:
                    tree.current_id = node_id
                    break
        self.conv_tree = tree
        self._tree_cursor_id = self.conv_tree.current_id

    def _save_active_session(self, rename_to: Optional[str] = None) -> ChatSession:
        title = (rename_to or self._session_title or "").strip() or self._session_title or "Untitled Session"
        session = self._session_store.save_tree(
            self._session_id,
            title,
            self.conv_tree,
            created_at=self._session_created_at,
            activate=True,
        )
        self._session_title = session.title
        self._session_created_at = session.created_at
        return session

    def _session_timestamp_label(self, value: str) -> str:
        text = str(value or "").replace("T", " ").replace("+00:00", "Z")
        return text[:16] if len(text) >= 16 else text

    def _session_row_label(self, summary: SessionSummary) -> str:
        status = (
            f"[bold {ACCENT_COLOR}]active[/bold {ACCENT_COLOR}]"
            if summary.is_active
            else "[#71717a]saved[/#71717a]"
        )
        title = (
            f"[bold {ACCENT_COLOR}]{esc(summary.title)}[/bold {ACCENT_COLOR}]"
            if summary.is_active
            else f"[#f4f4f5]{esc(summary.title)}[/#f4f4f5]"
        )
        turns = "turn" if summary.turn_count == 1 else "turns"
        branches = "branch" if summary.branch_count == 1 else "branches"
        return (
            f"{status}  {title} [#a1a1aa][{esc(summary.id)}][/#a1a1aa]  "
            f"[#f4f4f5]{summary.turn_count} {turns}[/#f4f4f5]  "
            f"[#f4f4f5]{summary.branch_count} {branches}[/#f4f4f5]  "
            f"[#a1a1aa]{esc(self._session_timestamp_label(summary.updated_at))}[/#a1a1aa]"
        )

    def _cmd_sessions(self) -> None:
        sessions = self._session_store.list_sessions()
        self._write_section_heading("Sessions")
        if not sessions:
            self._write_info("No saved sessions yet.")
            return
        self._write_indexed_dim_lines(
            [self._session_row_label(summary) for summary in sessions],
            color="#a1a1aa",
            allow_markup=True,
        )

    def _open_load_session_picker(self) -> None:
        sessions = self._session_store.list_sessions()
        self.push_screen(
            SelectionPickerModal(
                title="Load Session",
                subtitle="Choose a saved session to open.",
                confirm_label="Load Session",
                empty_text="No saved sessions available.",
                items=session_picker_items(sessions),
            ),
            self._on_load_session_picker_close,
        )

    def _on_load_session_picker_close(self, result: Optional[Dict[str, str]]) -> None:
        session_id = str((result or {}).get("id") or "").strip()
        if not session_id:
            return
        try:
            self._save_active_session()
            loaded = self._session_store.load_session(session_id)
            self._switch_to_session(loaded)
            self._write_command_action(f"Loaded session '{loaded.title}'", icon="✓")
        except Exception as exc:
            self._write_error(f"Load failed: {exc}")

    def _open_import_picker(self) -> None:
        exports = self._session_store.list_exports()
        self.push_screen(
            SelectionPickerModal(
                title="Import Export",
                subtitle="Choose a stored export to import as a new session.",
                confirm_label="Import Export",
                empty_text="No exports found in .alphanus/exports.",
                items=export_picker_items(exports),
            ),
            self._on_import_picker_close,
        )

    def _on_import_picker_close(self, result: Optional[Dict[str, str]]) -> None:
        export_id = str((result or {}).get("id") or "").strip()
        if not export_id:
            return
        try:
            self._save_active_session()
            path = self._session_store.resolve_export_path(export_id)
            imported = self._session_store.import_tree(path)
            self._switch_to_session(imported)
            self._write_command_action(f"Imported session '{imported.title}'", icon="✓")
        except Exception as exc:
            self._write_error(f"Import failed: {exc}")

    def _current_session_is_blank(self) -> bool:
        return (
            self.conv_tree.current_id == "root"
            and len(self.conv_tree.nodes) == 1
            and not self.conv_tree.current.children
            and not self.conv_tree._pending_branch
        )

    def _open_new_session(self, title: str = "") -> ChatSession:
        normalized = title.strip()
        if self._current_session_is_blank():
            return self._save_active_session(rename_to=normalized or None)
        self._save_active_session()
        return self._session_store.create_session(normalized)

    def _open_startup_session_picker(self) -> None:
        if self._startup_session_prompt_opened:
            return
        self._startup_session_prompt_opened = True
        sessions = self._session_store.list_sessions()
        self.push_screen(
            SessionPickerModal(sessions, self._session_id, self._session_title),
            self._on_startup_session_picker_close,
        )

    def _on_startup_session_picker_close(self, result: Optional[Dict[str, str]]) -> None:
        action = str((result or {}).get("action") or "continue")
        if action == "load":
            selector = str((result or {}).get("selector") or "").strip()
            if not selector:
                return
            try:
                session = self._session_store.load_session(selector)
                self._switch_to_session(session)
            except Exception as exc:
                self._write_error(f"Load failed: {exc}")
            return
        if action == "new":
            session = self._open_new_session(str((result or {}).get("title") or ""))
            self._switch_to_session(session)

    def _switch_to_session(self, session: ChatSession, *, clear_pending: bool = True) -> None:
        self._activate_session_state(session)
        if clear_pending:
            self.pending.clear()
        self._rebuild_viewport()
        self._update_sidebar()
        self._update_pending_attachments()
        self._update_status1()
        self._update_status2()
        self._update_input_placeholder()

    def _tree_rows(self) -> List[Tuple[str, str, bool]]:
        return self.conv_tree.render_tree(width=30)

    def _sync_tree_cursor(self) -> None:
        if self._tree_cursor_id in self.conv_tree.nodes:
            return
        self._tree_cursor_id = self.conv_tree.current_id

    def _current_branch_name(self) -> str:
        if self.conv_tree._pending_branch and self.conv_tree._pending_branch_label:
            return self.conv_tree._pending_branch_label
        for turn in reversed(self.conv_tree.active_path):
            if turn.branch_root:
                return turn.label or "branch"
        return "root"

    def _context_tokens(self) -> Optional[int]:
        return self._last_model_context_tokens

    def _memory_mode_label(self) -> str:
        try:
            return str(self.agent.skill_runtime.memory.embedding_backend)
        except Exception:
            return "unknown"

    def _apply_focus_classes(self) -> None:
        chat = self.query_one("#chat-scroll", ScrollableContainer)
        sidebar = self.query_one("#sidebar", Vertical)
        input_row = self.query_one("#input-row", Horizontal)
        chat.remove_class("-active-panel")
        sidebar.remove_class("-active-panel")
        input_row.remove_class("-active-panel")
        if self._focused_panel == "chat":
            chat.add_class("-active-panel")
        elif self._focused_panel == "tree":
            sidebar.add_class("-active-panel")
        else:
            input_row.add_class("-active-panel")

    def _set_focused_panel(self, panel: str) -> None:
        if panel == "tree" and not self.query_one("#sidebar", Vertical).display:
            panel = "chat"
        self._focused_panel = panel
        if panel == "input":
            self.query_one(ChatInput).focus()
        self._apply_focus_classes()
        self._update_topbar()

    def action_focus_next_panel(self) -> None:
        order = ["chat", "tree", "input"]
        if not self.query_one("#sidebar", Vertical).display:
            order = ["chat", "input"]
        current = order.index(self._focused_panel) if self._focused_panel in order else 0
        self._set_focused_panel(order[(current + 1) % len(order)])

    def action_focus_prev_panel(self) -> None:
        order = ["chat", "tree", "input"]
        if not self.query_one("#sidebar", Vertical).display:
            order = ["chat", "input"]
        current = order.index(self._focused_panel) if self._focused_panel in order else 0
        self._set_focused_panel(order[(current - 1) % len(order)])

    def action_focus_chat(self) -> None:
        self._set_focused_panel("chat")

    def action_focus_tree(self) -> None:
        self._set_focused_panel("tree")

    def action_tree_down(self) -> None:
        if self._focused_panel != "tree":
            return
        rows = self._tree_rows()
        ids = [tag for _text, tag, _active in rows if tag in self.conv_tree.nodes]
        if not ids:
            return
        current = ids.index(self._tree_cursor_id) if self._tree_cursor_id in ids else 0
        self._tree_cursor_id = ids[min(len(ids) - 1, current + 1)]
        self._update_sidebar()
        self._update_topbar()

    def action_tree_up(self) -> None:
        if self._focused_panel != "tree":
            return
        rows = self._tree_rows()
        ids = [tag for _text, tag, _active in rows if tag in self.conv_tree.nodes]
        if not ids:
            return
        current = ids.index(self._tree_cursor_id) if self._tree_cursor_id in ids else 0
        self._tree_cursor_id = ids[max(0, current - 1)]
        self._update_sidebar()
        self._update_topbar()

    def action_tree_top(self) -> None:
        if self._focused_panel != "tree":
            return
        ids = [tag for _text, tag, _active in self._tree_rows() if tag in self.conv_tree.nodes]
        if not ids:
            return
        self._tree_cursor_id = ids[0]
        self._update_sidebar()
        self._update_topbar()

    def action_tree_bottom(self) -> None:
        if self._focused_panel != "tree":
            return
        ids = [tag for _text, tag, _active in self._tree_rows() if tag in self.conv_tree.nodes]
        if not ids:
            return
        self._tree_cursor_id = ids[-1]
        self._update_sidebar()
        self._update_topbar()

    def _move_tree_sibling(self, direction: int) -> None:
        if self._focused_panel != "tree":
            return
        node = self.conv_tree.nodes.get(self._tree_cursor_id)
        if node is None or node.parent is None:
            return
        siblings = self.conv_tree.nodes[node.parent].children
        if self._tree_cursor_id not in siblings:
            return
        idx = siblings.index(self._tree_cursor_id) + direction
        if idx < 0 or idx >= len(siblings):
            return
        self._tree_cursor_id = siblings[idx]
        self._update_sidebar()
        self._update_topbar()

    def action_tree_prev_sibling(self) -> None:
        self._move_tree_sibling(-1)

    def action_tree_next_sibling(self) -> None:
        self._move_tree_sibling(1)

    def action_tree_open(self) -> None:
        if self._focused_panel != "tree":
            return
        if self._tree_cursor_id not in self.conv_tree.nodes:
            return
        self.conv_tree.current_id = self._tree_cursor_id
        self._save_active_session()
        self._rebuild_viewport()
        self._update_sidebar()
        self._update_topbar()

    def action_show_keymap(self) -> None:
        self._write_section_heading("Keymap")
        self._write_command_row("Tab / Shift+Tab", "Cycle active panels", col=20)
        self._write_command_row("Ctrl+H / Ctrl+L", "Focus transcript or tree", col=20)
        self._write_command_row("?", "Show this keymap", col=20)
        self._write_command_row("/", "Open slash command palette", col=20)
        self._write("")
        self._write_section_heading("Transcript")
        self._write_command_row("PgUp / PgDn", "Scroll transcript", col=20)
        self._write("")
        self._write_section_heading("Tree")
        self._write_command_row("j / k", "Move selection", col=20)
        self._write_command_row("Enter / o", "Open selected node", col=20)
        self._write_command_row("[ / ]", "Jump sibling branches", col=20)
        self._write_command_row("g / G", "Jump top or bottom", col=20)
        self._write("")
        self._write_section_heading("Input")
        self._write_command_row("Enter", "Send message", col=20)
        self._write_command_row("Esc", "Clear input or stop stream", col=20)
        self._write("")

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
        self._maybe_refresh_model_name()
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
        try:
            self.query_one("#attach-file", Button).disabled = value
        except Exception:
            pass
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
        self._last_log_was_blank = markup == ""
        self._maybe_scroll_end()

    def _write_indented(self, markup: str, indent: int = 2) -> None:
        self._log().write(Padding(Text.from_markup(markup), pad=(0, 0, 0, indent)))
        self._last_log_was_blank = False
        self._maybe_scroll_end()

    def _write_renderable(self, renderable, indent: int = 2) -> None:
        self._log().write(Padding(renderable, pad=(0, 0, 0, indent)))
        self._last_log_was_blank = False
        self._maybe_scroll_end()

    def _syntax_renderable(self, code: str, language: Optional[str]) -> Syntax:
        return Syntax(
            code,
            language or "text",
            theme="github-dark",
            word_wrap=True,
            background_color="#09090b",
            line_numbers=False,
        )

    def _code_panel_renderable(self, code: str, language: Optional[str]) -> Panel:
        return Panel(
            self._syntax_renderable(code, language),
            expand=True,
            padding=(0, 1),
            border_style="#27272a",
            style="on #09090b",
        )

    def _reasoning_panel_renderable(self, text: str) -> Panel:
        rendered, _ = render_md(text, False)
        return Panel(
            Text.from_markup(f"[dim]{rendered}[/dim]"),
            title="[dim #6366f1]thinking[/dim #6366f1]",
            title_align="left",
            expand=True,
            padding=(0, 1),
            border_style="#27272a",
            style="on #09090b",
            box=box.SQUARE,
        )

    def _tool_event_panel(
        self,
        title: str,
        title_color: str,
        border_color: str,
        name: str,
        detail: str = "",
    ) -> Panel:
        text = Text()
        text.append(name, style="bold #f4f4f5")
        if detail:
            text.append("   ")
            text.append(detail, style="#a1a1aa")
        return Panel(
            text,
            title=f"[bold {title_color}]{title}[/bold {title_color}]",
            title_align="left",
            expand=True,
            padding=(0, 1),
            border_style=border_color,
            style="on #09090b",
            box=box.SQUARE,
        )

    def _tool_lifecycle_panel(
        self,
        name: str,
        detail: str,
        *,
        ok: bool,
    ) -> Panel:
        return self._tool_event_panel(
            "tool → done" if ok else "tool → fail",
            "#10b981" if ok else "#f87171",
            "#10b981" if ok else "#f87171",
            name,
            detail,
        )

    def _update_tool_call_partial(self, name: str, detail: str = "", *, indent: int = 2) -> None:
        partial = self._partial()
        partial.display = True
        partial.update(Padding(self._tool_event_panel("tool", ACCENT_COLOR, ACCENT_COLOR, name, detail), (0, 0, 0, indent)))

    def _write_tool_lifecycle_block(self, name: str, ok: bool, detail: str = "", *, indent: int = 2) -> None:
        self._write_renderable(self._tool_lifecycle_panel(name, detail or ("completed" if ok else "failed"), ok=ok), indent=indent)

    def _show_tool_result_line(self, name: str, ok: bool) -> bool:
        if not ok:
            return True
        return self._show_tool_details

    def _write_create_files_preview_from_args(self, stream_id: str, args: Any) -> None:
        if not isinstance(args, dict):
            return
        files = args.get("files")
        if not isinstance(files, list):
            return
        rendered = self._live_preview.rendered_filepaths(stream_id) if stream_id else set()
        newly_rendered: set[str] = set()
        for item in files:
            if not isinstance(item, dict):
                continue
            filepath = str(item.get("filepath", ""))
            content = item.get("content")
            if not filepath or not isinstance(content, str) or not content.strip():
                continue
            if filepath in rendered:
                continue
            self._write(f"[dim]  · file draft: {esc(filepath)}[/dim]")
            self._write_code_block(content.splitlines(), self._live_preview._guess_language(filepath), 2)
            rendered.add(filepath)
            newly_rendered.add(filepath)
        if stream_id and newly_rendered:
            self._live_preview.mark_rendered_filepaths(stream_id, newly_rendered)

    def _take_pending_tool_detail(self, name: str) -> str:
        for idx, (pending_name, pending_detail) in enumerate(self._pending_tool_details):
            if pending_name == name:
                self._pending_tool_details.pop(idx)
                return pending_detail
        return ""

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

    def _render_static_markdown(self, text: str) -> None:
        in_fence = False
        fence_lang: Optional[str] = None
        fence_lines: List[str] = []
        for line in text.splitlines() or [""]:
            if self._is_fence_line(line):
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

    def _reset_fence_state(self) -> None:
        self._in_fence = False
        self._fence_lang = None
        self._fence_lines = []

    @staticmethod
    def _is_fence_line(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("```") or stripped.startswith("~~~")

    def _flush_fence_block(self) -> None:
        if self._fence_lines:
            self._write_code_block(self._fence_lines, self._fence_lang, indent=2)
        self._reset_fence_state()

    def _render_content_line(self, line: str) -> None:
        if self._is_fence_line(line):
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
            if self._buf_c and not self._is_fence_line(self._buf_c):
                lines.append(self._buf_c)
            if lines:
                partial.update(Padding(self._code_panel_renderable("\n".join(lines), self._fence_lang), (0, 0, 0, 2)))
            else:
                partial.update("")
            return
        if not self._buf_c:
            partial.update("")
            return
        rendered, _ = render_md(self._buf_c, False)
        partial.update(Padding(Text.from_markup(rendered), pad=(0, 0, 0, max(2, hanging_indent(self._buf_c)))))

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
        if self.streaming and self._auto_follow_stream and not self._is_near_bottom():
            self._auto_follow_stream = False
            self._update_status2()
        if not self.streaming or self._auto_follow_stream:
            self._scroll().scroll_end(animate=False)

    def _write_info(self, text: str) -> None:
        self._write(f"  [bold {ACCENT_COLOR}]›[/bold {ACCENT_COLOR}] [#f4f4f5]{esc(text)}[/#f4f4f5]")

    def _write_error(self, text: str) -> None:
        self._write(f"[bold red]  ✖ {esc(text)}[/bold red]")

    def _write_section_heading(self, title: str, color: str = ACCENT_COLOR) -> None:
        self._write("")
        self._write(f"[bold {color}]  {esc(title)}[/bold {color}]")

    def _write_detail_line(self, label: str, value: str, *, value_markup: bool = False) -> None:
        rendered = value if value_markup else esc(value)
        self._write(
            f"  [bold {ACCENT_COLOR}]{esc(label)}:[/bold {ACCENT_COLOR}] [#f4f4f5]{rendered}[/#f4f4f5]"
            if not value_markup
            else f"  [bold {ACCENT_COLOR}]{esc(label)}:[/bold {ACCENT_COLOR}] {rendered}"
        )

    def _write_indexed_dim_lines(self, rows: List[str], *, color: str = ACCENT_COLOR, allow_markup: bool = False) -> None:
        for index, row in enumerate(rows):
            if allow_markup:
                self._write(f"  [{color}]{index}.[/{color}] {row}")
            else:
                self._write(f"  [{color}]{index}.[/{color}] [#f4f4f5]{esc(row)}[/#f4f4f5]")

    def _write_command_action(self, text: str, *, icon: str = "•", color: str = ACCENT_COLOR) -> None:
        self._write(f"  [bold {color}]{esc(icon)}[/bold {color}] [#f4f4f5]{esc(text)}[/#f4f4f5]")

    def _write_command_row(self, command: str, desc: str, *, col: int) -> None:
        gap = max(1, col - len(command))
        self._write(
            f"  [bold {ACCENT_COLOR}]{esc(command)}[/bold {ACCENT_COLOR}]{' ' * gap}[#a1a1aa]{esc(desc)}[/#a1a1aa]"
        )

    def _write_muted_lines(self, rows: List[str]) -> None:
        for row in rows:
            self._write(f"  [#a1a1aa]{esc(row)}[/#a1a1aa]")

    def _write_usage(self, usage: str) -> bool:
        self._write_error(f"Usage: {usage}")
        return True

    def _ensure_command_gap(self) -> None:
        if not self._last_log_was_blank:
            self._write("")

    def _reload_skills(self) -> bool:
        self.agent.reload_skills()
        self._write_info("Reloaded skills")
        return True

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
        text = status_right_markup(
            model_name=self._model_name,
            branch_armed=bool(self.conv_tree._pending_branch),
            branch_label=self.conv_tree._pending_branch_label,
            thinking=self.thinking,
            width=self.size.width,
        )
        if text == self._last_status_right:
            self._update_topbar()
            return
        self._last_status_right = text
        self.query_one("#status-right", Static).update(text)
        self._update_topbar()

    def _maybe_refresh_model_name(self, *, force: bool = False) -> None:
        if self._model_refresh_inflight:
            return
        now = time.monotonic()
        if not force and now - self._last_model_refresh < self._model_refresh_interval_s:
            return
        self._last_model_refresh = now
        self._model_refresh_inflight = True
        self._refresh_model_name_worker()

    @work(thread=True, exclusive=True)
    def _refresh_model_name_worker(self) -> None:
        model_name = self.agent.fetch_model_name(timeout_s=min(self.agent.connect_timeout_s, 2.0))
        self.call_from_thread(self._apply_model_name_refresh, model_name)

    def _apply_model_name_refresh(self, model_name: Optional[str]) -> None:
        self._model_refresh_inflight = False
        self._model_name = model_name
        self._update_status1()

    def _update_status2(self) -> None:
        left = status_left_markup(
            await_shell_confirm=self._await_shell_confirm,
            streaming=self.streaming,
            spinner_frame=self._spin_frames[self._spin_i % len(self._spin_frames)],
            stop_requested=self._stop_event.is_set(),
            esc_pending=self._esc_pending,
            auto_follow_stream=self._auto_follow_stream,
            focus_panel=self._focused_panel,
            width=self.size.width,
        )
        if left == self._last_status_left:
            return
        self._last_status_left = left
        self.query_one("#status-left", Static).update(left)

    def _update_topbar(self) -> None:
        workspace_root = str(self.agent.skill_runtime.workspace.workspace_root)
        width = self.size.width
        self.query_one("#topbar-left", Static).update(topbar_left(workspace_root, width=width))
        self.query_one("#topbar-center", Static).update(
            topbar_center(
                session_name=self._session_title or "Session",
                branch_name=self._current_branch_name(),
                memory_mode=self._memory_mode_label(),
                width=width,
            )
        )
        self.query_one("#topbar-right", Static).update(
            topbar_right(
                endpoint=self.agent.model_endpoint,
                context_tokens=self._context_tokens(),
                width=width,
            )
        )

    def _update_input_placeholder(self) -> None:
        self.query_one(ChatInput).placeholder = (
            "Type to start branch…" if self.conv_tree._pending_branch else "Type a message…"
        )

    def _update_pending_attachments(self) -> None:
        try:
            self.query_one("#pending-attachments", Static).update(self._pending_attachment_markup())
        except Exception:
            return

    def _pending_attachment_markup(self) -> str:
        if not self.pending:
            return ""
        chips: List[str] = []
        visible = self.pending[:3]
        for path, _kind in visible:
            chips.append(f"[#f4f4f5 on #1a1730] {esc(os.path.basename(path))} [/#f4f4f5 on #1a1730]")
        overflow = len(self.pending) - len(visible)
        if overflow > 0:
            chips.append(f"[#a1a1aa on #1a1730] +{overflow} more [/#a1a1aa on #1a1730]")
        return " ".join(chips)

    def _workspace_root(self) -> Path:
        return Path(str(self.agent.skill_runtime.workspace.workspace_root)).resolve()

    def _home_root(self) -> Path:
        home_root = getattr(self.agent.skill_runtime.workspace, "home_root", None)
        if home_root:
            return Path(str(home_root)).resolve()
        return Path.home().resolve()

    def _attachment_root_path(self, root_id: str) -> Path:
        if root_id == "home":
            return self._home_root()
        return self._workspace_root()

    def _attachment_root_label(self, root_id: str) -> str:
        return "Home" if root_id == "home" else "Workspace"

    def _root_relative_label(self, path: Path, root: Path) -> str:
        try:
            relative = path.resolve().relative_to(root)
            text = relative.as_posix()
            return text if text else "."
        except Exception:
            return path.as_posix()

    def _resolve_attachment_path(self, raw_path: str) -> Path:
        candidate = Path(os.path.expanduser(raw_path.strip()))
        if candidate.is_absolute():
            resolved = candidate.resolve()
            if resolved.is_file():
                return resolved
            raise FileNotFoundError(str(resolved))

        workspace_candidate = (self._workspace_root() / candidate).resolve()
        cwd_candidate = (Path.cwd() / candidate).resolve()
        for resolved in (workspace_candidate, cwd_candidate):
            if resolved.is_file():
                return resolved
        raise FileNotFoundError(str(workspace_candidate))

    def _attach_file_path(self, path: str | Path) -> bool:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            self._write_error(f"File not found: {resolved}")
            return False
        kind = classify_attachment(str(resolved))
        if kind == "unknown":
            self._write_error("Unsupported file type")
            return False
        normalized = str(resolved)
        if any(existing_path == normalized for existing_path, _existing_kind in self.pending):
            self._write_info(f"Already attached: {resolved.name}")
            return False
        self.pending.append((normalized, kind))
        self._update_pending_attachments()
        self._update_status1()
        return True

    def _attachment_picker_items(self, relative_dir: str = ".", *, root_id: str = "workspace") -> List[PickerItem]:
        items: List[PickerItem] = []
        current = Path(relative_dir)
        for candidate_root in ("workspace", "home"):
            if candidate_root == root_id:
                continue
            items.append(
                PickerItem(
                    id=f"root:{candidate_root}:.",
                    prompt=(
                        f"[bold {ACCENT_COLOR}]switch → {self._attachment_root_label(candidate_root).lower()}[/bold {ACCENT_COLOR}]"
                    ),
                )
            )
        if relative_dir not in {".", ""}:
            parent = current.parent.as_posix()
            if parent == "":
                parent = "."
            items.append(PickerItem(id=f"nav:{parent}", prompt=f"[dim]{esc('../')}[/dim] [#a1a1aa]parent[/#a1a1aa]"))

        root_path = self._attachment_root_path(root_id)
        list_target = str(root_path if relative_dir in {".", ""} else (root_path / relative_dir))
        entries = self.agent.skill_runtime.workspace.list_files(list_target)
        for entry in entries:
            target = (current / entry.rstrip("/")).as_posix()
            if entry.endswith("/"):
                items.append(
                    PickerItem(
                        id=f"nav:{target}",
                        prompt=f"[bold {ACCENT_COLOR}]{esc(entry)}[/bold {ACCENT_COLOR}] [dim]open[/dim]",
                    )
                )
                continue
            target_path = root_path / target
            kind = classify_attachment(str(target_path))
            if kind == "unknown":
                continue
            items.append(
                PickerItem(
                    id=f"file:{target}",
                    prompt=f"[#f4f4f5]{esc(entry)}[/#f4f4f5] [dim]{kind}[/dim]",
                )
            )
        return items

    def _open_attachment_picker(self, relative_dir: str = ".", root_id: str = "workspace") -> None:
        clean_dir = relative_dir or "."
        root_path = self._attachment_root_path(root_id)
        title = f"Attach File · {self._attachment_root_label(root_id)}"
        subtitle = self._root_relative_label(root_path / clean_dir, root_path)
        items = self._attachment_picker_items(clean_dir, root_id=root_id)
        self.push_screen(
            SelectionPickerModal(
                title=title,
                subtitle=subtitle,
                confirm_label="Open / Attach",
                empty_text="No attachable files in this folder.",
                items=items,
            ),
            lambda result: self._on_attachment_picker_close(root_id, clean_dir, result),
        )

    def _on_attachment_picker_close(self, root_id: str, current_dir: str, result: Optional[Dict[str, str]]) -> None:
        selection = str((result or {}).get("id") or "").strip()
        if not selection:
            self.query_one(ChatInput).focus()
            return
        if selection.startswith("root:"):
            _, next_root, next_dir = selection.split(":", 2)
            self._open_attachment_picker(next_dir or ".", root_id=next_root or "workspace")
            return
        if selection.startswith("nav:"):
            self._open_attachment_picker(selection[4:] or ".", root_id=root_id)
            return
        if selection.startswith("file:"):
            target = selection[5:]
            self._attach_file_path(self._attachment_root_path(root_id) / target)
        self.query_one(ChatInput).focus()

    def _refresh_command_popup(self, value: str) -> None:
        popup = self._command_popup()
        options = self._command_options()
        next_matches = command_entries_for_query(value)
        if not next_matches or self.streaming or self._await_shell_confirm:
            next_matches = []

        if not next_matches:
            self._command_matches = []
            popup.display = False
            options.clear_options()
            popup.absolute_offset = None
            return

        was_hidden = not bool(popup.display)
        geometry_changed = self._command_anchor_region != self.query_one(ChatInput).region
        self._command_anchor_region = self.query_one(ChatInput).region
        self._command_matches = next_matches
        popup.display = True
        rendered = [
            Option(
                f"[bold #6366f1]{esc(command_label(entry))}[/bold #6366f1] [dim]{esc(entry.description)}[/dim]",
                id=str(index),
            )
            for index, entry in enumerate(self._command_matches)
        ]
        options.clear_options()
        options.add_options(rendered)
        options.highlighted = 0
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
        return base not in exact_command_inputs()

    @staticmethod
    def _config_for_editor(config: Dict[str, Any]) -> Dict[str, Any]:
        return config_for_editor_view(config)

    def _open_config_editor(self) -> None:
        warnings: List[str] = []
        try:
            raw = load_or_create_global_config(GLOBAL_CONFIG_PATH, warnings=warnings)
        except Exception as exc:  # noqa: BLE001
            self._write_error(f"Config load failed: {exc}")
            return
        safe = self._config_for_editor(raw if isinstance(raw, dict) else {})
        text = json.dumps(safe, indent=2) + "\n"
        self.push_screen(ConfigEditorModal(GLOBAL_CONFIG_PATH, text), self._on_config_editor_close)
        for warning in warnings:
            self._write_info(f"Config warning: {warning}")

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
        parsed = result.get("config")
        if not isinstance(parsed, dict):
            self._write_error("Config save failed: invalid config payload")
            return

        try:
            normalized, warnings = normalize_config(parsed)
            validate_endpoint_policy(normalized)
        except Exception as exc:  # noqa: BLE001
            self._write_error(f"Config save failed: {exc}")
            return

        cleaned = self._config_for_editor(normalized)
        GLOBAL_CONFIG_PATH.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
        merged = self._merge_live_config(self.agent.config, normalized)
        self.agent.reload_config(merged)
        self.agent.skill_runtime.config = merged
        self.thinking = bool(merged.get("agent", {}).get("enable_thinking", self.thinking))
        self._model_name = None
        self._update_topbar()
        self._apply_tui_config()
        self._maybe_refresh_model_name(force=True)
        suffix = f" ({len(warnings)} normalization warning{'s' if len(warnings) != 1 else ''})." if warnings else "."
        self._write_info("Saved global config. Use environment variables for secrets like TAVILY_API_KEY or BRAVE_SEARCH_API_KEY" + suffix)
        for warning in warnings:
            self._write_info(f"Config warning: {warning}")

    def _open_code_block(self, index: int) -> None:
        if index < 1 or index > len(self._code_blocks):
            self._write_error(f"No code block {index}")
            return
        code, language = self._code_blocks[index - 1]
        self.push_screen(CodeViewerModal(code, language, title=f"Code Block {index}"))

    def _update_sidebar(self) -> None:
        sidebar = self.query_one("#sidebar", Vertical)
        if not sidebar.display:
            return
        if self._focused_panel != "tree":
            self._tree_cursor_id = self.conv_tree.current_id
        else:
            self._sync_tree_cursor()
        self.query_one("#sidebar-tree-meta", Static).update(f"{self.conv_tree.turn_count()} turns")
        self.query_one("#sidebar-tree-content", Static).update(
            render_sidebar_tree_markup(self.conv_tree, width=30, selected_id=self._tree_cursor_id)
        )
        self.query_one("#sidebar-inspector-content", Static).update(
            render_sidebar_inspector_markup(self.conv_tree, width=30, selected_id=self._tree_cursor_id)
        )

    def _write_turn_user(self, turn: Turn) -> None:
        self._write("")
        if turn.branch_root:
            label = f" ⎇  {esc(turn.label)}" if turn.label else " ⎇  branch"
            self._write(f"[dim #6366f1]{label}[/dim #6366f1]")
        self._write(f"[bold {ACCENT_COLOR}]You[/bold {ACCENT_COLOR}]")
        attachment_summary = turn.attachment_summary()
        if attachment_summary:
            self._write_indented(f"[dim]attachments:[/dim] [#a1a1aa]{esc(attachment_summary)}[/#a1a1aa]", indent=2)
        body = turn.user_text()
        for line in body.splitlines() or [""]:
            self._write_indented(esc(line), indent=2)

    def _write_skill_exchanges(self, turn: Turn) -> None:
        if not self._show_historical_tool_details:
            return
        pending_details: List[Tuple[str, str]] = []
        for msg in turn.skill_exchanges:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for call in msg["tool_calls"]:
                    name = call.get("function", {}).get("name", "unknown")
                    raw_args = call.get("function", {}).get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args = raw_args
                    pending_details.append((name, self._live_preview.compact_tool_args(name, args)))
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
                    self._live_preview.write_result_preview(
                        name, payload, self._write, self._write_indented, self._write_code_block
                    )
                if not self._show_tool_result_line(name, bool(payload.get("ok"))):
                    continue
                detail = ""
                for idx, (pending_name, pending_detail) in enumerate(pending_details):
                    if pending_name == name:
                        detail = pending_detail
                        pending_details.pop(idx)
                        break
                if payload.get("ok"):
                    self._write_tool_lifecycle_block(name, True, detail or "completed", indent=2)
                else:
                    em = payload.get("error", {}).get("message", "failed")
                    self._write_tool_lifecycle_block(name, False, f"{detail}   {em}".strip(), indent=2)

    def _write_completed_turn_asst(self, turn: Turn) -> None:
        self._write("")
        self._write("[bold #6366f1]Assistant[/bold #6366f1]")
        self._write_skill_exchanges(turn)

        content = turn.assistant_content or ""
        state = str(getattr(turn, "assistant_state", "") or ("cancelled" if "[interrupted]" in content else "done"))
        interrupted = state == "cancelled"
        failed = state == "error"
        display = content.replace("\n[interrupted]", "").rstrip()
        self._render_static_markdown(display)

        if interrupted:
            self._write("[dim red]  ✖ interrupted[/dim red]")
        elif failed:
            self._write("[dim red]  ! failed[/dim red]")
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
        self._live_preview.reset()
        self._pending_tool_details = []
        self._last_stream_error_text = ""
        self._reasoning_open = False
        self._content_open = False
        self._buf_r = ""
        self._buf_c = ""
        self._reset_fence_state()
        self._stop_event = threading.Event()
        self._partial().display = True

        self._write("")
        self._write("[bold #6366f1]Assistant[/bold #6366f1]")

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

    def _visible_reasoning_text(self, text: str) -> str:
        if not text:
            return ""
        filtered_lines = [line for line in text.splitlines() if not self._is_tool_trace_line(line)]
        visible = "\n".join(filtered_lines).strip()
        return visible

    def _flush_reasoning_buffer(self) -> None:
        if not self._buf_r:
            self._partial().update("")
            return
        text = self._buf_r
        self._buf_r = ""
        visible = self._visible_reasoning_text(text)
        self._partial().update("")
        if visible:
            self._write_renderable(self._reasoning_panel_renderable(visible), indent=2)

    def _close_reasoning_section(self) -> bool:
        if not self._reasoning_open:
            return False
        self._flush_reasoning_buffer()
        self._reasoning_open = False
        return True

    def _flush_content_buffer(self, include_partial: bool = False) -> None:
        while "\n" in self._buf_c:
            line, self._buf_c = self._buf_c.split("\n", 1)
            if self._is_tool_trace_line(line):
                continue
            self._render_content_line(line)

        if include_partial and self._buf_c:
            if not self._is_tool_trace_line(self._buf_c):
                if self._in_fence:
                    if self._is_fence_line(self._buf_c):
                        self._flush_fence_block()
                    else:
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
            if self._close_reasoning_section():
                self._write("")
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
            self._buf_r += token
            display = self._visible_reasoning_text(self._buf_r)
            if display:
                partial.update(Padding(self._reasoning_panel_renderable(display), (0, 0, 0, 2)))
            else:
                partial.update("")

        elif etype == "content_token":
            token = event.get("text", "")
            self._handle_content_token(token)

        elif etype == "tool_phase_started":
            # Preserve in-progress text before tool call deltas start.
            self._close_reasoning_section()
            self._flush_content_buffer(include_partial=True)
            self._content_open = False

        elif etype == "tool_call_delta":
            if not self._show_tool_details:
                return
            stream_id = str(event.get("stream_id") or "")
            name = str(event.get("name") or "")
            raw_arguments = str(event.get("raw_arguments") or "")
            if stream_id and name:
                self._live_preview.update(
                    stream_id,
                    name,
                    raw_arguments,
                    self._write,
                    self._update_live_preview_partial,
                    self._write_indented,
                    self._write_code_block,
                    self._clear_partial_preview,
                )

        elif etype == "tool_call":
            self._close_reasoning_section()
            name = event.get("name", "tool")
            args = event.get("arguments", {})
            stream_id = str(event.get("stream_id") or "")
            detail = self._live_preview.compact_tool_args(name, args)
            if self._show_tool_details:
                self._pending_tool_details.append((name, detail))
                self._update_tool_call_partial(name, detail, indent=2)
                if name == "create_files":
                    self._write_create_files_preview_from_args(stream_id, args)
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
            elif stream_id:
                self._live_preview.close(
                    stream_id, self._write_indented, self._write_code_block, self._clear_partial_preview
                )

        elif etype == "tool_result":
            self._close_reasoning_section()
            name = event.get("name", "tool")
            result = event.get("result", {})
            if result.get("ok"):
                self._live_preview.write_result_preview(
                    name, result, self._write, self._write_indented, self._write_code_block
                )
            self._clear_partial_preview()
            if not self._show_tool_result_line(name, bool(result.get("ok"))):
                return
            detail = self._take_pending_tool_detail(name)
            if result.get("ok"):
                self._write_tool_lifecycle_block(name, True, detail or "completed", indent=2)
            else:
                msg = result.get("error", {}).get("message", "failed")
                self._write_tool_lifecycle_block(name, False, f"{detail}   {msg}".strip(), indent=2)

        elif etype == "error":
            self._last_stream_error_text = str(event.get("text", "Unknown error"))
            self._write_error(self._last_stream_error_text)

        elif etype == "info":
            self._write_info(str(event.get("text", "")))

        elif etype == "pass_end":
            finish_reason = str(event.get("finish_reason") or "")
            has_content = bool(event.get("has_content"))
            has_tool_calls = bool(event.get("has_tool_calls"))
            # Some model passes end with reasoning-only stop and no
            # visible output. Drop that provisional reasoning so it doesn't leak.
            if finish_reason in {"stop", "length"} and not has_content and not has_tool_calls:
                self._buf_r = ""
                partial.update("")

        elif etype == "usage":
            usage = event.get("usage") or {}
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens")
                total_tokens = usage.get("total_tokens")
                if isinstance(prompt_tokens, (int, float)):
                    self._last_model_context_tokens = int(prompt_tokens)
                elif isinstance(total_tokens, (int, float)):
                    self._last_model_context_tokens = int(total_tokens)
                self._update_topbar()

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
            self._close_reasoning_section()
        elif self._reasoning_open and not self._content_open:
            self._close_reasoning_section()

        self._flush_content_buffer(include_partial=True)

        reply = result.content if result.content else self._reply_acc
        if result.status == "done" and not self._content_open and reply.strip():
            self._render_static_markdown(reply)

        if turn_id in self.conv_tree.nodes:
            for msg in result.skill_exchanges:
                self.conv_tree.append_skill_exchange(turn_id, msg)

        if result.status == "done":
            self.conv_tree.complete_turn(turn_id, reply)
        elif result.status == "cancelled":
            self.conv_tree.cancel_turn(turn_id, reply)
        else:
            self.conv_tree.fail_turn(turn_id, reply)
        usage = result.journal.get("model_usage", {}) if isinstance(result.journal, dict) else {}
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            total_tokens = usage.get("total_tokens")
            if isinstance(prompt_tokens, (int, float)):
                self._last_model_context_tokens = int(prompt_tokens)
            elif isinstance(total_tokens, (int, float)):
                self._last_model_context_tokens = int(total_tokens)

        if result.status != "done":
            if result.error and result.error != self._last_stream_error_text:
                self._write_error(result.error)
            if result.status == "cancelled":
                self._write("[bold red]  ✖ interrupted[/bold red]")

        self._save_active_session()
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
        self._update_topbar()

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
        self._save_active_session()
        self._write_turn_user(turn)
        self._update_pending_attachments()
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
        handled = self._handle_command(text)
        if handled:
            cmd = text.split(None, 1)[0].lower()
            if cmd not in {"/quit", "/exit", "/q", "/config"}:
                self._ensure_command_gap()
            return
        if not handled:
            if not self.streaming:
                self._send(text)

    @on(Input.Changed, "#chat-input")
    def on_input_changed(self, event: Input.Changed) -> None:
        self._refresh_command_popup(event.value)

    @on(Button.Pressed, "#attach-file")
    def on_attach_file_pressed(self) -> None:
        if self.streaming:
            return
        self._hide_command_popup()
        self._open_attachment_picker(".")

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

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        if self.streaming:
            self._auto_follow_stream = False
            self._update_status2()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        if not self.streaming:
            return
        self.call_after_refresh(self._resume_auto_follow_if_near_bottom)

    def _resume_auto_follow_if_near_bottom(self) -> None:
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

        if cmd == "/details":
            self._show_tool_details = not self._show_tool_details
            self._write_info(f"Live tool details {'shown' if self._show_tool_details else 'hidden'}")
            return True

        if cmd == "/think":
            self.thinking = not self.thinking
            self._write_info(f"Thinking {'enabled' if self.thinking else 'disabled'}")
            return True

        if cmd == "/sessions":
            self._cmd_sessions()
            return True

        if self.streaming and cmd in {"/new", "/load", "/import", "/clear"}:
            self._write_error("Stop the active response before changing sessions.")
            return True

        if cmd == "/new":
            session = self._open_new_session(arg)
            self._switch_to_session(session)
            self._write_command_action(f"Opened session '{session.title}'", icon="✓")
            return True

        if cmd == "/rename":
            if not arg:
                return self._write_usage("/rename <name>")
            session = self._save_active_session(rename_to=arg)
            self._update_topbar()
            self._write_command_action(f"Renamed session to '{session.title}'", icon="✓")
            return True

        if cmd == "/branch":
            self.conv_tree.arm_branch(arg)
            self._save_active_session()
            label = self.conv_tree._pending_branch_label
            self._write_command_action(f"Branch armed '{label}'", icon="⎇", color=ACCENT_COLOR)
            self._update_status1()
            self._update_input_placeholder()
            return True

        if cmd == "/unbranch":
            if self.conv_tree._pending_branch:
                self.conv_tree.clear_pending_branch()
                self._save_active_session()
                self._write_command_action("Disarmed pending branch", icon="↩", color=ACCENT_COLOR)
                self._update_status1()
                self._update_input_placeholder()
                return True
            moved = self.conv_tree.unbranch()
            if moved is None:
                self._write_error("No branch to leave.")
            else:
                self._save_active_session()
                self._write_command_action("Returned to fork point", icon="↩", color=ACCENT_COLOR)
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
                self._write_indexed_dim_lines([self.conv_tree.nodes[cid].short(60) for cid in children])
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
                self._save_active_session()
                self._write_command_action(f"Switched to branch {idx}", icon="↪")
                self._rebuild_viewport()
                self._update_sidebar()
            return True

        if cmd == "/tree":
            self._cmd_tree()
            return True

        if cmd == "/save":
            try:
                session = self._save_active_session(rename_to=arg or None)
                self._update_topbar()
                self._write_command_action(f"Saved session '{session.title}'", icon="✓")
            except Exception as exc:
                self._write_error(f"Save failed: {exc}")
            return True

        if cmd == "/load":
            self._open_load_session_picker()
            return True

        if cmd == "/export":
            try:
                path = self._session_store.export_session_tree(self._session_title, self.conv_tree)
                self._write_command_action(f"Exported session to {path.name}", icon="✓")
            except Exception as exc:
                self._write_error(f"Export failed: {exc}")
            return True

        if cmd == "/import":
            self._open_import_picker()
            return True

        if cmd == "/clear":
            self.conv_tree = self._new_conv_tree()
            self.pending.clear()
            self._log().clear()
            self._partial().update("")
            self._save_active_session()
            self._update_pending_attachments()
            self._update_status1()
            self._update_status2()
            self._update_sidebar()
            self._update_input_placeholder()
            return True

        if cmd in {"/file", "/image"}:
            if not arg:
                self._open_attachment_picker(".")
                return True
            try:
                path = self._resolve_attachment_path(arg)
            except FileNotFoundError:
                self._write_error(f"File not found: {arg}")
                return True
            self._attach_file_path(path)
            return True

        if cmd == "/skills":
            self._cmd_skills()
            return True

        if cmd == "/reload":
            return self._reload_skills()

        if cmd == "/doctor":
            self._cmd_doctor()
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

        if cmd == "/report":
            return self._cmd_report(arg)

        if cmd == "/code":
            return self._cmd_code(arg)

        if cmd.startswith("/"):
            self._write_error(f"Unknown command: {cmd}")
            return True

        return False

    def _cmd_help(self) -> None:
        self._write("")
        col = max((len(command) for _, rows in HELP_SECTIONS for command, _ in rows), default=22) + 2
        for section, rows in HELP_SECTIONS:
            self._write_section_heading(section)
            for c, desc in rows:
                self._write_command_row(c, desc, col=col)
        self._write("")

    def _cmd_tree(self) -> None:
        self._write_section_heading("Tree")
        for text, tag, active in self.conv_tree.render_tree(width=80):
            if tag == self.conv_tree.current_id:
                self._write(f"  [bold {ACCENT_COLOR}]{esc(text)}[/bold {ACCENT_COLOR}]")
            elif active:
                self._write(f"  [{ACCENT_COLOR}]{esc(text)}[/{ACCENT_COLOR}]")
            else:
                self._write(f"  [#a1a1aa]{esc(text)}[/#a1a1aa]")
        self._write("")

    def _cmd_skills(self) -> None:
        skills = self.agent.skill_runtime.list_skills()
        self._write_section_heading("Skills")
        for skill in skills:
            state, color = self.agent.skill_runtime.skill_status_label(skill)
            source = self.agent.skill_runtime.skill_source_label(skill)
            provenance = self.agent.skill_runtime.skill_provenance_label(skill)
            self._write(
                f"  [bold {ACCENT_COLOR}]{esc(skill.id)}[/bold {ACCENT_COLOR}] "
                f"[#a1a1aa]({esc(skill.version)})[/#a1a1aa] "
                f"[{color}]{state}[/{color}]"
            )
            self._write(f"    [#a1a1aa]{esc(skill.description)}[/#a1a1aa]")
            source_bits = provenance
            if source:
                source_bits += f" · {source}"
            self._write(f"    [#71717a]{esc(source_bits)}[/#71717a]")
            if not skill.available and skill.availability_reason:
                code = esc(skill.availability_code or "blocked")
                self._write(f"    [bold {ACCENT_COLOR}]blocked ({code}):[/bold {ACCENT_COLOR}] [#a1a1aa]{esc(skill.availability_reason)}[/#a1a1aa]")
        self._write("")

    def _cmd_skill(self, arg: str) -> bool:
        parts = arg.split()
        if not parts:
            return self._write_usage("/skill on|off|reload|info <id>")

        sub = parts[0].lower()
        if sub == "reload":
            return self._reload_skills()

        if sub in {"on", "off"}:
            if len(parts) < 2:
                return self._write_usage("/skill on|off <id>")
            skill_id = parts[1]
            ok = self.agent.skill_runtime.set_enabled(skill_id, sub == "on")
            if not ok:
                self._write_error(f"Skill not found: {skill_id}")
            else:
                self._write_info(f"Skill {skill_id} {'enabled' if sub == 'on' else 'disabled'}")
            return True

        if sub == "info":
            if len(parts) < 2:
                return self._write_usage("/skill info <id>")
            skill = self.agent.skill_runtime.get_skill(parts[1])
            if not skill:
                self._write_error(f"Skill not found: {parts[1]}")
                return True
            self._write_section_heading(skill.name)
            self._write(f"  [#a1a1aa]{esc(skill.description)}[/#a1a1aa]")
            keywords = ", ".join(skill.triggers.get("keywords", [])) or "none"
            file_ext = ", ".join(skill.triggers.get("file_ext", [])) or "none"
            tools = ", ".join(skill.allowed_tools) or "all"
            enabled, color = self.agent.skill_runtime.skill_status_label(skill)
            self._write_detail_line("id", skill.id)
            self._write_detail_line("version", skill.version)
            self._write_detail_line("status", f"[{color}]{enabled}[/{color}]", value_markup=True)
            self._write_detail_line("provenance", self.agent.skill_runtime.skill_provenance_label(skill))
            self._write_detail_line("source", self.agent.skill_runtime.skill_source_label(skill) or "unknown")
            self._write_detail_line("compatibility", skill.compatibility or "none")
            self._write_detail_line("availability_code", skill.availability_code or "ready")
            self._write_detail_line("availability", skill.availability_reason or "ready")
            self._write_detail_line("keywords", keywords)
            self._write_detail_line("file_ext", file_ext)
            self._write_detail_line("tools", tools)
            self._write("")
            return True

        return self._write_usage("/skill on|off|reload|info <id>")

    def _cmd_memory(self, arg: str) -> bool:
        sub = arg.strip().lower()
        if sub == "stats":
            stats = self.agent.skill_runtime.memory.stats()
            self._write_section_heading("Memory Stats")
            self._write_detail_line("count", str(stats["count"]))
            self._write_detail_line("mode", str(stats.get("mode_label", stats["embedding_backend"])))
            self._write_detail_line("backend", str(stats["embedding_backend"]))
            self._write_detail_line("configured_backend", str(stats.get("configured_embedding_backend", "")))
            self._write_detail_line("allow_model_download", str(stats.get("allow_model_download", False)).lower())
            self._write_detail_line("encoder_status", str(stats.get("encoder_status", "")))
            self._write_detail_line("encoder_source", str(stats.get("encoder_source", "")))
            if stats.get("encoder_detail"):
                self._write_detail_line("encoder_detail", str(stats.get("encoder_detail", "")))
            self._write_detail_line("model", str(stats["model_name"]))
            self._write_detail_line("recommended_model", str(stats.get("recommended_model_name", "")))
            self._write_detail_line("dimension", str(stats["dimension"]))
            self._write_detail_line("by_type", json.dumps(stats["by_type"]))
            self._write("")
            return True
        return self._write_usage("/memory stats")

    def _cmd_doctor(self) -> None:
        report = self.agent.doctor_report()
        self._write_section_heading("Doctor")
        agent = report.get("agent", {})
        workspace = report.get("workspace", {})
        memory = report.get("memory", {})
        search = report.get("search", {})
        self._write_detail_line("endpoint_ready", str(agent.get("ready", False)).lower())
        if agent.get("endpoint_policy_error"):
            self._write_detail_line("endpoint_policy", str(agent.get("endpoint_policy_error")))
        self._write_detail_line("workspace", str(workspace.get("path", "")))
        self._write_detail_line("workspace_writable", str(workspace.get("writable", False)).lower())
        self._write_detail_line("memory_mode", str(memory.get("mode", "")))
        self._write_detail_line("memory_backend", str(memory.get("backend", "")))
        self._write_detail_line("memory_configured_backend", str(memory.get("configured_backend", "")))
        self._write_detail_line("memory_allow_model_download", str(memory.get("allow_model_download", False)).lower())
        self._write_detail_line("memory_encoder_status", str(memory.get("encoder_status", "")))
        self._write_detail_line("memory_encoder_source", str(memory.get("encoder_source", "")))
        if memory.get("encoder_detail"):
            self._write_detail_line("memory_encoder_detail", str(memory.get("encoder_detail", "")))
        self._write_detail_line("memory_model", str(memory.get("model_name", "")))
        self._write_detail_line("recommended_model", str(memory.get("recommended_model_name", "")))
        self._write_detail_line("search_provider", str(search.get("provider", "")))
        self._write_detail_line("search_ready", str(search.get("ready", False)).lower())
        if search.get("reason"):
            self._write_detail_line("search_reason", str(search.get("reason")))
        self._write_section_heading("Skills")
        for skill in report.get("skills", []):
            line = (
                f"  [bold {ACCENT_COLOR}]{esc(str(skill.get('id', '')))}[/bold {ACCENT_COLOR}] "
                f"[#a1a1aa]({esc(str(skill.get('source_tier', '')))} · {esc(str(skill.get('availability_code', 'ready')))})[/#a1a1aa] "
                f"[#a1a1aa]{esc(str(skill.get('status', 'unknown')))}[/#a1a1aa]"
            )
            self._write(line)
            reason = str(skill.get("availability_reason", "")).strip()
            if reason and reason != "ready":
                self._write(f"    [#a1a1aa]{esc(reason)}[/#a1a1aa]")
        self._write("")

    def _cmd_report(self, arg: str) -> bool:
        path = arg.strip() or "alphanus-support-report.json"
        payload = self.agent.build_support_bundle(self.conv_tree.to_dict())
        try:
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            self._write_error(f"Report save failed: {exc}")
            return True
        self._write_info(f"Saved support bundle to {path}")
        return True

    def _cmd_workspace(self, arg: str) -> bool:
        sub = arg.strip().lower()
        if sub == "tree":
            tree = self.agent.skill_runtime.workspace.workspace_tree()
            self._write_section_heading("Workspace Tree")
            self._write_muted_lines(tree.splitlines())
            self._write("")
            return True
        return self._write_usage("/workspace tree")

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
            return self._write_usage("/code [n|last]")
        self._open_code_block(index)
        return True
