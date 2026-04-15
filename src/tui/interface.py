from __future__ import annotations

from dataclasses import dataclass
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.console import RenderableType
from rich.highlighter import Highlighter
from rich.markup import escape as esc
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import Button, Input, OptionList, Static

from agent.core import Agent, AgentTurnResult
from agent.types import ModelStatus
from alphanus_paths import get_app_paths
from core.attachments import build_content, classify_attachment
from core.configuration import (
    config_for_editor_view,
    load_or_create_global_config,
    normalize_config,
    validate_endpoint_policy,
)
from core.conv_tree import ConvTree, Turn
from core.runtime_config import UiRuntimeConfig
from core.sessions import ChatSession
from tui.app_shell_runtime import compose_shell as compose_tui_shell
from tui.app_shell_runtime import initialize_shell_state as init_tui_shell_state
from tui.commands import (
    HELP_SECTIONS,
)
from tui.command_palette_runtime import (
    accept_command_selection as accept_tui_command_selection,
    command_popup_active as tui_command_popup_active,
    hide_command_popup as hide_tui_command_popup,
    move_command_selection as move_tui_command_selection,
    refresh_command_popup as refresh_tui_command_popup,
    select_command_option as select_tui_command_option,
    should_accept_popup_on_enter as should_tui_accept_popup_on_enter,
)
from tui.command_runtime import handle_command as handle_tui_command
from tui.command_output_runtime import (
    cmd_code as cmd_tui_code,
    cmd_context as cmd_tui_context,
    cmd_doctor as cmd_tui_doctor,
    cmd_help as cmd_tui_help,
    cmd_memory as cmd_tui_memory,
    cmd_report as cmd_tui_report,
    cmd_skill as cmd_tui_skill,
    cmd_skills as cmd_tui_skills,
    cmd_tree as cmd_tui_tree,
    cmd_workspace as cmd_tui_workspace,
    load_skill_into_session as load_tui_skill_into_session,
    unload_skill_from_session as unload_tui_skill_from_session,
)
from tui.config_runtime import apply_tui_config as apply_tui_runtime_config
from tui.config_runtime import install_stream_drain_timer as install_tui_stream_drain_timer
from tui.config_runtime import merge_live_config as merge_tui_live_config
from tui.interaction_runtime import (
    action_handle_esc as action_tui_handle_esc,
    begin_shell_confirm as begin_tui_shell_confirm,
    confirm_shell_command as confirm_tui_shell_command,
    expire_shell_confirm as expire_tui_shell_confirm,
    finish_shell_confirm as finish_tui_shell_confirm,
    on_input_submitted as on_tui_input_submitted,
    on_session_manager_close as on_tui_session_manager_close,
    show_keyboard_shortcuts as show_tui_keyboard_shortcuts,
)
from tui.popups import (
    CodeViewerModal,
    ConfigEditorModal,
    PickerItem,
    SelectionPickerModal,
    SessionManagerModal,
)
from tui.attachment_runtime import (
    attach_file_path as attach_tui_file_path,
    attachment_picker_items as attachment_tui_picker_items,
    attachment_root_label as attachment_tui_root_label,
    attachment_root_path as attachment_tui_root_path,
    home_root as home_tui_root,
    on_attachment_picker_close as on_tui_attachment_picker_close,
    open_attachment_picker as open_tui_attachment_picker,
    resolve_attachment_path as resolve_tui_attachment_path,
    root_relative_label as tui_root_relative_label,
    workspace_root as workspace_tui_root,
)
from tui.session_runtime import (
    activate_session_state as activate_tui_session_state,
    current_session_is_blank as is_blank_tui_session,
    delete_session_from_manager as delete_tui_session_from_manager,
    load_session_from_manager as load_tui_session_from_manager,
    open_new_session as open_tui_new_session,
    save_active_session as save_tui_session,
    switch_to_session as switch_tui_session,
)
from tui.status import status_right_markup
from tui.status_runtime import (
    StatusRuntimeState,
    apply_model_status as apply_tui_model_status,
    current_model_refresh_interval as tui_model_refresh_interval,
    finish_startup_readiness_poll as finish_tui_startup_readiness_poll,
    maybe_refresh_model_status as maybe_refresh_tui_model_status,
    should_startup_readiness_poll as should_tui_startup_readiness_poll,
    start_startup_readiness_poll as start_tui_startup_readiness_poll,
    update_status2 as update_tui_status2,
    update_topbar as update_tui_topbar,
)
from tui.stream_runtime import (
    StreamRuntimeState,
    close_reasoning_section as close_tui_reasoning_section,
    drain_events as drain_tui_stream_events,
    enqueue_event as enqueue_tui_stream_event,
    finish_turn_stream as finish_tui_turn_stream,
    flush_content_buffer as flush_tui_content_buffer,
    flush_reasoning_buffer as flush_tui_reasoning_buffer,
    handle_content_token as handle_tui_content_token,
    on_agent_event as on_tui_agent_event,
    refresh_deferred_partial as refresh_tui_deferred_partial,
    start_turn_stream as start_tui_turn_stream,
    visible_reasoning_text as visible_tui_reasoning_text,
)
from tui.transcript import ScrollAnchor, TranscriptEntry, TranscriptView, count_renderable_lines
from tui.transcript_runtime import (
    append_transcript_entry as tui_append_transcript_entry,
    bar_renderable as tui_bar_renderable,
    cached_partial_line_count as tui_cached_partial_line_count,
    capture_scroll_anchor as tui_capture_scroll_anchor,
    clear_partial_preview as tui_clear_partial_preview,
    code_panel_renderable as tui_code_panel_renderable,
    current_partial_line_count as tui_current_partial_line_count,
    defer_live_preview_partial as tui_defer_live_preview_partial,
    ensure_command_gap as tui_ensure_command_gap,
    flush_fence_block as tui_flush_fence_block,
    is_fence_line as tui_is_fence_line,
    is_near_bottom as tui_is_near_bottom,
    line_indents as tui_line_indents,
    maybe_scroll_end as tui_maybe_scroll_end,
    partial_measurement_width as tui_partial_measurement_width,
    pending_attachment_markup as tui_pending_attachment_markup,
    reasoning_panel_renderable as tui_reasoning_panel_renderable,
    remeasure_partial_line_count as tui_remeasure_partial_line_count,
    remember_code_block as tui_remember_code_block,
    render_content_line as tui_render_content_line,
    render_static_markdown as tui_render_static_markdown,
    reset_fence_state as tui_reset_fence_state,
    restore_scroll_anchor as tui_restore_scroll_anchor,
    set_partial_renderable as tui_set_partial_renderable,
    show_tool_result_line as tui_show_tool_result_line,
    syntax_renderable as tui_syntax_renderable,
    take_pending_tool_detail as tui_take_pending_tool_detail,
    tool_event_panel as tui_tool_event_panel,
    tool_lifecycle_panel as tui_tool_lifecycle_panel,
    update_live_preview_partial as tui_update_live_preview_partial,
    update_partial_content as tui_update_partial_content,
    update_tool_call_partial as tui_update_tool_call_partial,
    write_assistant_bar_line as tui_write_assistant_bar_line,
    write_assistant_bar_renderable as tui_write_assistant_bar_renderable,
    write_assistant_bar_wrapped_line as tui_write_assistant_bar_wrapped_line,
    write_code_block as tui_write_code_block,
    write_command_action as tui_write_command_action,
    write_command_row as tui_write_command_row,
    write_detail_line as tui_write_detail_line,
    write_error as tui_write_error,
    write_indexed_dim_lines as tui_write_indexed_dim_lines,
    write_info as tui_write_info,
    write_markup as tui_write_markup,
    write_muted_lines as tui_write_muted_lines,
    write_renderable as tui_write_renderable,
    write_section_heading as tui_write_section_heading,
    write_tool_lifecycle_block as tui_write_tool_lifecycle_block,
    write_usage as tui_write_usage,
    write_user_bar_line as tui_write_user_bar_line,
    write_user_bar_wrapped_line as tui_write_user_bar_wrapped_line,
)
from tui.ui_styles import ALPHANUS_TUI_CSS
from tui.tree_render import render_tree_rows
from tui.view_runtime import (
    rebuild_viewport as rebuild_tui_viewport,
    sidebar_render_width as tui_sidebar_render_width,
    update_sidebar as update_tui_sidebar,
    write_completed_turn_assistant as write_tui_completed_turn_assistant,
    write_skill_exchanges as write_tui_skill_exchanges,
    write_turn_user as write_tui_turn_user,
)

MAX_REPLY_ACC_CHARS = 24000
ACCENT_COLOR = "#6366f1"


def _global_config_path() -> Path:
    return get_app_paths().config_path


@dataclass
class _CompactPasteChunk:
    start: int
    end: int
    marker: str
    text: str


class _PasteTokenHighlighter(Highlighter):
    STYLE = "bold #6366f1 on #000000"

    def __init__(self, token_ranges_provider: Optional[Callable[[], List[Tuple[int, int]]]] = None) -> None:
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
        Binding("ctrl+k", "kill_to_end", show=False),
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
        self._compact_paste_chunks: List[_CompactPasteChunk] = []
        self._last_value = self.value
        if isinstance(self.highlighter, _PasteTokenHighlighter):
            self.highlighter._token_ranges_provider = self._highlighted_placeholder_ranges

    def _highlighted_placeholder_ranges(self) -> List[Tuple[int, int]]:
        return [(chunk.start, chunk.end) for chunk in self._compact_paste_chunks]

    @staticmethod
    def _paste_marker(text: str) -> str:
        return f"[Pasted {len(text)} chars]"

    @staticmethod
    def _changed_span(old_value: str, new_value: str) -> Tuple[int, int, int]:
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
        updated: List[_CompactPasteChunk] = []
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

    def sync_paste_placeholders(self, value: Optional[str] = None) -> None:
        self._sync_chunk_ranges(self.value if value is None else value)

    def expanded_value(self, value: Optional[str] = None) -> str:
        self.sync_paste_placeholders(self.value if value is None else value)
        expanded = self.value if value is None else value
        for chunk in sorted(self._compact_paste_chunks, key=lambda item: item.start, reverse=True):
            if chunk.start < 0 or chunk.end > len(expanded):
                continue
            if expanded[chunk.start : chunk.end] != chunk.marker:
                continue
            expanded = f"{expanded[:chunk.start]}{chunk.text}{expanded[chunk.end:]}"
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


class AlphanusTUI(App):
    TITLE = "Alphanus"

    CSS = ALPHANUS_TUI_CSS

    BINDINGS = [
        Binding("ctrl+c", "quit", show=False),
        Binding("ctrl+d", "quit", show=False),
        Binding("escape", "handle_esc", show=False),
        Binding("f1", "show_keymap", show=False),
        Binding("f2", "toggle_details", show=False),
        Binding("f3", "toggle_thinking", show=False),
        Binding("pageup", "scroll_up", show=False),
        Binding("pagedown", "scroll_down", show=False),
        Binding("ctrl+backspace", "remove_last_attachment", show=False),
        Binding("ctrl+shift+backspace", "clear_attachments", show=False),
        Binding("ctrl+f", "open_file_picker", show=False),
        Binding("ctrl+g", "focus_input", show=False),
        Binding("ctrl+p", "open_command_palette", show=False),
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
        init_tui_shell_state(self, agent=agent, debug=debug)

    def compose(self) -> ComposeResult:
        yield from compose_tui_shell(self, chat_input_cls=ChatInput, transcript_view_cls=TranscriptView)

    def _timing_config(self):
        timing = getattr(self, "_ui_timing", None)
        if timing is None:
            timing = UiRuntimeConfig.from_config({}).timing
            self._ui_timing = timing
        return timing

    def _stream_runtime_state(self) -> StreamRuntimeState:
        state = getattr(self, "_stream_runtime", None)
        if state is None:
            state = StreamRuntimeState()
            self._stream_runtime = state
        return state

    def on_mount(self) -> None:
        self.thinking = bool(self.agent.config.get("agent", {}).get("enable_thinking", True))
        self.set_interval(0.1, self._tick)
        self._install_stream_drain_timer()
        self._sync_tree_cursor()
        self._apply_sidebar_layout(self.size.width)
        self._apply_focus_classes()
        self._update_topbar()
        self._update_status1()
        self._update_status2()
        self._update_footer_separator()
        self._update_sidebar()
        self._update_pending_attachments()
        self._maybe_refresh_model_status(force=True)
        self._maybe_start_startup_readiness_poll()
        self.query_one(ChatInput).focus()
        self.call_after_refresh(self._open_startup_session_manager)

    def on_resize(self, event) -> None:
        self._apply_sidebar_layout(event.size.width)
        if not self.query_one("#sidebar", Vertical).display and self._focused_panel == "tree":
            self._focused_panel = "chat"
        self._schedule_resize_redraw()

    def _apply_sidebar_layout(self, width: int) -> None:
        sidebar = self.query_one("#sidebar", Vertical)
        target_width = self._sidebar_target_width(width)
        sidebar.display = target_width > 0
        if target_width > 0 and hasattr(sidebar, "styles"):
            sidebar.styles.width = target_width

    @staticmethod
    def _sidebar_target_width(width: int) -> int:
        if width < 120:
            return 0
        if width < 140:
            return 32
        return 38

    def _schedule_resize_redraw(self) -> None:
        if self._resize_redraw_pending:
            return
        self._resize_redraw_pending = True
        self.call_after_refresh(self._run_resize_redraw)

    def _run_resize_redraw(self) -> None:
        self._resize_redraw_pending = False
        self._redraw_after_resize()

    def _refresh_command_popup_for_resize(self) -> None:
        if not self._command_popup_active():
            return
        try:
            self._refresh_command_popup(self.query_one(ChatInput).value)
        except NoMatches:
            return

    def _redraw_after_resize(self) -> None:
        self._apply_focus_classes()
        self._update_topbar()
        self._update_status1()
        self._update_status2()
        self._update_footer_separator()
        self._update_sidebar()
        self._update_pending_attachments()
        self._refresh_transcript_after_resize()
        if self.streaming:
            self._refresh_deferred_partial()
        self._refresh_command_popup_for_resize()

    def _refresh_transcript_after_resize(self) -> None:
        log = self._log()
        anchor = self._capture_scroll_anchor()
        log.refresh_for_width_change()
        if anchor is not None:
            self.call_after_refresh(lambda current=anchor: self._restore_scroll_anchor(current))

    def _new_conv_tree(self) -> ConvTree:
        return ConvTree(
            compact_inactive_branches=self._tree_compaction_enabled,
            inactive_assistant_char_limit=self._inactive_assistant_char_limit,
            inactive_tool_argument_char_limit=self._inactive_tool_argument_char_limit,
            inactive_tool_content_char_limit=self._inactive_tool_content_char_limit,
        )

    def _activate_session_state(self, session: ChatSession) -> None:
        activate_tui_session_state(self, session)

    def _save_active_session(self, rename_to: Optional[str] = None) -> ChatSession:
        return save_tui_session(self, rename_to=rename_to)

    def _cmd_sessions(self) -> None:
        self._open_session_manager()

    def _open_session_manager(self) -> None:
        sessions = self._session_store.list_sessions()
        self.push_screen(
            SessionManagerModal(sessions, self._session_id),
            self._on_session_manager_close,
        )

    def _load_session_from_manager(self, session_id: str) -> ChatSession:
        return load_tui_session_from_manager(self, session_id)

    def _current_session_is_blank(self) -> bool:
        return is_blank_tui_session(self)

    def _open_new_session(self, title: str = "") -> ChatSession:
        return open_tui_new_session(self, title)

    def _open_startup_session_manager(self) -> None:
        if self._startup_session_prompt_opened:
            return
        self._startup_session_prompt_opened = True
        self._open_session_manager()

    def _on_session_manager_close(self, result: Optional[Dict[str, str]]) -> None:
        on_tui_session_manager_close(self, result)

    def _delete_session_from_manager(self, session_id: str) -> None:
        delete_tui_session_from_manager(self, session_id)

    def _switch_to_session(self, session: ChatSession, *, clear_pending: bool = True) -> None:
        switch_tui_session(self, session, clear_pending=clear_pending)

    def _tree_rows(self) -> List[Tuple[str, str, bool]]:
        return render_tree_rows(self.conv_tree, width=30)

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

    def _context_window_tokens(self) -> Optional[int]:
        return self._status_runtime.model_context_window

    def _reset_context_usage(self) -> None:
        self._last_model_context_tokens = None

    def _update_context_usage_from_payload(self, usage: Dict[str, Any]) -> None:
        for key in ("prompt_tokens", "input_tokens", "prompt_eval_count"):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                self._last_model_context_tokens = max(0, int(value))
                break
        self._update_topbar()

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

    def action_focus_input(self) -> None:
        self._set_focused_panel("input")

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

    def _show_keyboard_shortcuts(self) -> None:
        sections = [
            (
                "Keymap",
                [
                    ("F1 / ?", "Show keyboard shortcuts"),
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
                    ("Ctrl+K", "Delete to end of line"),
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
        show_tui_keyboard_shortcuts(self, sections=sections)

    def action_show_keymap(self) -> None:
        self._show_keyboard_shortcuts()

    def _toggle_tool_details(self) -> None:
        self._show_tool_details = not self._show_tool_details
        self._write_info(f"Live tool details {'shown' if self._show_tool_details else 'hidden'}")

    def action_toggle_details(self) -> None:
        self._toggle_tool_details()

    def _toggle_thinking_mode(self) -> None:
        self.thinking = not self.thinking
        self._write_info(f"Thinking {'enabled' if self.thinking else 'disabled'}")

    def action_toggle_thinking(self) -> None:
        self._toggle_thinking_mode()

    def action_open_command_palette(self) -> None:
        if self.streaming or self._await_shell_confirm:
            return
        chat_input = self.query_one(ChatInput)
        self._set_focused_panel("input")
        current = chat_input.value
        if current.strip() and not current.lstrip().startswith("/"):
            self._write_info("Command palette opens on an empty draft or an existing slash command.")
            return
        if not current.strip():
            chat_input.value = "/"
            chat_input.cursor_position = len(chat_input.value)
        self._refresh_command_popup(chat_input.value)

    def action_open_file_picker(self) -> None:
        if self.streaming or self._await_shell_confirm:
            return
        self._hide_command_popup()
        self._set_focused_panel("input")
        self._open_attachment_picker(".")

    def action_remove_last_attachment(self) -> None:
        if self.streaming or self._await_shell_confirm:
            return
        if not self.pending:
            return
        removed_path, _removed_kind = self.pending.pop()
        self._update_pending_attachments()
        self._update_status1()
        self._write_info(f"Removed attachment: {Path(removed_path).name}")

    def action_clear_attachments(self) -> None:
        if self.streaming or self._await_shell_confirm:
            return
        if not self.pending:
            return
        count = len(self.pending)
        self.pending.clear()
        self._update_pending_attachments()
        self._update_status1()
        self._write_info(f"Cleared {count} attachment{'s' if count != 1 else ''}.")

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
        if (
            chat_input.has_focus
            and not self.streaming
            and not self._await_shell_confirm
            and not self._command_popup_active()
            and event.key.lower() == "backspace"
            and not chat_input.value
            and self.pending
        ):
            self.action_remove_last_attachment()
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
        self._maybe_refresh_model_status()
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
        except NoMatches:
            pass
        if value:
            self._hide_command_popup()
        self._update_status2()

    def watch_thinking(self, value: bool) -> None:
        self._update_status1()
        self._update_status2()

    def _log(self) -> TranscriptView:
        return self.query_one("#chat-log", TranscriptView)

    def _scroll(self) -> ScrollableContainer:
        return self.query_one("#chat-scroll", ScrollableContainer)

    def _partial(self) -> Static:
        return self.query_one("#partial", Static)

    def _partial_measurement_width(self) -> int:
        return tui_partial_measurement_width(self)

    def _set_partial_renderable(
        self,
        renderable: Optional[RenderableType],
        *,
        visible: Optional[bool] = None,
    ) -> None:
        tui_set_partial_renderable(self, renderable, visible=visible)

    def _remeasure_partial_line_count(self, width: Optional[int] = None) -> int:
        return tui_remeasure_partial_line_count(self, width, count_lines=count_renderable_lines)

    def _cached_partial_line_count(self) -> int:
        return tui_cached_partial_line_count(self, count_lines=count_renderable_lines)

    def _current_partial_line_count(self) -> int:
        return tui_current_partial_line_count(self, count_lines=count_renderable_lines)

    def _command_popup(self) -> Vertical:
        return self.query_one("#command-popup", Vertical)

    def _command_options(self) -> OptionList:
        return self.query_one("#command-options", OptionList)

    def _append_transcript_entry(self, entry: TranscriptEntry) -> None:
        tui_append_transcript_entry(self, entry)

    def _write(self, markup: str) -> None:
        tui_write_markup(self, markup)

    @staticmethod
    def _bar_renderable(
        renderable: RenderableType,
        color: str,
        *,
        content_indent: int = 0,
        continuation_indent: Optional[int] = None,
    ):
        return tui_bar_renderable(
            renderable,
            color,
            content_indent=content_indent,
            continuation_indent=continuation_indent,
        )

    @staticmethod
    def _line_indents(line: str, *, base_indent: int = 2) -> tuple[int, int]:
        return tui_line_indents(line, base_indent=base_indent)

    def _write_user_bar_line(self, markup: str = "", *, content_indent: int = 0) -> None:
        tui_write_user_bar_line(self, markup, content_indent=content_indent)

    def _write_assistant_bar_line(self, markup: str = "", *, content_indent: int = 0) -> None:
        tui_write_assistant_bar_line(self, markup, content_indent=content_indent)

    def _write_assistant_bar_renderable(self, renderable: RenderableType, *, content_indent: int = 0) -> None:
        tui_write_assistant_bar_renderable(self, renderable, content_indent=content_indent)

    def _write_user_bar_wrapped_line(self, line: str) -> None:
        tui_write_user_bar_wrapped_line(self, line)

    def _write_assistant_bar_wrapped_line(self, line: str, markup: str) -> None:
        tui_write_assistant_bar_wrapped_line(self, line, markup)

    def _write_renderable(self, renderable, indent: int = 2) -> None:
        tui_write_renderable(self, renderable, indent=indent)

    def _syntax_renderable(self, code: str, language: Optional[str]):
        return tui_syntax_renderable(code, language)

    def _code_panel_renderable(self, code: str, language: Optional[str]):
        return tui_code_panel_renderable(self, code, language)

    def _reasoning_panel_renderable(self, text: str):
        return tui_reasoning_panel_renderable(text)

    def _tool_event_panel(
        self,
        title: str,
        title_color: str,
        border_color: str,
        name: str,
        detail: str = "",
    ):
        return tui_tool_event_panel(title, title_color, border_color, name, detail)

    def _tool_lifecycle_panel(
        self,
        name: str,
        detail: str,
        *,
        ok: bool,
    ):
        return tui_tool_lifecycle_panel(self, name, detail, ok=ok)

    def _update_tool_call_partial(self, name: str, detail: str = "") -> None:
        tui_update_tool_call_partial(self, name, detail)

    def _write_tool_lifecycle_block(self, name: str, ok: bool, detail: str = "") -> None:
        tui_write_tool_lifecycle_block(self, name, ok, detail)

    def _show_tool_result_line(self, name: str, ok: bool) -> bool:
        return tui_show_tool_result_line(self, name, ok)

    def _take_pending_tool_detail(self, name: str) -> str:
        return tui_take_pending_tool_detail(self, name)

    def _remember_code_block(self, code: str, language: Optional[str]) -> int:
        return tui_remember_code_block(self, code, language)

    def _write_code_block(self, lines: List[str], language: Optional[str], content_indent: int = 2) -> None:
        tui_write_code_block(self, lines, language, content_indent)

    def _render_static_markdown(self, text: str) -> None:
        tui_render_static_markdown(self, text)

    def _reset_fence_state(self) -> None:
        tui_reset_fence_state(self)

    @staticmethod
    def _is_fence_line(line: str) -> bool:
        return tui_is_fence_line(line)

    def _flush_fence_block(self) -> None:
        tui_flush_fence_block(self)

    def _render_content_line(self, line: str) -> None:
        tui_render_content_line(self, line)

    def _update_partial_content(self) -> None:
        tui_update_partial_content(self)

    def _update_live_preview_partial(self, lines: List[str], language: Optional[str]) -> None:
        tui_update_live_preview_partial(self, lines, language)

    def _defer_live_preview_partial(self, lines: List[str], language: Optional[str]) -> None:
        tui_defer_live_preview_partial(self, lines, language)

    def _clear_partial_preview(self) -> None:
        tui_clear_partial_preview(self)

    def _is_near_bottom(self, threshold: float = 1.0) -> bool:
        return tui_is_near_bottom(self, threshold=threshold)

    def _capture_scroll_anchor(self) -> Optional[ScrollAnchor]:
        return tui_capture_scroll_anchor(self)

    def _restore_scroll_anchor(self, anchor: Optional[ScrollAnchor]) -> None:
        tui_restore_scroll_anchor(self, anchor)

    def _maybe_scroll_end(self, force: bool = False) -> None:
        tui_maybe_scroll_end(self, force=force)

    def _write_info(self, text: str) -> None:
        tui_write_info(self, text, accent_color=ACCENT_COLOR)

    def _write_error(self, text: str) -> None:
        tui_write_error(self, text)

    def _write_section_heading(self, title: str, color: str = ACCENT_COLOR) -> None:
        tui_write_section_heading(self, title, color=color)

    def _write_detail_line(self, label: str, value: str, *, value_markup: bool = False) -> None:
        tui_write_detail_line(self, label, value, accent_color=ACCENT_COLOR, value_markup=value_markup)

    def _write_indexed_dim_lines(self, rows: List[str], *, color: str = ACCENT_COLOR, allow_markup: bool = False) -> None:
        tui_write_indexed_dim_lines(self, rows, color=color, allow_markup=allow_markup)

    def _write_command_action(self, text: str, *, icon: str = "•", color: str = ACCENT_COLOR) -> None:
        tui_write_command_action(self, text, color=color, icon=icon)

    def _write_command_row(self, command: str, desc: str, *, col: int) -> None:
        tui_write_command_row(self, command, desc, col=col, accent_color=ACCENT_COLOR)

    def _write_muted_lines(self, rows: List[str]) -> None:
        tui_write_muted_lines(self, rows)

    def _write_usage(self, usage: str) -> bool:
        return tui_write_usage(self, usage)

    def _ensure_command_gap(self) -> None:
        tui_ensure_command_gap(self)

    def _reload_skills(self) -> bool:
        self.agent.reload_skills()
        self._loaded_skill_ids = [skill.id for skill in self.agent.skill_runtime.skills_by_ids(self._loaded_skill_ids)]
        self._save_active_session()
        self._write_info("Reloaded skills")
        return True

    @property
    def _reply_acc(self) -> str:
        parts = getattr(self, "_reply_acc_parts", [])
        return "".join(parts)

    @_reply_acc.setter
    def _reply_acc(self, value: str) -> None:
        text = str(value or "")
        self._reply_acc_parts = [text] if text else []
        self._reply_acc_len = len(text)

    def _append_reply_token(self, token: str) -> None:
        if not token:
            return
        if getattr(self, "_reply_acc_len", 0) >= MAX_REPLY_ACC_CHARS:
            return
        remaining = MAX_REPLY_ACC_CHARS - self._reply_acc_len
        chunk = token[:remaining]
        if not chunk:
            return
        self._reply_acc_parts.append(chunk)
        self._reply_acc_len += len(chunk)

    @property
    def _reasoning_open(self) -> bool:
        return bool(self._stream_runtime_state().text.reasoning_open)

    @_reasoning_open.setter
    def _reasoning_open(self, value: bool) -> None:
        self._stream_runtime_state().text.reasoning_open = bool(value)

    @property
    def _content_open(self) -> bool:
        return bool(self._stream_runtime_state().text.content_open)

    @_content_open.setter
    def _content_open(self, value: bool) -> None:
        self._stream_runtime_state().text.content_open = bool(value)

    @property
    def _buf_r(self) -> str:
        return str(self._stream_runtime_state().text.reasoning_buffer)

    @_buf_r.setter
    def _buf_r(self, value: str) -> None:
        self._stream_runtime_state().text.reasoning_buffer = str(value)

    @property
    def _buf_c(self) -> str:
        return str(self._stream_runtime_state().text.content_buffer)

    @_buf_c.setter
    def _buf_c(self, value: str) -> None:
        self._stream_runtime_state().text.content_buffer = str(value)

    @property
    def _in_fence(self) -> bool:
        return bool(self._stream_runtime_state().text.in_fence)

    @_in_fence.setter
    def _in_fence(self, value: bool) -> None:
        self._stream_runtime_state().text.in_fence = bool(value)

    @property
    def _fence_lang(self) -> Optional[str]:
        return self._stream_runtime_state().text.fence_lang

    @_fence_lang.setter
    def _fence_lang(self, value: Optional[str]) -> None:
        self._stream_runtime_state().text.fence_lang = value

    @property
    def _fence_lines(self) -> List[str]:
        return self._stream_runtime_state().text.fence_lines

    @_fence_lines.setter
    def _fence_lines(self, value: List[str]) -> None:
        self._stream_runtime_state().text.fence_lines = list(value)

    def _is_tool_trace_line(self, line: str) -> bool:
        s = line.strip().lower()
        return "tool call:" in s

    def _update_status1(self) -> None:
        text = status_right_markup(
            model_name=self._status_runtime.model_name,
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

    def _current_model_refresh_interval(self) -> float:
        return tui_model_refresh_interval(self)

    def _should_startup_readiness_poll(self) -> bool:
        return should_tui_startup_readiness_poll(self)

    def _maybe_start_startup_readiness_poll(self) -> None:
        start_tui_startup_readiness_poll(self)

    @work(thread=True, exclusive=True)
    def _startup_readiness_worker(self) -> None:
        self.agent.ensure_ready(timeout_s=self.agent.readiness_timeout_s)
        status = self.agent.get_model_status()
        self.call_from_thread(self._finish_startup_readiness_poll, status)

    def _finish_startup_readiness_poll(self, status: ModelStatus) -> None:
        finish_tui_startup_readiness_poll(self, status)

    def _maybe_refresh_model_status(self, *, force: bool = False) -> None:
        maybe_refresh_tui_model_status(self, force=force)

    @work(thread=True, exclusive=True)
    def _refresh_model_status_worker(self) -> None:
        status = self.agent.refresh_model_status(timeout_s=min(self.agent.connect_timeout_s, 2.0), force=True)
        self.call_from_thread(self._apply_model_status_refresh, status)

    def _apply_model_status_refresh(self, status: ModelStatus) -> None:
        apply_tui_model_status(self, status)

    def _update_status2(self) -> None:
        update_tui_status2(self)

    def _update_topbar(self) -> None:
        update_tui_topbar(self)

    def _update_input_placeholder(self) -> None:
        self.query_one(ChatInput).placeholder = (
            "Type to start branch…" if self.conv_tree._pending_branch else "Type a message…"
        )

    def _update_footer_separator(self) -> None:
        try:
            separator = self.query_one("#footer-sep", Static)
            width = int(getattr(separator.region, "width", 0) or 0)
            if width <= 0:
                width = int(getattr(separator.size, "width", 0) or 0)
            separator.update("─" * max(1, width))
        except (NoMatches, AttributeError):
            return

    def _update_pending_attachments(self) -> None:
        try:
            markup = self._pending_attachment_markup()
            bar = self.query_one("#attachment-bar", Static)
            bar.update(markup)
        except (NoMatches, AttributeError):
            return

    def _pending_attachment_markup(self) -> str:
        return tui_pending_attachment_markup(self)

    def _workspace_root(self) -> Path:
        return workspace_tui_root(self)

    def _home_root(self) -> Path:
        return home_tui_root(self)

    def _attachment_root_path(self, root_id: str) -> Path:
        return attachment_tui_root_path(self, root_id)

    def _attachment_root_label(self, root_id: str) -> str:
        return attachment_tui_root_label(root_id)

    def _root_relative_label(self, path: Path, root: Path) -> str:
        return tui_root_relative_label(path, root)

    def _resolve_attachment_path(self, raw_path: str) -> Path:
        return resolve_tui_attachment_path(self, raw_path)

    def _attach_file_path(self, path: str | Path) -> bool:
        return attach_tui_file_path(self, path)

    def _attachment_picker_items(self, relative_dir: str = ".", *, root_id: str = "workspace") -> List[PickerItem]:
        return attachment_tui_picker_items(self, relative_dir, root_id=root_id, accent_color=ACCENT_COLOR)

    def _open_attachment_picker(self, relative_dir: str = ".", root_id: str = "workspace") -> None:
        open_tui_attachment_picker(self, relative_dir, root_id=root_id, accent_color=ACCENT_COLOR)

    def _on_attachment_picker_close(self, root_id: str, current_dir: str, result: Optional[Dict[str, str]]) -> None:
        on_tui_attachment_picker_close(self, root_id, current_dir, result, accent_color=ACCENT_COLOR)

    def _refresh_command_popup(self, value: str) -> None:
        refresh_tui_command_popup(self, value, chat_input_cls=ChatInput)

    def _hide_command_popup(self) -> None:
        hide_tui_command_popup(self)

    def _command_popup_active(self) -> bool:
        return tui_command_popup_active(self)

    def _move_command_selection(self, delta: int) -> None:
        move_tui_command_selection(self, delta)

    def _accept_command_selection(self) -> bool:
        return accept_tui_command_selection(self, chat_input_cls=ChatInput)

    def _should_accept_popup_on_enter(self, text: str) -> bool:
        _ = text
        return should_tui_accept_popup_on_enter(self, chat_input_cls=ChatInput)

    @staticmethod
    def _config_for_editor(config: Dict[str, Any]) -> Dict[str, Any]:
        return config_for_editor_view(config)

    def _open_config_editor(self) -> None:
        warnings: List[str] = []
        try:
            raw = load_or_create_global_config(_global_config_path(), warnings=warnings)
        except (OSError, ValueError) as exc:
            self._write_error(f"Config load failed: {exc}")
            return
        safe = self._config_for_editor(raw if isinstance(raw, dict) else {})
        text = json.dumps(safe, indent=2) + "\n"
        self.push_screen(ConfigEditorModal(_global_config_path(), text), self._on_config_editor_close)
        for warning in warnings:
            self._write_info(f"Config warning: {warning}")

    def _merge_live_config(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        return merge_tui_live_config(base, updates)

    def _install_stream_drain_timer(self) -> None:
        install_tui_stream_drain_timer(self)

    def _apply_tui_config(self) -> None:
        apply_tui_runtime_config(self)

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
        except ValueError as exc:
            self._write_error(f"Config save failed: {exc}")
            return

        cleaned = self._config_for_editor(normalized)
        _global_config_path().write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
        merged = self._merge_live_config(self.agent.config, normalized)
        self.agent.reload_config(merged)
        self.thinking = bool(merged.get("agent", {}).get("enable_thinking", self.thinking))
        self._ui_config = UiRuntimeConfig.from_config(merged)
        self._ui_timing = self._ui_config.timing
        refreshed_status = self.agent.get_model_status()
        self._status_runtime = StatusRuntimeState(
            model_status=refreshed_status,
            model_name=refreshed_status.model_name if refreshed_status.state != "offline" else None,
            model_context_window=(
                refreshed_status.context_window
                if isinstance(refreshed_status.context_window, int) and refreshed_status.context_window > 0
                else None
            ),
        )
        self._update_topbar()
        self._apply_tui_config()
        self._maybe_refresh_model_status(force=True)
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
        update_tui_sidebar(self)

    @staticmethod
    def _sidebar_render_width(sidebar: Vertical) -> int:
        return tui_sidebar_render_width(sidebar)

    def _write_turn_user(self, turn: Turn) -> None:
        write_tui_turn_user(self, turn, accent_color=ACCENT_COLOR)

    def _write_skill_exchanges(self, turn: Turn) -> None:
        write_tui_skill_exchanges(self, turn)

    def _write_completed_turn_asst(self, turn: Turn) -> None:
        write_tui_completed_turn_assistant(self, turn)

    def _rebuild_viewport(self, *, preserve_scroll: bool = False) -> None:
        rebuild_tui_viewport(self, preserve_scroll=preserve_scroll)

    def _start_stream(self, turn: Turn, user_input: str, attachment_paths: List[str]) -> None:
        start_tui_turn_stream(self, turn, user_input, attachment_paths)

    @work(thread=True, exclusive=True)
    def _stream_worker(
        self,
        turn_id: str,
        history_messages: List[Dict[str, Any]],
        user_input: str,
        branch_labels: List[str],
        attachment_paths: List[str],
        loaded_skill_ids: List[str],
        thinking: bool,
        stop_event: threading.Event,
    ) -> None:
        def on_event(event: Dict[str, Any]) -> None:
            enqueue_tui_stream_event(self, event)

        result = self.agent.run_turn(
            history_messages=history_messages,
            user_input=user_input,
            thinking=thinking,
            branch_labels=branch_labels,
            attachments=attachment_paths,
            loaded_skill_ids=loaded_skill_ids,
            stop_event=stop_event,
            on_event=on_event,
            confirm_shell=self._confirm_shell_command,
        )
        self.call_from_thread(self._drain_stream_event_queue)
        self.call_from_thread(self._on_stream_end, turn_id, result)

    def _visible_reasoning_text(self, text: str) -> str:
        filtered_lines = [line for line in visible_tui_reasoning_text(text).splitlines() if not self._is_tool_trace_line(line)]
        return "\n".join(filtered_lines).strip()

    def _flush_reasoning_buffer(self) -> None:
        flush_tui_reasoning_buffer(self)

    def _close_reasoning_section(self) -> bool:
        return close_tui_reasoning_section(self)

    def _refresh_deferred_partial(self) -> None:
        refresh_tui_deferred_partial(self)

    def _flush_content_buffer(self, include_partial: bool = False, *, update_partial: bool = True) -> None:
        flush_tui_content_buffer(self, include_partial=include_partial, update_partial=update_partial)

    def _handle_content_token(self, token: str, *, update_partial: bool = True) -> None:
        handle_tui_content_token(self, token, update_partial=update_partial)

    def _drain_stream_event_queue(self) -> None:
        drain_tui_stream_events(self)

    def _on_agent_event(self, event: Dict[str, Any]) -> None:
        on_tui_agent_event(self, event)

    def _on_stream_end(self, turn_id: str, result: AgentTurnResult) -> None:
        finish_tui_turn_stream(self, turn_id, result)

    def _confirm_shell_command(self, command: str) -> bool:
        return confirm_tui_shell_command(self, command)

    def _begin_shell_confirm(self, command: str, event: threading.Event, holder: Dict[str, bool]) -> None:
        begin_tui_shell_confirm(self, command, event, holder, esc=esc)

    def _expire_shell_confirm(self, event: threading.Event) -> None:
        expire_tui_shell_confirm(self, event)

    def _finish_shell_confirm(self, approved: bool) -> None:
        finish_tui_shell_confirm(self, approved)

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
        on_tui_input_submitted(self, event, chat_input_cls=ChatInput)

    @on(Input.Changed, "#chat-input")
    def on_input_changed(self, event: Input.Changed) -> None:
        chat_input = self.query_one(ChatInput)
        chat_input.sync_paste_placeholders(event.value)
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
        if not select_tui_command_option(self, index, chat_input_cls=ChatInput):
            return
        self.query_one(ChatInput).focus()

    def action_handle_esc(self) -> None:
        action_tui_handle_esc(self, chat_input_cls=ChatInput)

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
        return handle_tui_command(self, text)

    def _cmd_help(self) -> None:
        cmd_tui_help(self, help_sections=HELP_SECTIONS, accent_color=ACCENT_COLOR)

    def _cmd_tree(self) -> None:
        cmd_tui_tree(self, accent_color=ACCENT_COLOR)

    def _cmd_skills(self) -> None:
        cmd_tui_skills(self, accent_color=ACCENT_COLOR)

    def _load_skill_into_session(self, skill_id: str) -> bool:
        return load_tui_skill_into_session(self, skill_id)

    def _unload_skill_from_session(self, skill_id: str) -> bool:
        return unload_tui_skill_from_session(self, skill_id)

    def _cmd_skill(self, arg: str) -> bool:
        return cmd_tui_skill(self, arg)

    def _cmd_memory(self, arg: str) -> bool:
        return cmd_tui_memory(self, arg)

    def _cmd_context(self, arg: str) -> bool:
        return cmd_tui_context(self, arg)

    def _cmd_doctor(self) -> None:
        cmd_tui_doctor(self, accent_color=ACCENT_COLOR)

    def _cmd_report(self, arg: str) -> bool:
        return cmd_tui_report(self, arg)

    def _cmd_workspace(self, arg: str) -> bool:
        return cmd_tui_workspace(self, arg)

    def _cmd_code(self, arg: str) -> bool:
        return cmd_tui_code(self, arg)
