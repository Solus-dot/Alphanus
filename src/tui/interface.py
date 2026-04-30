from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.console import RenderableType
from rich.markup import escape as esc
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import Button, Input, OptionList, Static

from agent.core import Agent, AgentTurnResult
from agent.types import ModelStatus
from alphanus_paths import get_app_paths
from core.attachments import build_content, classify_attachment
from core.configuration import (
    config_for_editor_view,
    load_global_config,
)
from core.conv_tree import ConvTree, Turn
from core.runtime_config import UiRuntimeConfig
from core.sessions import ChatSession
from core.theme_catalog import DEFAULT_THEME_ID, normalize_theme_id
from tui.app_shell_runtime import compose_shell as compose_tui_shell
from tui.app_shell_runtime import initialize_shell_state as init_tui_shell_state
from tui.commands import (
    HELP_SECTIONS,
)
from tui.chat_input import ChatInput, _PasteTokenHighlighter
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
from tui.navigation_runtime import (
    action_tree_bottom as action_tui_tree_bottom,
    action_tree_down as action_tui_tree_down,
    action_tree_open as action_tui_tree_open,
    action_tree_top as action_tui_tree_top,
    action_tree_up as action_tui_tree_up,
    apply_focus_classes as apply_tui_focus_classes,
    apply_sidebar_layout as apply_tui_sidebar_layout,
    focus_next_panel as action_tui_focus_next_panel,
    focus_prev_panel as action_tui_focus_prev_panel,
    move_tree_sibling as move_tui_tree_sibling,
    set_focused_panel as set_tui_focused_panel,
    sync_tree_cursor as sync_tui_tree_cursor,
)
from tui.interaction_runtime import (
    action_handle_esc as action_tui_handle_esc,
    begin_shell_confirm as begin_tui_shell_confirm,
    confirm_shell_command as confirm_tui_shell_command,
    expire_shell_confirm as expire_tui_shell_confirm,
    finish_shell_confirm as finish_tui_shell_confirm,
    on_input_submitted as on_tui_input_submitted,
    on_session_name_close as on_tui_session_name_close,
    on_session_manager_close as on_tui_session_manager_close,
    show_keyboard_shortcuts as show_tui_keyboard_shortcuts,
)
from tui.interface_input_runtime import (
    handle_key as on_tui_key,
    on_config_editor_close as on_tui_config_editor_close,
    show_keyboard_shortcuts as show_tui_keyboard_shortcuts_sections,
)
from tui.popups import (
    CommandPaletteItem,
    CommandPaletteModal,
    CodeViewerModal,
    ConfigEditorModal,
    PickerItem,
    SelectionPickerModal,
    SessionNameModal,
    SessionManagerModal,
)
from tui.palette_theme_runtime import (
    build_global_palette_catalog as build_tui_global_palette_catalog,
    on_global_palette_close as on_tui_global_palette_close,
    on_theme_picker_close as on_tui_theme_picker_close,
    open_theme_picker as open_tui_theme_picker,
    persist_theme_preference as persist_tui_theme_preference,
    workspace_file_candidates as workspace_tui_file_candidates,
)
from tui.attachment_runtime import (
    attach_file_path as attach_tui_file_path,
    attachment_picker_items as attachment_tui_picker_items,
    open_attachment_picker as open_tui_attachment_picker,
    resolve_attachment_path as resolve_tui_attachment_path,
    root_relative_label as tui_root_relative_label,
    workspace_root as workspace_tui_root,
)
from tui.session_runtime import (
    activate_session_state as activate_tui_session_state,
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
from tui.transcript import ScrollAnchor, TranscriptView, count_renderable_lines
from tui.transcript_runtime import (
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
    pending_attachment_markup as tui_pending_attachment_markup,
    reasoning_panel_renderable as tui_reasoning_panel_renderable,
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
from tui.themes import ThemeSpec, available_theme_ids, default_theme_variables, fallback_color, theme_spec
from tui.ui_styles import ALPHANUS_TUI_CSS
from tui.view_runtime import (
    rebuild_viewport as rebuild_tui_viewport,
    update_sidebar as update_tui_sidebar,
    write_completed_turn_assistant as write_tui_completed_turn_assistant,
    write_skill_exchanges as write_tui_skill_exchanges,
    write_turn_user as write_tui_turn_user,
)

MAX_REPLY_ACC_CHARS = 24000
DEFAULT_ACCENT_COLOR = fallback_color("accent")
DEFAULT_PANEL_BG = fallback_color("panel_bg")
DEFAULT_TEXT_COLOR = fallback_color("text")
DEFAULT_SUBTLE_COLOR = fallback_color("subtle")
DEFAULT_SUCCESS_COLOR = fallback_color("success")
DEFAULT_WARNING_COLOR = fallback_color("warning")
DEFAULT_MUTED_COLOR = fallback_color("muted")


def _global_config_path() -> Path: return get_app_paths().config_path


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
        Binding("ctrl+k", "open_global_palette", show=False),
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

    def get_theme_variable_defaults(self) -> dict[str, str]: return default_theme_variables()

    def __init__(self, agent: Agent, debug: bool = False):
        super().__init__()
        init_tui_shell_state(self, agent=agent, debug=debug)
        self._chat_input_cls = ChatInput

    def compose(self) -> ComposeResult:
        yield from compose_tui_shell(self, chat_input_cls=ChatInput, transcript_view_cls=TranscriptView)

    def _timing_config(self):
        timing = getattr(self, "_ui_timing", None)
        if timing is None:
            timing = UiRuntimeConfig.from_config({}).timing
            self._ui_timing = timing
        return timing

    def _theme_id(self) -> str:
        current = str(getattr(self, "_active_theme_id", "") or "").strip().lower()
        if current:
            resolved, _ = normalize_theme_id(current, default=DEFAULT_THEME_ID)
            return resolved
        ui_cfg = getattr(self, "_ui_config", None)
        configured = str(getattr(ui_cfg, "theme", "") or "").strip().lower()
        resolved, _ = normalize_theme_id(configured, default=DEFAULT_THEME_ID)
        return resolved

    def _theme_spec(self) -> ThemeSpec: return theme_spec(self._theme_id())

    def _theme_color(self, key: str, default: str) -> str:
        spec = self._theme_spec()
        return str(spec.colors.get(key, default))

    def _register_themes(self) -> None:
        if getattr(self, "_themes_registered", False):
            return
        for theme_id in available_theme_ids():
            self.register_theme(theme_spec(theme_id).theme)
        self._themes_registered = True

    def _apply_theme(self, raw_theme_id: str) -> str:
        resolved, _ = normalize_theme_id(raw_theme_id, default=DEFAULT_THEME_ID)
        self._register_themes()
        self.theme = resolved
        self._active_theme_id = resolved
        style = f"bold {self._theme_color('accent', DEFAULT_ACCENT_COLOR)} on {self._theme_color('panel_bg', DEFAULT_PANEL_BG)}"
        _PasteTokenHighlighter.STYLE = style
        ChatInput.PASTE_TOKEN_STYLE = style
        try:
            chat_input = self.query_one(ChatInput)
        except Exception:
            chat_input = None
        if chat_input is not None:
            highlighter = getattr(chat_input, "highlighter", None)
            if isinstance(highlighter, _PasteTokenHighlighter):
                highlighter.STYLE = style
            chat_input.sync_paste_placeholders(chat_input.value)
        try:
            self.refresh_css(animate=False)
            self._redraw_after_resize()
        except NoMatches:
            pass
        return resolved

    def _apply_theme_from_config(self) -> str:
        configured = getattr(self, "_ui_config", None)
        configured_theme = str(getattr(configured, "theme", DEFAULT_THEME_ID))
        try:
            return self._apply_theme(configured_theme)
        except Exception:
            resolved, _ = normalize_theme_id(configured_theme, default=DEFAULT_THEME_ID)
            self._active_theme_id = resolved
            return resolved

    def _stream_runtime_state(self) -> StreamRuntimeState:
        state = getattr(self, "_stream_runtime", None)
        if state is None:
            state = StreamRuntimeState()
            self._stream_runtime = state
        return state

    def on_mount(self) -> None:
        self.thinking = bool(self.agent.config.get("agent", {}).get("enable_thinking", True))
        self._register_themes()
        self._apply_theme_from_config()
        self.set_interval(0.1, self._tick)
        install_tui_stream_drain_timer(self)
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
        if self._resize_redraw_pending:
            return
        self._resize_redraw_pending = True
        self.call_after_refresh(self._run_resize_redraw)

    def _apply_sidebar_layout(self, width: int) -> None: apply_tui_sidebar_layout(self, width)

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

    def _new_conv_tree(self) -> ConvTree: return ConvTree(compact_inactive_branches=self._tree_compaction_enabled, inactive_assistant_char_limit=self._inactive_assistant_char_limit, inactive_tool_argument_char_limit=self._inactive_tool_argument_char_limit, inactive_tool_content_char_limit=self._inactive_tool_content_char_limit)

    def _activate_session_state(self, session: ChatSession) -> None: activate_tui_session_state(self, session)

    def _save_active_session(self, rename_to: Optional[str] = None) -> ChatSession: return save_tui_session(self, rename_to=rename_to)

    def _set_collaboration_mode(self, mode: str, *, persist: bool = False) -> str:
        normalized = "plan" if str(mode or "").strip().lower() == "plan" else "execute"
        self._collaboration_mode = normalized
        self._update_status1()
        if persist:
            self._save_active_session()
            self._update_topbar()
        return normalized

    def _open_session_manager(self) -> None:
        sessions = self._session_store.list_sessions()
        self.push_screen(SessionManagerModal(sessions, self._session_id), self._on_session_manager_close)

    def _open_session_name_modal(self) -> None: self.push_screen(SessionNameModal(), self._on_session_name_close)

    def _load_session_from_manager(self, session_id: str) -> ChatSession: return load_tui_session_from_manager(self, session_id)
    def _open_new_session(self, title: str = "") -> ChatSession: return open_tui_new_session(self, title)

    def _open_startup_session_manager(self) -> None:
        if self._startup_session_prompt_opened:
            return
        self._startup_session_prompt_opened = True; self._open_session_manager()

    def _on_session_manager_close(self, result: Optional[Dict[str, str]]) -> None: on_tui_session_manager_close(self, result)
    def _on_session_name_close(self, result: Optional[Dict[str, str]]) -> None: on_tui_session_name_close(self, result)
    def _delete_session_from_manager(self, session_id: str) -> None: delete_tui_session_from_manager(self, session_id)
    def _switch_to_session(self, session: ChatSession, *, clear_pending: bool = True) -> None: switch_tui_session(self, session, clear_pending=clear_pending)
    def _sync_tree_cursor(self) -> None: sync_tui_tree_cursor(self)

    def _current_branch_name(self) -> str:
        if self.conv_tree._pending_branch and self.conv_tree._pending_branch_label:
            return self.conv_tree._pending_branch_label
        for turn in reversed(self.conv_tree.active_path):
            if turn.branch_root:
                return turn.label or "branch"
        return "root"

    def _context_tokens(self) -> Optional[int]: return self._last_model_context_tokens
    def _context_window_tokens(self) -> Optional[int]: return self._status_runtime.model_context_window
    def _reset_context_usage(self) -> None: self._last_model_context_tokens = None

    def _update_context_usage_from_payload(self, usage: Dict[str, Any]) -> None:
        for key in ("prompt_tokens", "input_tokens", "prompt_eval_count"):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                self._last_model_context_tokens = max(0, int(value))
                break
        self._update_topbar()

    def _apply_focus_classes(self) -> None: apply_tui_focus_classes(self)
    def _set_focused_panel(self, panel: str) -> None: set_tui_focused_panel(self, panel)
    def action_focus_next_panel(self) -> None: action_tui_focus_next_panel(self)
    def action_focus_prev_panel(self) -> None: action_tui_focus_prev_panel(self)
    def action_focus_chat(self) -> None: self._set_focused_panel("chat")
    def action_focus_tree(self) -> None: self._set_focused_panel("tree")
    def action_focus_input(self) -> None: self._set_focused_panel("input")
    def action_tree_down(self) -> None: action_tui_tree_down(self)
    def action_tree_up(self) -> None: action_tui_tree_up(self)
    def action_tree_top(self) -> None: action_tui_tree_top(self)
    def action_tree_bottom(self) -> None: action_tui_tree_bottom(self)
    def _move_tree_sibling(self, direction: int) -> None: move_tui_tree_sibling(self, direction)
    def action_tree_prev_sibling(self) -> None: self._move_tree_sibling(-1)
    def action_tree_next_sibling(self) -> None: self._move_tree_sibling(1)
    def action_tree_open(self) -> None: action_tui_tree_open(self)
    def _show_keyboard_shortcuts(self) -> None: show_tui_keyboard_shortcuts_sections(self, show_fn=show_tui_keyboard_shortcuts)
    def action_show_keymap(self) -> None: self._show_keyboard_shortcuts()

    def action_toggle_details(self) -> None:
        self._show_tool_details = not self._show_tool_details; self._write_info(f"Live tool details {'shown' if self._show_tool_details else 'hidden'}")

    def action_toggle_thinking(self) -> None:
        self.thinking = not self.thinking; self._write_info(f"Thinking {'enabled' if self.thinking else 'disabled'}")

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

    def action_open_global_palette(self) -> None:
        if self.streaming or self._await_shell_confirm:
            return
        self._hide_command_popup(); self._set_focused_panel("input")
        items, actions = self._build_global_palette_catalog(); self._global_palette_actions = actions
        self.push_screen(CommandPaletteModal(items=items), self._on_global_palette_close)

    def action_open_file_picker(self) -> None:
        if self.streaming or self._await_shell_confirm:
            return
        self._hide_command_popup(); self._set_focused_panel("input"); self._open_attachment_picker(".")

    def action_remove_last_attachment(self) -> None:
        if self.streaming or self._await_shell_confirm:
            return
        if not self.pending:
            return
        removed_path, _removed_kind = self.pending.pop()
        self._update_pending_attachments(); self._update_status1(); self._write_info(f"Removed attachment: {Path(removed_path).name}")

    def action_clear_attachments(self) -> None:
        if self.streaming or self._await_shell_confirm:
            return
        if not self.pending:
            return
        count = len(self.pending)
        self.pending.clear()
        self._update_pending_attachments(); self._update_status1(); self._write_info(f"Cleared {count} attachment{'s' if count != 1 else ''}.")

    def _apply_tree_compaction_policy(self, tree: ConvTree) -> ConvTree:
        tree.set_compaction_policy(
            enabled=self._tree_compaction_enabled,
            inactive_assistant_char_limit=self._inactive_assistant_char_limit,
            inactive_tool_argument_char_limit=self._inactive_tool_argument_char_limit,
            inactive_tool_content_char_limit=self._inactive_tool_content_char_limit,
        )
        return tree

    def on_key(self, event) -> None: on_tui_key(self, event, chat_input_cls=ChatInput)

    def _tick(self) -> None:
        self._maybe_refresh_model_status()
        if self._esc_pending and time.monotonic() - self._esc_ts > 3.0:
            self._esc_pending = False; self._update_status2(); return
        if self.streaming:
            if self._auto_follow_stream and not self._is_near_bottom():
                self._auto_follow_stream = False; self._update_status2()
            elif not self._auto_follow_stream and self._is_near_bottom():
                self._auto_follow_stream = True; self._update_status2()
        if self.streaming and not self._esc_pending: self._spin_i += 1; self._update_status2()

    def watch_streaming(self, value: bool) -> None:
        self.query_one(ChatInput).disabled = value
        try:
            self.query_one("#attach-file", Button).disabled = value
        except NoMatches:
            pass
        if value:
            self._hide_command_popup()
        self._update_status2()

    def watch_thinking(self, value: bool) -> None: self._update_status1(); self._update_status2()

    def _log(self) -> TranscriptView: return self.query_one("#chat-log", TranscriptView)
    def _scroll(self) -> ScrollableContainer: return self.query_one("#chat-scroll", ScrollableContainer)
    def _partial(self) -> Static: return self.query_one("#partial", Static)
    def _set_partial_renderable(self, renderable: Optional[RenderableType], *, visible: Optional[bool] = None) -> None: tui_set_partial_renderable(self, renderable, visible=visible)

    def _cached_partial_line_count(self) -> int: return tui_cached_partial_line_count(self, count_lines=count_renderable_lines)
    def _current_partial_line_count(self) -> int: return tui_current_partial_line_count(self, count_lines=count_renderable_lines)
    def _command_popup(self) -> Vertical: return self.query_one("#command-popup", Vertical)
    def _command_options(self) -> OptionList: return self.query_one("#command-options", OptionList)
    def _write(self, markup: str) -> None: tui_write_markup(self, markup)

    @staticmethod
    def _bar_renderable(renderable: RenderableType, color: str, *, content_indent: int = 0, continuation_indent: Optional[int] = None): return tui_bar_renderable(renderable, color, content_indent=content_indent, continuation_indent=continuation_indent)

    @staticmethod
    def _line_indents(line: str, *, base_indent: int = 2) -> tuple[int, int]: return tui_line_indents(line, base_indent=base_indent)
    def _write_user_bar_line(self, markup: str = "", *, content_indent: int = 0) -> None: tui_write_user_bar_line(self, markup, content_indent=content_indent)
    def _write_assistant_bar_line(self, markup: str = "", *, content_indent: int = 0) -> None: tui_write_assistant_bar_line(self, markup, content_indent=content_indent)
    def _write_assistant_bar_renderable(self, renderable: RenderableType, *, content_indent: int = 0) -> None: tui_write_assistant_bar_renderable(self, renderable, content_indent=content_indent)
    def _write_user_bar_wrapped_line(self, line: str) -> None: tui_write_user_bar_wrapped_line(self, line)
    def _write_assistant_bar_wrapped_line(self, line: str, markup: str) -> None: tui_write_assistant_bar_wrapped_line(self, line, markup)
    def _write_renderable(self, renderable, indent: int = 2) -> None: tui_write_renderable(self, renderable, indent=indent)

    def _syntax_renderable(self, code: str, language: Optional[str]): return tui_syntax_renderable(self, code, language, syntax_theme=self._theme_spec().syntax_theme, background_color=self._theme_color("panel_bg", DEFAULT_PANEL_BG))

    def _code_panel_renderable(self, code: str, language: Optional[str]): return tui_code_panel_renderable(self, code, language)
    def _reasoning_panel_renderable(self, text: str): return tui_reasoning_panel_renderable(self, text)

    def _tool_event_panel(self, title: str, title_color: str, border_color: str, name: str, detail: str = ""): return tui_tool_event_panel(self, title, title_color, border_color, name, detail)
    def _tool_lifecycle_panel(self, name: str, detail: str, *, ok: bool): return tui_tool_lifecycle_panel(self, name, detail, ok=ok)

    def _update_tool_call_partial(self, name: str, detail: str = "") -> None: tui_update_tool_call_partial(self, name, detail)
    def _write_tool_lifecycle_block(self, name: str, ok: bool, detail: str = "") -> None: tui_write_tool_lifecycle_block(self, name, ok, detail)
    def _show_tool_result_line(self, name: str, ok: bool) -> bool: return tui_show_tool_result_line(self, name, ok)
    def _take_pending_tool_detail(self, name: str) -> str: return tui_take_pending_tool_detail(self, name)
    def _remember_code_block(self, code: str, language: Optional[str]) -> int: return tui_remember_code_block(self, code, language)
    def _write_code_block(self, lines: List[str], language: Optional[str], content_indent: int = 2) -> None: tui_write_code_block(self, lines, language, content_indent)
    def _render_static_markdown(self, text: str) -> None: tui_render_static_markdown(self, text)
    def _reset_fence_state(self) -> None: tui_reset_fence_state(self)

    @staticmethod
    def _is_fence_line(line: str) -> bool: return tui_is_fence_line(line)
    def _flush_fence_block(self) -> None: tui_flush_fence_block(self)
    def _render_content_line(self, line: str) -> None: tui_render_content_line(self, line)
    def _update_partial_content(self) -> None: tui_update_partial_content(self)
    def _update_live_preview_partial(self, lines: List[str], language: Optional[str]) -> None: tui_update_live_preview_partial(self, lines, language)
    def _defer_live_preview_partial(self, lines: List[str], language: Optional[str]) -> None: tui_defer_live_preview_partial(self, lines, language)
    def _clear_partial_preview(self) -> None: tui_clear_partial_preview(self)
    def _is_near_bottom(self, threshold: float = 1.0) -> bool: return tui_is_near_bottom(self, threshold=threshold)
    def _capture_scroll_anchor(self) -> Optional[ScrollAnchor]: return tui_capture_scroll_anchor(self)
    def _restore_scroll_anchor(self, anchor: Optional[ScrollAnchor]) -> None: tui_restore_scroll_anchor(self, anchor)
    def _maybe_scroll_end(self, force: bool = False) -> None: tui_maybe_scroll_end(self, force=force)

    def _write_info(self, text: str) -> None: tui_write_info(self, text, accent_color=self._theme_color("accent", DEFAULT_ACCENT_COLOR))

    def _write_error(self, text: str) -> None: tui_write_error(self, text)

    def _write_section_heading(self, title: str, color: Optional[str] = None) -> None: tui_write_section_heading(self, title, color=color or self._theme_color("accent", DEFAULT_ACCENT_COLOR))
    def _write_detail_line(self, label: str, value: str, *, value_markup: bool = False) -> None: tui_write_detail_line(self, label, value, accent_color=self._theme_color("accent", DEFAULT_ACCENT_COLOR), value_markup=value_markup)
    def _write_indexed_dim_lines(self, rows: List[str], *, color: Optional[str] = None, allow_markup: bool = False) -> None: tui_write_indexed_dim_lines(self, rows, color=color or self._theme_color("accent", DEFAULT_ACCENT_COLOR), allow_markup=allow_markup)
    def _write_command_action(self, text: str, *, icon: str = "•", color: Optional[str] = None) -> None: tui_write_command_action(self, text, color=color or self._theme_color("accent", DEFAULT_ACCENT_COLOR), icon=icon)

    def _write_command_row(self, command: str, desc: str, *, col: int) -> None: tui_write_command_row(self, command, desc, col=col, accent_color=self._theme_color("accent", DEFAULT_ACCENT_COLOR))
    def _write_muted_lines(self, rows: List[str]) -> None: tui_write_muted_lines(self, rows)
    def _write_usage(self, usage: str) -> bool: return tui_write_usage(self, usage)
    def _ensure_command_gap(self) -> None: tui_ensure_command_gap(self)

    def _reload_skills(self) -> bool:
        self.agent.reload_skills(); self._loaded_skill_ids = [skill.id for skill in self.agent.skill_runtime.skills_by_ids(self._loaded_skill_ids)]; self._save_active_session(); self._write_info("Reloaded skills"); return True

    @property
    def _reply_acc(self) -> str: return "".join(getattr(self, "_reply_acc_parts", []))

    @_reply_acc.setter
    def _reply_acc(self, value: str) -> None:
        text = str(value or ""); self._reply_acc_parts = [text] if text else []; self._reply_acc_len = len(text)

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
    def _reasoning_open(self) -> bool: return bool(self._stream_runtime_state().text.reasoning_open)

    @_reasoning_open.setter
    def _reasoning_open(self, value: bool) -> None: self._stream_runtime_state().text.reasoning_open = bool(value)

    @property
    def _content_open(self) -> bool: return bool(self._stream_runtime_state().text.content_open)

    @_content_open.setter
    def _content_open(self, value: bool) -> None: self._stream_runtime_state().text.content_open = bool(value)

    @property
    def _buf_r(self) -> str: return str(self._stream_runtime_state().text.reasoning_buffer)

    @_buf_r.setter
    def _buf_r(self, value: str) -> None: self._stream_runtime_state().text.reasoning_buffer = str(value)

    @property
    def _buf_c(self) -> str: return str(self._stream_runtime_state().text.content_buffer)

    @_buf_c.setter
    def _buf_c(self, value: str) -> None: self._stream_runtime_state().text.content_buffer = str(value)

    @property
    def _in_fence(self) -> bool: return bool(self._stream_runtime_state().text.in_fence)

    @_in_fence.setter
    def _in_fence(self, value: bool) -> None: self._stream_runtime_state().text.in_fence = bool(value)

    @property
    def _fence_lang(self) -> Optional[str]: return self._stream_runtime_state().text.fence_lang

    @_fence_lang.setter
    def _fence_lang(self, value: Optional[str]) -> None: self._stream_runtime_state().text.fence_lang = value

    @property
    def _fence_lines(self) -> List[str]: return self._stream_runtime_state().text.fence_lines

    @_fence_lines.setter
    def _fence_lines(self, value: List[str]) -> None: self._stream_runtime_state().text.fence_lines = list(value)

    def _is_tool_trace_line(self, line: str) -> bool:
        s = line.strip().lower()
        return "tool call:" in s

    def _update_status1(self) -> None:
        colors = self._theme_spec().colors
        text = status_right_markup(
            model_name=self._status_runtime.model_name,
            branch_armed=bool(self.conv_tree._pending_branch),
            branch_label=self.conv_tree._pending_branch_label,
            thinking=self.thinking,
            collaboration_mode=str(getattr(self, "_collaboration_mode", "execute")),
            width=self.size.width,
            colors=colors,
        )
        if text == self._last_status_right:
            self._update_topbar()
            return
        self._last_status_right = text
        self.query_one("#status-right", Static).update(text)
        self._update_topbar()

    def _current_model_refresh_interval(self) -> float: return tui_model_refresh_interval(self)
    def _should_startup_readiness_poll(self) -> bool: return should_tui_startup_readiness_poll(self)
    def _maybe_start_startup_readiness_poll(self) -> None: start_tui_startup_readiness_poll(self)

    @work(thread=True, exclusive=True)
    def _startup_readiness_worker(self) -> None:
        self.agent.ensure_ready(timeout_s=self.agent.readiness_timeout_s)
        status = self.agent.get_model_status()
        self.call_from_thread(self._finish_startup_readiness_poll, status)

    def _finish_startup_readiness_poll(self, status: ModelStatus) -> None: finish_tui_startup_readiness_poll(self, status)
    def _maybe_refresh_model_status(self, *, force: bool = False) -> None: maybe_refresh_tui_model_status(self, force=force)

    @work(thread=True, exclusive=True)
    def _refresh_model_status_worker(self) -> None:
        status = self.agent.refresh_model_status(timeout_s=min(self.agent.connect_timeout_s, 2.0), force=True)
        self.call_from_thread(self._apply_model_status_refresh, status)

    def _apply_model_status_refresh(self, status: ModelStatus) -> None: apply_tui_model_status(self, status)
    def _update_status2(self) -> None: update_tui_status2(self)
    def _update_topbar(self) -> None: update_tui_topbar(self)

    def _update_input_placeholder(self) -> None:
        self.query_one(ChatInput).placeholder = "Type to start branch…" if self.conv_tree._pending_branch else "Type a message…"

    def _update_footer_separator(self) -> None:
        try:
            separator = self.query_one("#footer-sep", Static)
            separator_region = getattr(separator, "region", None)
            separator_y = int(getattr(separator_region, "y", 0) or 0)
            if separator_y <= 0:
                self.call_after_refresh(self._update_footer_separator)
                return
        except (NoMatches, AttributeError):
            return

        try:
            sidebar = self.query_one("#sidebar", Vertical)
            sidebar_sep = self.query_one("#sidebar-footer-sep", Static)
            if not bool(getattr(sidebar, "display", True)):
                sidebar_sep.display = False
                return
            sidebar_sep.display = True

            sidebar_region = getattr(sidebar, "region", None)
            sidebar_y = int(getattr(sidebar_region, "y", 0) or 0)

            if sidebar_y <= 0 or separator_y <= 0:
                self.call_after_refresh(self._update_footer_separator)
                return

            target_y = max(0, separator_y - sidebar_y)
            sidebar_sep.styles.offset = (0, target_y)
        except (NoMatches, AttributeError):
            return

    def _update_pending_attachments(self) -> None:
        try:
            markup = self._pending_attachment_markup(); bar = self.query_one("#attachment-bar", Static); bar.update(markup)
        except (NoMatches, AttributeError):
            return

    def _pending_attachment_markup(self) -> str: return tui_pending_attachment_markup(self)
    def _workspace_root(self) -> Path: return workspace_tui_root(self)
    def _root_relative_label(self, path: Path, root: Path) -> str: return tui_root_relative_label(path, root)
    def _resolve_attachment_path(self, raw_path: str) -> Path: return resolve_tui_attachment_path(self, raw_path)
    def _attach_file_path(self, path: str | Path) -> bool: return attach_tui_file_path(self, path)
    def _attachment_picker_items(self, relative_dir: str = ".", *, root_id: str = "workspace") -> List[PickerItem]: return attachment_tui_picker_items(self, relative_dir, root_id=root_id, accent_color=self._theme_color("accent", DEFAULT_ACCENT_COLOR))
    def _open_attachment_picker(self, relative_dir: str = ".", root_id: str = "workspace") -> None: open_tui_attachment_picker(self, relative_dir, root_id=root_id, accent_color=self._theme_color("accent", DEFAULT_ACCENT_COLOR))

    def _workspace_file_candidates(self, *, max_items: int = 60) -> List[Path]: return workspace_tui_file_candidates(self, max_items=max_items, classify_attachment_fn=classify_attachment)

    def _build_global_palette_catalog(self) -> tuple[List[CommandPaletteItem], Dict[str, Dict[str, str]]]: return build_tui_global_palette_catalog(self, command_palette_item_cls=CommandPaletteItem)

    def _on_global_palette_close(self, result: Optional[Dict[str, str]]) -> None: on_tui_global_palette_close(self, result, chat_input_cls=ChatInput)
    def _refresh_command_popup(self, value: str) -> None: refresh_tui_command_popup(self, value, chat_input_cls=ChatInput)

    def _hide_command_popup(self) -> None: hide_tui_command_popup(self)
    def _command_popup_active(self) -> bool: return tui_command_popup_active(self)
    def _move_command_selection(self, delta: int) -> None: move_tui_command_selection(self, delta)
    def _accept_command_selection(self) -> bool: return accept_tui_command_selection(self, chat_input_cls=ChatInput)

    def _should_accept_popup_on_enter(self, text: str) -> bool:
        _ = text
        return should_tui_accept_popup_on_enter(self, chat_input_cls=ChatInput)

    @staticmethod
    def _config_for_editor(config: Dict[str, Any]) -> Dict[str, Any]: return config_for_editor_view(config)

    def _open_config_editor(self) -> None:
        warnings: List[str] = []
        try:
            raw = load_global_config(_global_config_path(), warnings=warnings)
        except (OSError, ValueError) as exc:
            self._write_error(f"Config load failed: {exc}")
            return
        safe = self._config_for_editor(raw if isinstance(raw, dict) else {})
        text = json.dumps(safe, indent=2) + "\n"
        self.push_screen(ConfigEditorModal(_global_config_path(), text, syntax_theme=self._theme_spec().text_area_theme), self._on_config_editor_close)
        for warning in warnings:
            self._write_info(f"Config warning: {warning}")

    def _cmd_theme(self) -> None: open_tui_theme_picker(self, picker_item_cls=PickerItem, selection_picker_modal_cls=SelectionPickerModal)
    def _on_theme_picker_close(self, result: Optional[Dict[str, str]]) -> None: on_tui_theme_picker_close(self, result, chat_input_cls=ChatInput, config_path=_global_config_path())
    def _persist_theme_preference(self, theme_id: str) -> List[str]: return persist_tui_theme_preference(self, theme_id, config_path=_global_config_path())
    def _merge_live_config(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]: return merge_tui_live_config(base, updates)
    def _apply_tui_config(self) -> None: apply_tui_runtime_config(self)
    def _on_config_editor_close(self, result: Optional[Dict[str, Any]]) -> None: on_tui_config_editor_close(self, result, config_path=_global_config_path(), status_runtime_state_cls=StatusRuntimeState)

    def _open_code_block(self, index: int) -> None:
        if index < 1 or index > len(self._code_blocks):
            self._write_error(f"No code block {index}")
            return
        code, language = self._code_blocks[index - 1]
        self.push_screen(CodeViewerModal(code, language, title=f"Code Block {index}", syntax_theme=self._theme_spec().text_area_theme))

    def _update_sidebar(self) -> None: update_tui_sidebar(self)
    def _write_turn_user(self, turn: Turn) -> None: write_tui_turn_user(self, turn, accent_color=self._theme_color("accent", DEFAULT_ACCENT_COLOR))
    def _write_skill_exchanges(self, turn: Turn) -> None: write_tui_skill_exchanges(self, turn)
    def _write_completed_turn_asst(self, turn: Turn) -> None: write_tui_completed_turn_assistant(self, turn)
    def _rebuild_viewport(self, *, preserve_scroll: bool = False) -> None: rebuild_tui_viewport(self, preserve_scroll=preserve_scroll)
    def _start_stream(self, turn: Turn, user_input: str, attachment_paths: List[str]) -> None: start_tui_turn_stream(self, turn, user_input, attachment_paths)

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
            collaboration_mode=str(getattr(self, "_collaboration_mode", "execute")),
            stop_event=stop_event,
            on_event=on_event,
            confirm_shell=self._confirm_shell_command,
        )
        self.call_from_thread(self._drain_stream_event_queue)
        self.call_from_thread(self._on_stream_end, turn_id, result)

    def _visible_reasoning_text(self, text: str) -> str: return "\n".join(line for line in visible_tui_reasoning_text(text).splitlines() if not self._is_tool_trace_line(line)).strip()

    def _flush_reasoning_buffer(self) -> None: flush_tui_reasoning_buffer(self)
    def _close_reasoning_section(self) -> bool: return close_tui_reasoning_section(self)
    def _refresh_deferred_partial(self) -> None: refresh_tui_deferred_partial(self)
    def _flush_content_buffer(self, include_partial: bool = False, *, update_partial: bool = True) -> None: flush_tui_content_buffer(self, include_partial=include_partial, update_partial=update_partial)
    def _handle_content_token(self, token: str, *, update_partial: bool = True) -> None: handle_tui_content_token(self, token, update_partial=update_partial)
    def _drain_stream_event_queue(self) -> None: drain_tui_stream_events(self)
    def _on_agent_event(self, event: Dict[str, Any]) -> None: on_tui_agent_event(self, event)
    def _on_stream_end(self, turn_id: str, result: AgentTurnResult) -> None: finish_tui_turn_stream(self, turn_id, result)
    def _confirm_shell_command(self, command: str) -> bool: return confirm_tui_shell_command(self, command)
    def _begin_shell_confirm(self, command: str, event: threading.Event, holder: Dict[str, bool]) -> None: begin_tui_shell_confirm(self, command, event, holder, esc=esc)
    def _expire_shell_confirm(self, event: threading.Event) -> None: expire_tui_shell_confirm(self, event)
    def _finish_shell_confirm(self, approved: bool) -> None: finish_tui_shell_confirm(self, approved)

    def _send(self, text: str) -> None:
        attachments = list(self.pending); self.pending.clear()
        content = build_content(text, attachments); turn = self.conv_tree.add_turn(content)
        self._save_active_session()
        self._write_turn_user(turn)
        for update in (self._update_pending_attachments, self._update_status1, self._update_status2, self._update_input_placeholder, self._update_sidebar): update()
        self._start_stream(turn, text, [p for p, _ in attachments])

    @on(Input.Submitted, "#chat-input")
    def on_input_submitted(self, event: Input.Submitted) -> None: on_tui_input_submitted(self, event, chat_input_cls=ChatInput)

    @on(Input.Changed, "#chat-input")
    def on_input_changed(self, event: Input.Changed) -> None:
        chat_input = self.query_one(ChatInput)
        chat_input.sync_paste_placeholders(event.value); self._refresh_command_popup(event.value)

    @on(Button.Pressed, "#attach-file")
    def on_attach_file_pressed(self) -> None:
        if self.streaming:
            return
        self._hide_command_popup(); self._open_attachment_picker(".")

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

    def action_handle_esc(self) -> None: action_tui_handle_esc(self, chat_input_cls=ChatInput)

    def action_scroll_up(self) -> None:
        self._scroll().scroll_page_up()
        if self.streaming: self._auto_follow_stream = False; self._update_status2()

    def action_scroll_down(self) -> None:
        self._scroll().scroll_page_down()
        if self.streaming and self._is_near_bottom(): self._auto_follow_stream = True; self._update_status2()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        if self.streaming: self._auto_follow_stream = False; self._update_status2()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        if not self.streaming:
            return
        self.call_after_refresh(self._resume_auto_follow_if_near_bottom)

    def _resume_auto_follow_if_near_bottom(self) -> None:
        if self.streaming and self._is_near_bottom(): self._auto_follow_stream = True; self._update_status2()

    def _handle_command(self, text: str) -> bool: return handle_tui_command(self, text)
    def _cmd_help(self) -> None: cmd_tui_help(self, help_sections=HELP_SECTIONS, accent_color=self._theme_color("accent", DEFAULT_ACCENT_COLOR))
    def _cmd_tree(self) -> None: cmd_tui_tree(self, accent_color=self._theme_color("accent", DEFAULT_ACCENT_COLOR))
    def _cmd_skills(self) -> None: cmd_tui_skills(self, accent_color=self._theme_color("accent", DEFAULT_ACCENT_COLOR))

    def _load_skill_into_session(self, skill_id: str) -> bool: return load_tui_skill_into_session(self, skill_id)
    def _unload_skill_from_session(self, skill_id: str) -> bool: return unload_tui_skill_from_session(self, skill_id)
    def _cmd_skill(self, arg: str) -> bool: return cmd_tui_skill(self, arg)
    def _cmd_memory(self, arg: str) -> bool: return cmd_tui_memory(self, arg)
    def _cmd_context(self, arg: str) -> bool: return cmd_tui_context(self, arg)
    def _cmd_doctor(self) -> None: cmd_tui_doctor(self, accent_color=self._theme_color("accent", DEFAULT_ACCENT_COLOR))
    def _cmd_report(self, arg: str) -> bool: return cmd_tui_report(self, arg)
    def _cmd_workspace(self, arg: str) -> bool: return cmd_tui_workspace(self, arg)
    def _cmd_code(self, arg: str) -> bool: return cmd_tui_code(self, arg)
