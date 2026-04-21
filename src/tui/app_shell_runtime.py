from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import Button, OptionList, Static

from core.runtime_config import UiRuntimeConfig
from core.sessions import SessionStore
from tui.live_tool_preview import LiveToolPreviewManager
from tui.status_runtime import StatusRuntimeState
from tui.stream_runtime import StreamRuntimeState


def initialize_shell_state(app: Any, *, agent: Any, debug: bool) -> None:
    app.agent = agent
    app._debug_mode = debug
    app._ui_config = UiRuntimeConfig.from_config(app.agent.config)
    app._ui_timing = app._ui_config.timing
    app._chat_log_max_lines = app._ui_config.chat_log_max_lines
    app._tree_compaction_enabled = app._ui_config.tree_compaction_enabled
    app._inactive_assistant_char_limit = app._ui_config.inactive_assistant_char_limit
    app._inactive_tool_argument_char_limit = app._ui_config.inactive_tool_argument_char_limit
    app._inactive_tool_content_char_limit = app._ui_config.inactive_tool_content_char_limit

    runtime_cfg = app.agent.config.get("runtime", {}) if isinstance(app.agent.config, dict) else {}
    configured_state_root = str(runtime_cfg.get("state_root", "")).strip() if isinstance(runtime_cfg, dict) else ""
    if not configured_state_root:
        raise ValueError("runtime.state_root is required in normalized runtime config")
    state_root = Path(configured_state_root).expanduser().resolve()
    app._session_store = SessionStore(
        state_root,
        storage_dir=state_root / "sessions",
    )
    app._session_id = ""
    app._session_title = ""
    app._session_created_at = ""
    app._loaded_skill_ids = []
    app.conv_tree = app._new_conv_tree()
    app._activate_session_state(app._session_store.bootstrap())
    app.pending = []

    app._stop_event = threading.Event()
    app._active_turn_id = None
    app._reply_acc = ""
    app._live_preview = LiveToolPreviewManager()
    app._stream_runtime = StreamRuntimeState()
    app._reasoning_open = False
    app._content_open = False
    app._buf_r = ""
    app._buf_c = ""
    app._in_fence = False
    app._fence_lang = None
    app._fence_lines = []
    app._stream_drain_timer = None
    app._stream_drain_interval_s = None
    app._partial_renderable = None
    app._last_partial_render_width = 1
    app._last_partial_line_count = 0
    app._partial_line_count_dirty = False

    app._last_scroll = 0.0
    app._scroll_interval = app._ui_timing.scroll_interval_s
    app._last_status_left = ""
    app._last_status_right = ""
    app._auto_follow_stream = True
    initial_status = app.agent.get_model_status()
    app._status_runtime = StatusRuntimeState(
        model_status=initial_status,
        model_name=initial_status.model_name if initial_status.state != "offline" else None,
        model_context_window=(
            initial_status.context_window
            if isinstance(initial_status.context_window, int) and initial_status.context_window > 0
            else None
        ),
    )

    app._esc_pending = False
    app._esc_ts = 0.0
    app._spin_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    app._spin_i = 0

    app._await_shell_confirm = False
    app._shell_confirm_command = ""
    app._shell_confirm_event = None
    app._shell_confirm_result = None
    app._command_matches = []
    app._global_palette_actions = {}
    app._code_blocks = []
    app._show_tool_details = True
    app._pending_tool_details = []
    app._focused_panel = "input"
    app._tree_cursor_id = "root"
    app._last_log_was_blank = False
    app._last_model_context_tokens = None
    app._startup_session_prompt_opened = False
    app._resize_redraw_pending = False


def compose_shell(app: Any, *, chat_input_cls: Any, transcript_view_cls: Any):
    with Horizontal(id="topbar"):
        yield Static("", id="topbar-left")
        yield Static("", id="topbar-center")
        yield Static("", id="topbar-right")
    with Horizontal(id="main-area"):
        with Vertical(id="chat-column"):
            with ScrollableContainer(id="chat-scroll"):
                yield transcript_view_cls(id="chat-log", max_lines=app._chat_log_max_lines)
                yield Static("", id="partial", markup=True)
            with Vertical(id="footer"):
                yield Static("", id="footer-sep", markup=True)
                yield Static("", id="attachment-bar", markup=True)
                with Horizontal(id="input-row"):
                    with Horizontal(id="composer-shell"):
                        yield chat_input_cls(id="chat-input", placeholder="Type a message…")
                        with Horizontal(id="input-accessories"):
                            yield Button("+ File (Ctrl+F)", id="attach-file")
                with Horizontal(id="status-bar"):
                    yield Static("", id="status-left")
                    yield Static("", id="status-right")
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
            yield Static("", id="sidebar-footer-sep", markup=True)
    with Vertical(id="command-popup"):
        yield Static("commands", id="command-popup-title")
        yield Static("type to filter · tab to insert", id="command-popup-hint")
        yield OptionList(id="command-options")
