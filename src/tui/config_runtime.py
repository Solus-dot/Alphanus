from __future__ import annotations

from typing import Any, Dict

from core.runtime_config import UiRuntimeConfig


def merge_live_config(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = merge_live_config(base[key], value)
        else:
            merged[key] = value
    return merged


def install_stream_drain_timer(app: Any) -> None:
    interval_s = app._timing_config().stream_drain_interval_s
    current_timer = getattr(app, "_stream_drain_timer", None)
    current_interval = getattr(app, "_stream_drain_interval_s", getattr(current_timer, "_alphanus_interval_s", None))
    if current_timer is not None and current_interval == interval_s:
        return
    if current_timer is not None and hasattr(current_timer, "stop"):
        current_timer.stop()
    timer = app.set_interval(interval_s, app._drain_stream_event_queue)
    app._stream_drain_interval_s = interval_s
    app._stream_drain_timer = timer


def apply_tui_config(app: Any) -> None:
    app._ui_config = UiRuntimeConfig.from_config(app.agent.config)
    app._ui_timing = app._ui_config.timing
    app._chat_log_max_lines = app._ui_config.chat_log_max_lines
    app._tree_compaction_enabled = app._ui_config.tree_compaction_enabled
    app._inactive_assistant_char_limit = app._ui_config.inactive_assistant_char_limit
    app._inactive_tool_argument_char_limit = app._ui_config.inactive_tool_argument_char_limit
    app._inactive_tool_content_char_limit = app._ui_config.inactive_tool_content_char_limit
    app._scroll_interval = app._ui_timing.scroll_interval_s
    apply_theme = getattr(app, "_apply_theme_from_config", None)
    if callable(apply_theme):
        apply_theme()
    install_stream_drain_timer(app)
    app.conv_tree = app._apply_tree_compaction_policy(app.conv_tree)
    app._log().max_lines = app._chat_log_max_lines
    app._update_status1()
    app._update_status2()
    app._update_sidebar()
