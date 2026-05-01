from __future__ import annotations

import json
from typing import Any

from rich.markup import escape as esc
from textual.widgets import Static

from core.conv_tree import Turn
from tui.sidebar import render_sidebar_inspector_markup, render_sidebar_tree_markup
from tui.themes import fallback_color

DEFAULT_ACCENT_COLOR = fallback_color("accent")
DEFAULT_MUTED_COLOR = fallback_color("muted")


def sidebar_render_width(sidebar: Any) -> int:
    width = int(getattr(sidebar.region, "width", 0) or 0)
    if width <= 0:
        width = int(getattr(sidebar.size, "width", 0) or 0)
    return max(20, width - 8) if width > 0 else 30


def update_sidebar(app: Any) -> None:
    sidebar = app.query_one("#sidebar")
    if not sidebar.display:
        return
    if app._focused_panel != "tree":
        app._tree_cursor_id = app.conv_tree.current_id
    else:
        app._sync_tree_cursor()
    width = sidebar_render_width(sidebar)
    colors = app._theme_spec().colors if hasattr(app, "_theme_spec") else None
    app.query_one("#sidebar-tree-meta", Static).update(f"{app.conv_tree.turn_count()} turns")
    app.query_one("#sidebar-tree-content", Static).update(
        render_sidebar_tree_markup(app.conv_tree, width=width, selected_id=app._tree_cursor_id, colors=colors)
    )
    app.query_one("#sidebar-inspector-content", Static).update(
        render_sidebar_inspector_markup(app.conv_tree, width=width, selected_id=app._tree_cursor_id, colors=colors)
    )


def write_turn_user(app: Any, turn: Turn, accent_color: str = DEFAULT_ACCENT_COLOR) -> None:
    muted = app._theme_color("muted", DEFAULT_MUTED_COLOR) if hasattr(app, "_theme_color") else DEFAULT_MUTED_COLOR
    app._write("")
    if turn.branch_root:
        label = f" ⎇  {esc(turn.label)}" if turn.label else " ⎇  branch"
        app._write(f"[dim {accent_color}]{label}[/dim {accent_color}]")
    attachment_summary = turn.attachment_summary()
    if attachment_summary:
        app._write_user_bar_line(
            f"[dim]attachments:[/dim] [{muted}]{esc(attachment_summary)}[/{muted}]",
            content_indent=2,
        )
    body = turn.user_text()
    for line in body.splitlines() or [""]:
        app._write_user_bar_wrapped_line(line)


def write_skill_exchanges(app: Any, turn: Turn) -> None:
    pending_details: list[tuple[str, str]] = []
    for msg in turn.skill_exchanges:
        raw_tool_calls = msg.get("tool_calls")
        if msg.get("role") == "assistant" and isinstance(raw_tool_calls, list):
            if not app._show_tool_details:
                continue
            for call in raw_tool_calls:
                call_obj = call if isinstance(call, dict) else {}
                function_obj = call_obj.get("function")
                function_map = function_obj if isinstance(function_obj, dict) else {}
                name = str(function_map.get("name") or "unknown")
                raw_args = function_map.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = raw_args
                pending_details.append((name, app._live_preview.compact_tool_args(name, args)))
                app._live_preview.write_static_preview(
                    name,
                    args,
                    app._write_assistant_bar_line,
                    lambda markup, _indent=0: app._write_assistant_bar_line(markup),
                    app._write_code_block,
                )
        elif msg.get("role") == "tool":
            name = msg.get("name", "tool")
            content = msg.get("content", "{}")
            try:
                payload = json.loads(content) if isinstance(content, str) else content
            except json.JSONDecodeError:
                payload = {"ok": False, "error": {"message": "invalid tool response"}}
            if not isinstance(payload, dict):
                payload = {"ok": False, "error": {"message": "invalid tool response"}}
            payload_obj = dict(payload)
            if payload_obj.get("ok") and app._show_tool_details:
                app._live_preview.write_result_preview(
                    name,
                    payload_obj,
                    app._write_assistant_bar_line,
                    lambda markup, _indent=0: app._write_assistant_bar_line(markup),
                    app._write_code_block,
                )
            if not app._show_tool_result_line(name, bool(payload_obj.get("ok"))):
                continue
            detail = ""
            for idx, (pending_name, pending_detail) in enumerate(pending_details):
                if pending_name == name:
                    detail = pending_detail
                    pending_details.pop(idx)
                    break
            if payload_obj.get("ok"):
                app._write_tool_lifecycle_block(name, True, detail or "completed")
            else:
                error_obj = payload_obj.get("error")
                error_map = error_obj if isinstance(error_obj, dict) else {}
                em = error_map.get("message", "failed")
                app._write_tool_lifecycle_block(name, False, f"{detail}   {em}".strip())


def write_completed_turn_assistant(app: Any, turn: Turn) -> None:
    app._write("")
    app._write_skill_exchanges(turn)

    content = turn.assistant_content or ""
    state = str(getattr(turn, "assistant_state", "") or ("cancelled" if "[interrupted]" in content else "done"))
    interrupted = state == "cancelled"
    failed = state == "error"
    display = content.replace("\n[interrupted]", "").rstrip()
    app._render_static_markdown(display)

    if interrupted:
        app._write("[dim red]  ✖ interrupted[/dim red]")
    elif failed:
        app._write("[dim red]  ! failed[/dim red]")


def rebuild_viewport(app: Any, *, preserve_scroll: bool = False) -> None:
    scroll_anchor = app._capture_scroll_anchor() if preserve_scroll else None
    log = app._log()
    log.clear_entries()
    for turn in app.conv_tree.active_path:
        if turn.id == "root":
            continue
        app._write_turn_user(turn)
        if turn.assistant_content:
            app._write_completed_turn_asst(turn)
    if scroll_anchor is not None:
        app.call_after_refresh(lambda anchor=scroll_anchor: app._restore_scroll_anchor(anchor))
    else:
        app._maybe_scroll_end(force=True)
