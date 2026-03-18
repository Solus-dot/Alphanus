from __future__ import annotations

import os
from typing import Optional
from urllib.parse import urlparse

from rich.markup import escape as esc


def _short_workspace(path: str) -> str:
    path = os.path.abspath(path)
    home = os.path.expanduser("~")
    if path.startswith(home):
        path = "~" + path[len(home) :]
    return path


def _short_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme and parsed.netloc:
        base = f"{parsed.scheme}://{parsed.netloc}"
        if parsed.path and parsed.path != "/":
            return base + parsed.path
        return base
    return endpoint


def topbar_left(workspace_root: str) -> str:
    short_ws = _short_workspace(workspace_root)
    return (
        "[bold #6366f1 on #1a1730] ALPHANUS [/bold #6366f1 on #1a1730] "
        f"[#f4f4f5]{esc(os.path.basename(workspace_root) or workspace_root)}[/#f4f4f5] "
        f"[#71717a]{esc(short_ws)}[/#71717a]"
    )


def topbar_center(*, branch_name: str, memory_mode: str, focus_panel: str) -> str:
    focus_label = {
        "chat": "transcript",
        "tree": "tree",
        "input": "input",
    }.get(focus_panel, focus_panel)
    return (
        f"[dim]branch:[/dim] [#8b5cf6]{esc(branch_name)}[/#8b5cf6]   "
        f"[dim]memory:[/dim] [#10b981]{esc(memory_mode)}[/#10b981]   "
        f"[dim]focus:[/dim] [#f59e0b]{esc(focus_label)}[/#f59e0b]"
    )


def topbar_right(*, endpoint: str, context_tokens: int, context_limit: int) -> str:
    short_endpoint = _short_endpoint(endpoint)
    return (
        f"[#a1a1aa]{esc(short_endpoint)}[/#a1a1aa]   "
        f"[dim]ctx:[/dim] [#6366f1]{context_tokens}[/#6366f1][dim]/[/dim][#a1a1aa]{context_limit}[/#a1a1aa]"
    )


def status_right_markup(
    *,
    pending_count: int,
    branch_armed: bool,
    branch_label: Optional[str],
    latest_path: Optional[str],
    latest_kind: Optional[str],
    thinking: bool,
) -> str:
    parts = [f"[dim]files:[/dim] {pending_count}"]
    if branch_armed:
        if branch_label:
            parts.append(f"[#6366f1]branch: {esc(branch_label)}[/#6366f1]")
        else:
            parts.append("[#6366f1]branch: armed[/#6366f1]")
    else:
        parts.append("[dim]branch:[/dim] idle")

    if latest_path:
        color = "#6366f1" if latest_kind == "image" else "#10b981"
        parts.append(f"[{color}]{esc(os.path.basename(latest_path))}[/{color}]")

    think_label = "auto" if thinking else "off"
    parts.append(f"[dim]thinking:[/dim] [#6366f1]{think_label}[/#6366f1]")
    return "  ".join(parts)


def status_left_markup(
    *,
    await_shell_confirm: bool,
    streaming: bool,
    spinner_frame: str,
    stop_requested: bool,
    esc_pending: bool,
    auto_follow_stream: bool,
) -> str:
    if await_shell_confirm:
        return "[bold yellow]approve shell command?[/bold yellow] [dim][y/n][/dim]"
    if streaming:
        if stop_requested:
            return f"[dim]{spinner_frame}[/dim] [yellow]stopping...[/yellow]"
        if esc_pending:
            return f"[dim]{spinner_frame}[/dim] [dim]generating[/dim] [bold red]esc again to stop[/bold red]"
        if not auto_follow_stream:
            return "[dim]pgup/dn ·[/dim] [#6366f1]free scroll[/#6366f1]"
        return f"[dim]{spinner_frame}[/dim] [dim]generating[/dim] [dim]esc · stop[/dim]"
    return "[dim]esc · clear[/dim]   [#6366f1]pgup/dn · free scroll[/#6366f1]"
