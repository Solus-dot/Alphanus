from __future__ import annotations

import os
from typing import Optional
from urllib.parse import urlparse

from rich.markup import escape as esc


def _short_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme and parsed.netloc:
        return parsed.netloc
    return endpoint


def topbar_left(workspace_root: str) -> str:
    workspace_name = os.path.basename(workspace_root) or workspace_root
    return (
        "[bold #6366f1 on #1a1730] ALPHANUS [/bold #6366f1 on #1a1730] "
        f"[#f4f4f5]{esc(workspace_name)}[/#f4f4f5]"
    )


def topbar_center(*, branch_name: str, memory_mode: str) -> str:
    return (
        f"[dim]branch:[/dim] [#8b5cf6]{esc(branch_name)}[/#8b5cf6]   "
        f"[dim]memory:[/dim] [#10b981]{esc(memory_mode)}[/#10b981]"
    )


def topbar_right(*, endpoint: str, context_tokens: Optional[int]) -> str:
    short_endpoint = _short_endpoint(endpoint)
    ctx_markup = "[#a1a1aa]—[/#a1a1aa]" if context_tokens is None else f"[#6366f1]{context_tokens}[/#6366f1]"
    return (
        f"[#a1a1aa]{esc(short_endpoint)}[/#a1a1aa]   "
        f"[dim]ctx:[/dim] {ctx_markup}"
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
    focus_panel: str,
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
    if focus_panel == "tree":
        return "[dim]j/k move[/dim]   [#6366f1]enter open[/#6366f1]   [dim][/] sib[/dim]   [dim]g/G ends[/dim]"
    if focus_panel == "chat":
        return "[dim]pgup/dn scroll[/dim]   [dim]tab panel[/dim]"
    return "[dim]esc clear[/dim]   [dim]tab panel[/dim]"
