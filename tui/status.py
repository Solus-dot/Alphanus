from __future__ import annotations

import os
from typing import Optional

from rich.markup import escape as esc


def topbar_left(endpoint: str) -> str:
    return (
        "[bold #6366f1 on #1a1730] ALPHANUS [/bold #6366f1 on #1a1730] "
        f"[#a1a1aa]{esc(endpoint)}[/#a1a1aa]"
    )


def topbar_right() -> str:
    return "[#a1a1aa]Session active[/#a1a1aa]"


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
