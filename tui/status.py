import os
from typing import Optional
from urllib.parse import urlparse

from rich.markup import escape as esc


def _truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1] + "…"


def _short_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme and parsed.netloc:
        return parsed.netloc
    return endpoint


def topbar_left(workspace_root: str, *, width: int) -> str:
    workspace_name = os.path.basename(workspace_root) or workspace_root
    if width < 110:
        workspace_name = _truncate(workspace_name, 14)
    elif width < 140:
        workspace_name = _truncate(workspace_name, 20)
    return (
        "[bold #6366f1 on #1a1730] ALPHANUS [/bold #6366f1 on #1a1730] "
        f"[#f4f4f5]{esc(workspace_name)}[/#f4f4f5]"
    )


def topbar_center(*, session_name: str, branch_name: str, memory_mode: str, width: int) -> str:
    if width < 105:
        return (
            f"[dim]ss:[/dim] [#f4f4f5]{esc(_truncate(session_name, 10))}[/#f4f4f5]   "
            f"[dim]br:[/dim] [#6366f1]{esc(_truncate(branch_name, 10))}[/#6366f1]"
        )
    if width < 140:
        return (
            f"[dim]session:[/dim] [#f4f4f5]{esc(_truncate(session_name, 14))}[/#f4f4f5]   "
            f"[dim]branch:[/dim] [#6366f1]{esc(_truncate(branch_name, 12))}[/#6366f1]"
        )
    return (
        f"[dim]session:[/dim] [#f4f4f5]{esc(session_name)}[/#f4f4f5]   "
        f"[dim]branch:[/dim] [#6366f1]{esc(branch_name)}[/#6366f1]   "
        f"[dim]memory:[/dim] [#10b981]{esc(memory_mode)}[/#10b981]"
    )


def topbar_right(*, endpoint: str, context_tokens: Optional[int], width: int) -> str:
    short_endpoint = _short_endpoint(endpoint)
    if width < 105:
        short_endpoint = ""
    ctx_markup = "[#a1a1aa]—[/#a1a1aa]" if context_tokens is None else f"[#6366f1]{context_tokens}[/#6366f1]"
    if not short_endpoint:
        return f"  [dim]ctx:[/dim] {ctx_markup}"
    return f"  [#a1a1aa]{esc(_truncate(short_endpoint, 22 if width < 140 else 28))}[/#a1a1aa]   [dim]ctx:[/dim] {ctx_markup}"


def status_right_markup(
    *,
    model_name: Optional[str],
    branch_armed: bool,
    branch_label: Optional[str],
    thinking: bool,
    width: int,
) -> str:
    model_limit = 24 if width < 120 else 40
    model_label = _truncate(model_name or "—", model_limit)
    model_markup = f"[dim]model:[/dim] [#6366f1]{esc(model_label)}[/#6366f1]"
    if width < 90:
        return model_markup

    parts = [model_markup]
    if branch_armed:
        if branch_label:
            label = _truncate(branch_label, 10 if width < 120 else 18)
            parts.append(f"[#6366f1]branch: {esc(label)}[/#6366f1]")
        else:
            parts.append("[#6366f1]branch: armed[/#6366f1]")
    else:
        parts.append("[dim]branch:[/dim] idle")

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
    width: int,
) -> str:
    if await_shell_confirm:
        return "[bold yellow]approve shell command?[/bold yellow] [dim][y/n][/dim]"
    if streaming:
        if stop_requested:
            return f"[dim]{spinner_frame}[/dim] [yellow]stopping...[/yellow]"
        if esc_pending:
            return f"[dim]{spinner_frame}[/dim] [dim]generating[/dim] [bold red]esc again to stop[/bold red]"
        if not auto_follow_stream:
            return "[dim]pgup/dn[/dim] [#6366f1]scroll[/#6366f1]" if width < 110 else "[dim]pgup/dn ·[/dim] [#6366f1]free scroll[/#6366f1]"
        return f"[dim]{spinner_frame}[/dim] [dim]generating[/dim] [dim]esc · stop[/dim]"
    if focus_panel == "tree":
        if width < 110:
            return "[dim]j/k move[/dim]   [#6366f1]enter[/#6366f1]"
        return "[dim]j/k move[/dim]   [#6366f1]enter open[/#6366f1]   [dim][/] sib[/dim]   [dim]g/G ends[/dim]"
    if focus_panel == "chat":
        return "[dim]pgup/dn scroll[/dim]   [dim]tab panel[/dim]" if width >= 110 else "[dim]tab panel[/dim]"
    return "[dim]esc clear[/dim]   [dim]tab panel[/dim]" if width >= 110 else "[dim]esc clear[/dim]"
