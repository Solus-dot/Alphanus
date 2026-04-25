from typing import Optional
from urllib.parse import urlparse

from rich.markup import escape as esc

from tui.themes import fallback_color

_DEFAULT_COLORS = {
    "accent": fallback_color("accent"),
    "muted": fallback_color("muted"),
    "text": fallback_color("text"),
    "success": fallback_color("success"),
    "error": fallback_color("error"),
    "badge_bg": fallback_color("badge_bg"),
}


def _theme_colors(colors: Optional[dict[str, str]] = None) -> dict[str, str]:
    merged = dict(_DEFAULT_COLORS)
    if isinstance(colors, dict):
        merged.update({key: str(value) for key, value in colors.items() if value})
    return merged


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


def _join_topbar_segments(*segments: str) -> str:
    return "  ".join(segment for segment in segments if segment)


def context_usage_percent(context_tokens: Optional[int], context_window: Optional[int]) -> Optional[int]:
    if context_tokens is None or context_window is None or context_window <= 0:
        return None
    pct = int(round((max(0, context_tokens) * 100) / context_window))
    if context_tokens > 0 and pct == 0:
        pct = 1
    return min(pct, 999)


def _context_usage_markup(context_tokens: Optional[int], context_window: Optional[int], *, colors: Optional[dict[str, str]] = None) -> str:
    theme = _theme_colors(colors)
    if context_tokens is None or context_window is None or context_window <= 0:
        return f"[{theme['muted']}]—[/{theme['muted']}]"
    pct = context_usage_percent(context_tokens, context_window)
    if pct is None:
        return f"[{theme['muted']}]—[/{theme['muted']}]"
    return f"[{theme['accent']}]{pct}%[/{theme['accent']}]"


def _endpoint_state_markup(state: str, *, width: int, colors: Optional[dict[str, str]] = None) -> str:
    theme = _theme_colors(colors)
    normalized = (state or "unknown").strip().lower() or "unknown"
    if width < 110:
        label = {"online": "on", "offline": "off", "unknown": "?"}.get(normalized, "?")
    else:
        label = {"online": "online", "offline": "offline", "unknown": "unknown"}.get(normalized, normalized)
    color = {"online": theme["success"], "offline": theme["error"], "unknown": theme["muted"]}.get(normalized, theme["muted"])
    return f"[dim]llm:[/dim] [{color}]{esc(label)}[/{color}]"


def topbar_left(workspace_root: str, *, width: int, colors: Optional[dict[str, str]] = None) -> str:
    _ = workspace_root
    _ = width
    theme = _theme_colors(colors)
    return f"[bold {theme['accent']} on {theme['badge_bg']}] ALPHANUS [/bold {theme['accent']} on {theme['badge_bg']}]"


def topbar_center(*, session_name: str, branch_name: str, width: int, colors: Optional[dict[str, str]] = None) -> str:
    theme = _theme_colors(colors)
    if width < 105:
        return _join_topbar_segments(
            f"[dim]ss:[/dim] [{theme['text']}]{esc(_truncate(session_name, 10))}[/{theme['text']}]",
            f"[dim]br:[/dim] [{theme['accent']}]{esc(_truncate(branch_name, 10))}[/{theme['accent']}]",
        )
    if width < 140:
        return _join_topbar_segments(
            f"[dim]session:[/dim] [{theme['text']}]{esc(_truncate(session_name, 14))}[/{theme['text']}]",
            f"[dim]branch:[/dim] [{theme['accent']}]{esc(_truncate(branch_name, 12))}[/{theme['accent']}]",
        )
    return _join_topbar_segments(
        f"[dim]session:[/dim] [{theme['text']}]{esc(session_name)}[/{theme['text']}]",
        f"[dim]branch:[/dim] [{theme['accent']}]{esc(branch_name)}[/{theme['accent']}]",
    )


def topbar_right(
    *,
    endpoint: str,
    context_tokens: Optional[int],
    context_window: Optional[int],
    width: int,
    endpoint_state: str = "unknown",
    collaboration_mode: str = "execute",
    colors: Optional[dict[str, str]] = None,
) -> str:
    theme = _theme_colors(colors)
    short_endpoint = _short_endpoint(endpoint)
    if width < 105:
        short_endpoint = ""
    ctx_markup = _context_usage_markup(context_tokens, context_window, colors=theme)
    endpoint_markup = ""
    if short_endpoint:
        endpoint_markup = f"[{theme['muted']}]{esc(_truncate(short_endpoint, 22 if width < 140 else 28))}[/{theme['muted']}]"
    mode_label = "plan" if str(collaboration_mode or "").strip().lower() == "plan" else "execute"
    return _join_topbar_segments(
        endpoint_markup,
        _endpoint_state_markup(endpoint_state, width=width, colors=theme),
        f"[dim]ctx:[/dim] {ctx_markup}",
        f"[dim]mode:[/dim] [{theme['accent']}]{mode_label}[/{theme['accent']}]",
    )


def status_right_markup(
    *,
    model_name: Optional[str],
    branch_armed: bool,
    branch_label: Optional[str],
    thinking: bool,
    collaboration_mode: str = "execute",
    width: int,
    colors: Optional[dict[str, str]] = None,
) -> str:
    theme = _theme_colors(colors)
    model_limit = 24 if width < 120 else 40
    model_label = _truncate(model_name or "—", model_limit)
    model_markup = f"[dim]model:[/dim] [{theme['accent']}]{esc(model_label)}[/{theme['accent']}]"
    if width < 90:
        return model_markup

    parts = [model_markup]
    if branch_armed:
        if branch_label:
            label = _truncate(branch_label, 10 if width < 120 else 18)
            parts.append(f"[{theme['accent']}]branch: {esc(label)}[/{theme['accent']}]")
        else:
            parts.append(f"[{theme['accent']}]branch: armed[/{theme['accent']}]")
    else:
        parts.append("[dim]branch:[/dim] idle")

    think_label = "auto" if thinking else "off"
    parts.append(f"[dim]thinking:[/dim] [{theme['accent']}]{think_label}[/{theme['accent']}]")
    mode_label = "plan" if str(collaboration_mode or "").strip().lower() == "plan" else "execute"
    parts.append(f"[dim]mode:[/dim] [{theme['accent']}]{mode_label}[/{theme['accent']}]")
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
    colors: Optional[dict[str, str]] = None,
) -> str:
    theme = _theme_colors(colors)
    if await_shell_confirm:
        return "[bold yellow]approve shell command?[/bold yellow] [dim][y/n][/dim]"
    if streaming:
        if stop_requested:
            return f"[dim]{spinner_frame}[/dim] [yellow]stopping...[/yellow]"
        if esc_pending:
            return f"[dim]{spinner_frame}[/dim] [dim]generating[/dim] [bold red]esc again to stop[/bold red]"
        if not auto_follow_stream:
            return (
                f"[dim]pgup/dn[/dim] [{theme['accent']}]scroll[/{theme['accent']}]"
                if width < 110
                else f"[dim]pgup/dn ·[/dim] [{theme['accent']}]free scroll[/{theme['accent']}]"
            )
        return f"[dim]{spinner_frame}[/dim] [dim]generating[/dim] [dim]esc · stop[/dim]"
    if focus_panel == "tree":
        if width < 110:
            return f"[dim]j/k move[/dim]   [{theme['accent']}]enter[/{theme['accent']}]"
        return (
            f"[dim]j/k move[/dim]   [{theme['accent']}]enter open[/{theme['accent']}]"
            "   [dim][/] sib[/dim]   [dim]g/G ends[/dim]"
        )
    if focus_panel == "chat":
        return "[dim]pgup/dn scroll[/dim]   [dim]tab panel[/dim]" if width >= 110 else "[dim]tab panel[/dim]"
    if width >= 110:
        return "[dim]esc clear[/dim]   [dim]ctrl+f file[/dim]   [dim]tab panel[/dim]"
    return "[dim]esc clear[/dim]   [dim]ctrl+f[/dim]"
