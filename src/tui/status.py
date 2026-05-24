from urllib.parse import urlparse

from rich.markup import escape as esc

from tui.themes import fallback_color

_DEFAULT_COLORS = {
    "accent": fallback_color("accent"),
    "muted": fallback_color("muted"),
    "text": fallback_color("text"),
    "success": fallback_color("success"),
    "error": fallback_color("error"),
}


def _theme_colors(colors: dict[str, str] | None = None) -> dict[str, str]:
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


def _join_metadata_segments(*segments: str) -> str:
    return " [dim]·[/dim] ".join(segment for segment in segments if segment)


def context_usage_percent(context_tokens: int | None, context_window: int | None) -> int | None:
    if context_tokens is None or context_window is None or context_window <= 0:
        return None
    pct = int(round((max(0, context_tokens) * 100) / context_window))
    if context_tokens > 0 and pct == 0:
        pct = 1
    return min(pct, 999)


def metadata_center_markup(*, session_name: str, branch_name: str, width: int, colors: dict[str, str] | None = None) -> str:
    theme = _theme_colors(colors)
    if width < 105:
        return _join_metadata_segments(
            f"[dim]ss:[/dim] [{theme['text']}]{esc(_truncate(session_name, 10))}[/{theme['text']}]",
            f"[dim]br:[/dim] [{theme['accent']}]{esc(_truncate(branch_name, 10))}[/{theme['accent']}]",
        )
    if width < 140:
        return _join_metadata_segments(
            f"[dim]session:[/dim] [{theme['text']}]{esc(_truncate(session_name, 14))}[/{theme['text']}]",
            f"[dim]branch:[/dim] [{theme['accent']}]{esc(_truncate(branch_name, 12))}[/{theme['accent']}]",
        )
    return _join_metadata_segments(
        f"[dim]session:[/dim] [{theme['text']}]{esc(session_name)}[/{theme['text']}]",
        f"[dim]branch:[/dim] [{theme['accent']}]{esc(branch_name)}[/{theme['accent']}]",
    )


def metadata_right_markup(
    *,
    endpoint: str,
    context_tokens: int | None,
    context_window: int | None,
    width: int,
    endpoint_state: str = "unknown",
    model_integrity: str = "unknown",
    colors: dict[str, str] | None = None,
) -> str:
    theme = _theme_colors(colors)
    parsed_endpoint = urlparse(endpoint)
    short_endpoint = parsed_endpoint.netloc if parsed_endpoint.scheme and parsed_endpoint.netloc else endpoint
    if width < 105:
        short_endpoint = ""
    usage_pct = context_usage_percent(context_tokens, context_window)
    ctx_markup = f"[{theme['muted']}]—[/{theme['muted']}]" if usage_pct is None else f"[{theme['accent']}]{usage_pct}%[/{theme['accent']}]"
    endpoint_markup = ""
    if short_endpoint:
        endpoint_markup = (
            f"[dim]endpoint:[/dim] [{theme['muted']}]{esc(_truncate(short_endpoint, 22 if width < 140 else 28))}[/{theme['muted']}]"
        )
    integrity = str(model_integrity or "").strip().lower()
    integrity_markup = ""
    if integrity == "violation":
        integrity_markup = f"[dim]int:[/dim] [{theme['error']}]fail[/{theme['error']}]"
    normalized_state = (endpoint_state or "unknown").strip().lower() or "unknown"
    if width < 110:
        state_label = {"online": "on", "offline": "off", "unknown": "?"}.get(normalized_state, "?")
    else:
        state_label = {"online": "online", "offline": "offline", "unknown": "unknown"}.get(normalized_state, normalized_state)
    state_color = {"online": theme["success"], "offline": theme["error"], "unknown": theme["muted"]}.get(normalized_state, theme["muted"])
    rendered = _join_metadata_segments(
        endpoint_markup,
        integrity_markup,
        f"[dim]llm:[/dim] [{state_color}]{esc(state_label)}[/{state_color}]",
        f"[dim]ctx:[/dim] {ctx_markup}",
    )
    return f"[dim]·[/dim] {rendered}" if rendered else ""


def status_right_markup(
    *,
    model_name: str | None,
    thinking: bool,
    collaboration_mode: str = "execute",
    width: int,
    colors: dict[str, str] | None = None,
) -> str:
    theme = _theme_colors(colors)
    model_limit = 24 if width < 120 else 40
    model_label = _truncate(model_name or "—", model_limit)
    model_markup = f"[dim]model:[/dim] [{theme['accent']}]{esc(model_label)}[/{theme['accent']}]"
    if width < 90:
        return model_markup

    parts = [model_markup]
    think_label = "auto" if thinking else "off"
    parts.append(f"[dim]thinking:[/dim] [{theme['accent']}]{think_label}[/{theme['accent']}]")
    if str(collaboration_mode or "").strip().lower() == "plan":
        parts.append(f"[dim]mode:[/dim] [{theme['accent']}]plan[/{theme['accent']}]")
    return " [dim]·[/dim] ".join(parts)


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
    colors: dict[str, str] | None = None,
) -> str:
    theme = _theme_colors(colors)
    if await_shell_confirm:
        return "[bold yellow]approve shell command?[/bold yellow] [dim][y/n][/dim]"
    if streaming:
        if stop_requested:
            return f"[dim]{spinner_frame}[/dim] [yellow]stopping after current step…[/yellow]"
        if esc_pending:
            return f"[dim]{spinner_frame}[/dim] [dim]generating[/dim] [bold red]esc to confirm[/bold red]"
        if not auto_follow_stream:
            return (
                f"[dim]pgup/dn[/dim] [{theme['accent']}]scroll[/{theme['accent']}]"
                if width < 110
                else f"[dim]pgup/dn ·[/dim] [{theme['accent']}]free scroll[/{theme['accent']}]"
            )
        return f"[dim]{spinner_frame}[/dim] [dim]generating[/dim] [dim]esc stop[/dim]"
    if focus_panel == "tree":
        if width < 110:
            return f"[dim]j/k move[/dim] [dim]·[/dim] [{theme['accent']}]enter[/{theme['accent']}]"
        return (
            f"[dim]tree split[/dim] [dim]·[/dim] [dim]j/k move[/dim] [dim]·[/dim] "
            f"[{theme['accent']}]enter open[/{theme['accent']}] [dim]·[/dim] [dim]\\[/] sib[/dim] [dim]·[/dim] [dim]g/G ends[/dim]"
        )
    if focus_panel == "chat":
        return "[dim]PgUp/PgDn to Scroll[/dim] [dim]·[/dim] [dim]Tab for Panel[/dim]" if width >= 110 else "[dim]Tab for Panel[/dim]"
    if width >= 110:
        return "[dim]Esc to Clear[/dim] [dim]·[/dim] [dim]Ctrl+F to Upload File[/dim] [dim]·[/dim] [dim]Tab for Panel[/dim]"
    return "[dim]Esc to Clear[/dim] [dim]·[/dim] [dim]Ctrl+F[/dim]"
