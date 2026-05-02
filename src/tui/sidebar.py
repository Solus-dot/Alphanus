from __future__ import annotations

from rich.markup import escape as esc

from core.conv_tree import ConvTree
from tui.themes import fallback_color
from tui.tree_render import render_tree_rows

_DEFAULT_COLORS = {
    "accent": fallback_color("accent"),
    "text": fallback_color("text"),
    "muted": fallback_color("muted"),
    "subtle": fallback_color("subtle"),
}


def _theme_colors(colors: dict[str, str] | None = None) -> dict[str, str]:
    merged = dict(_DEFAULT_COLORS)
    if isinstance(colors, dict):
        merged.update({key: str(value) for key, value in colors.items() if value})
    return merged


def _truncate(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len == 1:
        return "…"
    return text[: max_len - 1] + "…"


def _field_line(label: str, value: str, *, width: int, theme: dict[str, str]) -> str:
    prefix = f"{label}: "
    visible_value = _truncate(value, max(0, width - len(prefix)))
    return f"[{theme['muted']}]{esc(label)}:[/{theme['muted']}] [{theme['text']}]{esc(visible_value)}[/{theme['text']}]"


def render_sidebar_tree_markup(
    tree: ConvTree,
    width: int = 30,
    selected_id: str | None = None,
    *,
    colors: dict[str, str] | None = None,
) -> str:
    theme = _theme_colors(colors)
    lines: list[str] = []
    current = tree.current_id
    selected = selected_id or current
    for text, tag, active in render_tree_rows(tree, width=width):
        line = esc(text)
        if tag == "root":
            style = theme["text"] if tag == selected else theme["muted"]
            lines.append(f"[{style}]{line}[/{style}]")
        elif tag == selected and tag == current:
            lines.append(f"[bold {theme['text']}]{line}[/bold {theme['text']}]")
        elif tag == selected:
            lines.append(f"[{theme['text']}]{line}[/{theme['text']}]")
        elif tag == current:
            lines.append(f"[bold {theme['accent']}]{line}[/bold {theme['accent']}]")
        elif active:
            lines.append(f"[{theme['subtle']}]{line}[/{theme['subtle']}]")
        else:
            lines.append(f"[{theme['muted']}]{line}[/{theme['muted']}]")
    return "\n".join(lines)


def render_sidebar_inspector_markup(
    tree: ConvTree,
    width: int = 30,
    selected_id: str | None = None,
    *,
    colors: dict[str, str] | None = None,
) -> str:
    theme = _theme_colors(colors)
    width = max(1, int(width))
    lines: list[str] = []
    selected = selected_id or tree.current_id

    node = tree.nodes.get(selected) if selected else None
    if node is not None:
        parent = node.parent or "none"
        tools = len(node.skill_exchanges)
        branch_desc = node.label or ("branch" if node.branch_root else ("root" if node.id == "root" else "none"))
        user_preview = node.short(max_len=width) if node.id != "root" else "root"
        tool_names = []
        for message in node.skill_exchanges:
            if message.get("role") == "tool":
                name = str(message.get("name") or "").strip()
                if name and name not in tool_names:
                    tool_names.append(name)
        tool_summary = ", ".join(tool_names[:3])
        if len(tool_names) > 3:
            tool_summary += ", …"
        lines.extend(
            [
                _field_line("id", node.id, width=width, theme=theme),
                _field_line("state", node.assistant_state, width=width, theme=theme),
                _field_line("parent", parent, width=width, theme=theme),
                _field_line("children", str(len(node.children)), width=width, theme=theme),
                _field_line("tools", str(tools), width=width, theme=theme),
                _field_line("branch", branch_desc, width=width, theme=theme),
                _field_line("user", user_preview, width=width, theme=theme),
            ]
        )
        if tool_summary:
            lines.append(_field_line("calls", tool_summary, width=width, theme=theme))
        if node.assistant_content:
            lines.append(_field_line("assistant", f"{len(node.assistant_content)} chars", width=width, theme=theme))
    return "\n".join(lines)
