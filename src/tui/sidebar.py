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
            lines.append(f"[dim]{line}[/dim]")
    return "\n".join(lines)


def render_sidebar_inspector_markup(
    tree: ConvTree,
    width: int = 30,
    selected_id: str | None = None,
    *,
    colors: dict[str, str] | None = None,
) -> str:
    theme = _theme_colors(colors)
    lines: list[str] = []
    selected = selected_id or tree.current_id

    node = tree.nodes.get(selected) if selected else None
    if node is not None:
        parent = node.parent or "none"
        tools = len(node.skill_exchanges)
        branch_desc = node.label or ("branch" if node.branch_root else ("root" if node.id == "root" else "none"))
        user_preview = node.short(max_len=width + 14) if node.id != "root" else "root"
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
                f"[dim]id:[/dim] [{theme['text']}]{esc(node.id)}[/{theme['text']}]",
                f"[dim]state:[/dim] [{theme['text']}]{esc(node.assistant_state)}[/{theme['text']}]",
                f"[dim]parent:[/dim] [{theme['text']}]{esc(parent)}[/{theme['text']}]",
                f"[dim]children:[/dim] [{theme['text']}]{len(node.children)}[/{theme['text']}]",
                f"[dim]tools:[/dim] [{theme['text']}]{tools}[/{theme['text']}]",
                f"[dim]branch:[/dim] [{theme['text']}]{esc(branch_desc)}[/{theme['text']}]",
                f"[dim]user:[/dim] [{theme['text']}]{esc(user_preview)}[/{theme['text']}]",
            ]
        )
        if tool_summary:
            lines.append(f"[dim]calls:[/dim] [{theme['text']}]{esc(tool_summary)}[/{theme['text']}]")
        if node.assistant_content:
            lines.append(f"[dim]assistant:[/dim] [{theme['text']}]{len(node.assistant_content)} chars[/{theme['text']}]")
    return "\n".join(lines)
