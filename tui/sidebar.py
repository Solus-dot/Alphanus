from __future__ import annotations

from rich.markup import escape as esc

from core.conv_tree import ConvTree


def render_sidebar_markup(tree: ConvTree, width: int = 30) -> str:
    lines = [
        "[bold #a1a1aa]Conversation Tree[/bold #a1a1aa]",
        f"[dim]{tree.turn_count()} turns[/dim]",
        "",
    ]
    current = tree.current_id
    for text, tag, active in tree.render_tree(width=width):
        line = esc(text)
        if tag == "root":
            lines.append(f"[#a1a1aa]{line}[/#a1a1aa]")
        elif tag == current:
            lines.append(f"[bold #6366f1]{line}[/bold #6366f1]")
        elif active:
            lines.append(f"[#8b5cf6]{line}[/#8b5cf6]")
        else:
            lines.append(f"[dim]{line}[/dim]")
    return "\n".join(lines)
