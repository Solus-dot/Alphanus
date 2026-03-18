from __future__ import annotations

from rich.markup import escape as esc

from core.conv_tree import ConvTree


def render_sidebar_markup(tree: ConvTree, width: int = 30, selected_id: str | None = None) -> str:
    lines = [
        "[bold #a1a1aa]Conversation Tree[/bold #a1a1aa]",
        f"[dim]{tree.turn_count()} turns[/dim]",
        "",
    ]
    current = tree.current_id
    selected = selected_id or current
    for text, tag, active in tree.render_tree(width=width):
        line = esc(text)
        if tag == "root":
            style = "#f4f4f5" if tag == selected else "#a1a1aa"
            lines.append(f"[{style}]{line}[/{style}]")
        elif tag == selected and tag == current:
            lines.append(f"[bold reverse #09090b on #8b5cf6]{line}[/bold reverse #09090b on #8b5cf6]")
        elif tag == selected:
            lines.append(f"[bold reverse #09090b on #f59e0b]{line}[/bold reverse #09090b on #f59e0b]")
        elif tag == current:
            lines.append(f"[bold #6366f1]{line}[/bold #6366f1]")
        elif active:
            lines.append(f"[#8b5cf6]{line}[/#8b5cf6]")
        else:
            lines.append(f"[dim]{line}[/dim]")

    node = tree.nodes.get(selected) if selected else None
    if node is not None:
        parent = node.parent or "none"
        tools = len(node.skill_exchanges)
        branch_desc = node.label or ("branch" if node.branch_root else ("root" if node.id == "root" else "none"))
        lines.extend(
            [
                "",
                "[bold #a1a1aa]Inspector[/bold #a1a1aa]",
                f"[dim]id:[/dim] [#f4f4f5]{esc(node.id)}[/#f4f4f5]",
                f"[dim]state:[/dim] [#f4f4f5]{esc(node.assistant_state)}[/#f4f4f5]",
                f"[dim]parent:[/dim] [#f4f4f5]{esc(parent)}[/#f4f4f5]",
                f"[dim]children:[/dim] [#f4f4f5]{len(node.children)}[/#f4f4f5]",
                f"[dim]tools:[/dim] [#f4f4f5]{tools}[/#f4f4f5]",
                f"[dim]branch:[/dim] [#f4f4f5]{esc(branch_desc)}[/#f4f4f5]",
            ]
        )
        if node.assistant_content:
            lines.append(
                f"[dim]assistant:[/dim] [#f4f4f5]{len(node.assistant_content)} chars[/#f4f4f5]"
            )
    return "\n".join(lines)
