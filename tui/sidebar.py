from __future__ import annotations

from rich.markup import escape as esc

from core.conv_tree import ConvTree


def render_sidebar_tree_markup(tree: ConvTree, width: int = 30, selected_id: str | None = None) -> str:
    lines: list[str] = []
    current = tree.current_id
    selected = selected_id or current
    for text, tag, active in tree.render_tree(width=width):
        line = esc(text)
        if tag == "root":
            style = "#f4f4f5" if tag == selected else "#a1a1aa"
            lines.append(f"[{style}]{line}[/{style}]")
        elif tag == selected and tag == current:
            lines.append(f"[bold #f4f4f5]{line}[/bold #f4f4f5]")
        elif tag == selected:
            lines.append(f"[#f4f4f5]{line}[/#f4f4f5]")
        elif tag == current:
            lines.append(f"[bold #6366f1]{line}[/bold #6366f1]")
        elif active:
            lines.append(f"[#71717a]{line}[/#71717a]")
        else:
            lines.append(f"[dim]{line}[/dim]")
    return "\n".join(lines)


def render_sidebar_inspector_markup(tree: ConvTree, width: int = 30, selected_id: str | None = None) -> str:
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
                f"[dim]id:[/dim] [#f4f4f5]{esc(node.id)}[/#f4f4f5]",
                f"[dim]state:[/dim] [#f4f4f5]{esc(node.assistant_state)}[/#f4f4f5]",
                f"[dim]parent:[/dim] [#f4f4f5]{esc(parent)}[/#f4f4f5]",
                f"[dim]children:[/dim] [#f4f4f5]{len(node.children)}[/#f4f4f5]",
                f"[dim]tools:[/dim] [#f4f4f5]{tools}[/#f4f4f5]",
                f"[dim]branch:[/dim] [#f4f4f5]{esc(branch_desc)}[/#f4f4f5]",
                f"[dim]user:[/dim] [#f4f4f5]{esc(user_preview)}[/#f4f4f5]",
            ]
        )
        if tool_summary:
            lines.append(f"[dim]calls:[/dim] [#f4f4f5]{esc(tool_summary)}[/#f4f4f5]")
        if node.assistant_content:
            lines.append(
                f"[dim]assistant:[/dim] [#f4f4f5]{len(node.assistant_content)} chars[/#f4f4f5]"
            )
    return "\n".join(lines)
