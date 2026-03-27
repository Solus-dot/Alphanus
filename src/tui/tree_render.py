from __future__ import annotations

from typing import List, Tuple

from core.conv_tree import ConvTree, Turn

TreeRow = Tuple[str, str, bool]


def _status_marker(node: Turn) -> str:
    state = str(getattr(node, "assistant_state", "pending") or "pending")
    if state == "pending":
        return "…"
    if state == "cancelled":
        return "✖"
    if state == "error":
        return "!"
    return "✓"


def render_tree_rows(tree: ConvTree, width: int = 80) -> List[TreeRow]:
    active_ids = {turn.id for turn in tree.active_path}
    rows: List[TreeRow] = [("● [root]", "root", True)]

    def dot(node_id: str) -> str:
        if node_id == tree.current_id:
            return "●"
        if node_id in active_ids:
            return "○"
        return "·"

    def node_line(node_id: str, depth: int) -> str:
        node = tree.nodes[node_id]
        label = f" [{node.label}]" if node.label else (" [branch]" if node.branch_root else "")
        branch = " ⎇" if node.branch_root else ""
        indent = "  " * max(0, depth)
        text = node.short(max_len=max(8, width - len(indent) - 10))
        return f"{indent}{dot(node_id)}{label}{branch} {_status_marker(node)}  {text}"

    def walk(node_id: str, depth: int) -> None:
        for child_id in tree.nodes[node_id].children:
            child = tree.nodes[child_id]
            child_depth = depth + (1 if child.branch_root else 0)
            rows.append((node_line(child_id, child_depth), child_id, child_id in active_ids))
            walk(child_id, child_depth)

    walk("root", 0)
    if len(rows) == 1:
        rows.append(("(empty)", "sub", False))
    return rows
