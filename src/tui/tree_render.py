from __future__ import annotations

from core.conv_tree import ConvTree, Turn

TreeRow = tuple[str, str, bool]


def _status_marker(node: Turn) -> str:
    state = str(getattr(node, "assistant_state", "pending") or "pending")
    if state == "pending":
        return "…"
    if state == "cancelled":
        return "✖"
    if state == "error":
        return "!"
    return "✓"


def _truncate(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len == 1:
        return "…"
    return text[: max_len - 1] + "…"


def render_tree_rows(tree: ConvTree, width: int = 80) -> list[TreeRow]:
    active_ids = {turn.id for turn in tree.active_path}
    rows: list[TreeRow] = [("● [root]", "root", True)]

    def dot(node_id: str) -> str:
        if node_id == tree.current_id:
            return "●"
        if node_id in active_ids:
            return "○"
        return "·"

    def node_line(node_id: str, depth: int) -> str:
        node = tree.nodes[node_id]
        label_text = _truncate(node.label, 8) if node.label else ("branch" if node.branch_root else "")
        label = f" [{label_text}]" if label_text else ""
        branch = " ⎇" if node.branch_root else ""
        indent = "  " * max(0, depth)
        prefix = f"{indent}{dot(node_id)}{label}{branch} {_status_marker(node)}  "
        text_width = max(0, width - len(prefix))
        text = _truncate(node.user_text().replace("\n", " ").strip(), text_width)
        return f"{prefix}{text}".rstrip()

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
