from __future__ import annotations

from core.conv_tree import ConvTree

TreeRow = tuple[str, str, bool]


def _truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def render_tree_rows(tree: ConvTree, width: int = 80) -> list[TreeRow]:
    active_ids = {turn.id for turn in tree.active_path}
    rows: list[TreeRow] = [("● [root]", "root", True)]

    def walk(node_id: str, depth: int) -> None:
        for child_id in tree.nodes[node_id].children:
            child = tree.nodes[child_id]
            child_depth = depth + int(child.branch_root)
            dot = "●" if child_id == tree.current_id else ("○" if child_id in active_ids else "·")
            state = str(child.assistant_state or "pending")
            status = {"pending": "…", "cancelled": "✖", "error": "!"}.get(state, "✓")
            label_text = _truncate(child.label, 8) if child.label else ("branch" if child.branch_root else "")
            label = f" [{label_text}]" if label_text else ""
            branch = " ⎇" if child.branch_root else ""
            prefix = f"{'  ' * child_depth}{dot}{label}{branch} {status}  "
            text = _truncate(child.user_text().replace("\n", " ").strip(), max(0, width - len(prefix)))
            rows.append(((prefix + text).rstrip(), child_id, child_id in active_ids))
            walk(child_id, child_depth)

    walk("root", 0)
    return rows if len(rows) > 1 else [*rows, ("(empty)", "sub", False)]
