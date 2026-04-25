from __future__ import annotations

from typing import Any

from textual.containers import Horizontal, ScrollableContainer, Vertical

from tui.tree_render import render_tree_rows


def tree_rows(app: Any) -> list[tuple[str, str, bool]]:
    return render_tree_rows(app.conv_tree, width=30)


def sync_tree_cursor(app: Any) -> None:
    if app._tree_cursor_id in app.conv_tree.nodes:
        return
    app._tree_cursor_id = app.conv_tree.current_id


def sidebar_target_width(width: int) -> int:
    if width < 120:
        return 0
    if width < 140:
        return 32
    return 38


def apply_sidebar_layout(app: Any, width: int) -> None:
    sidebar = app.query_one("#sidebar", Vertical)
    target_width = sidebar_target_width(width)
    sidebar.display = target_width > 0
    if target_width > 0 and hasattr(sidebar, "styles"):
        sidebar.styles.width = target_width


def apply_focus_classes(app: Any) -> None:
    chat = app.query_one("#chat-scroll", ScrollableContainer)
    sidebar = app.query_one("#sidebar", Vertical)
    input_row = app.query_one("#input-row", Horizontal)
    chat.remove_class("-active-panel")
    sidebar.remove_class("-active-panel")
    input_row.remove_class("-active-panel")
    if app._focused_panel == "chat":
        chat.add_class("-active-panel")
    elif app._focused_panel == "tree":
        sidebar.add_class("-active-panel")
    else:
        input_row.add_class("-active-panel")


def set_focused_panel(app: Any, panel: str) -> None:
    if panel == "tree" and not app.query_one("#sidebar", Vertical).display:
        panel = "chat"
    app._focused_panel = panel
    if panel == "input":
        app.query_one(app._chat_input_cls).focus()
    apply_focus_classes(app)
    app._update_topbar()


def focus_next_panel(app: Any) -> None:
    order = ["chat", "tree", "input"]
    if not app.query_one("#sidebar", Vertical).display:
        order = ["chat", "input"]
    current = order.index(app._focused_panel) if app._focused_panel in order else 0
    set_focused_panel(app, order[(current + 1) % len(order)])


def focus_prev_panel(app: Any) -> None:
    order = ["chat", "tree", "input"]
    if not app.query_one("#sidebar", Vertical).display:
        order = ["chat", "input"]
    current = order.index(app._focused_panel) if app._focused_panel in order else 0
    set_focused_panel(app, order[(current - 1) % len(order)])


def _tree_navigable_ids(app: Any) -> list[str]:
    return [tag for _text, tag, _active in tree_rows(app) if tag in app.conv_tree.nodes]


def action_tree_down(app: Any) -> None:
    if app._focused_panel != "tree":
        return
    ids = _tree_navigable_ids(app)
    if not ids:
        return
    current = ids.index(app._tree_cursor_id) if app._tree_cursor_id in ids else 0
    app._tree_cursor_id = ids[min(len(ids) - 1, current + 1)]
    app._update_sidebar()
    app._update_topbar()


def action_tree_up(app: Any) -> None:
    if app._focused_panel != "tree":
        return
    ids = _tree_navigable_ids(app)
    if not ids:
        return
    current = ids.index(app._tree_cursor_id) if app._tree_cursor_id in ids else 0
    app._tree_cursor_id = ids[max(0, current - 1)]
    app._update_sidebar()
    app._update_topbar()


def action_tree_top(app: Any) -> None:
    if app._focused_panel != "tree":
        return
    ids = _tree_navigable_ids(app)
    if not ids:
        return
    app._tree_cursor_id = ids[0]
    app._update_sidebar()
    app._update_topbar()


def action_tree_bottom(app: Any) -> None:
    if app._focused_panel != "tree":
        return
    ids = _tree_navigable_ids(app)
    if not ids:
        return
    app._tree_cursor_id = ids[-1]
    app._update_sidebar()
    app._update_topbar()


def move_tree_sibling(app: Any, direction: int) -> None:
    if app._focused_panel != "tree":
        return
    node = app.conv_tree.nodes.get(app._tree_cursor_id)
    if node is None or node.parent is None:
        return
    siblings = app.conv_tree.nodes[node.parent].children
    if app._tree_cursor_id not in siblings:
        return
    idx = siblings.index(app._tree_cursor_id) + direction
    if idx < 0 or idx >= len(siblings):
        return
    app._tree_cursor_id = siblings[idx]
    app._update_sidebar()
    app._update_topbar()


def action_tree_open(app: Any) -> None:
    if app._focused_panel != "tree":
        return
    if app._tree_cursor_id not in app.conv_tree.nodes:
        return
    app.conv_tree.current_id = app._tree_cursor_id
    app._save_active_session()
    app._rebuild_viewport()
    app._update_sidebar()
    app._update_topbar()
