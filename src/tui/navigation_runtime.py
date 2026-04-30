from __future__ import annotations

from typing import Any

from textual.containers import Horizontal, ScrollableContainer, Vertical

from tui.tree_render import render_tree_rows

def tree_rows(app: Any) -> list[tuple[str, str, bool]]: return render_tree_rows(app.conv_tree, width=30)
def sync_tree_cursor(app: Any) -> None:
    if app._tree_cursor_id not in app.conv_tree.nodes: app._tree_cursor_id = app.conv_tree.current_id

def sidebar_target_width(width: int) -> int:
    return 0 if width < 120 else 32 if width < 140 else 38

def apply_sidebar_layout(app: Any, width: int) -> None:
    sidebar = app.query_one("#sidebar", Vertical)
    target_width = sidebar_target_width(width)
    sidebar.display = target_width > 0
    if target_width > 0 and hasattr(sidebar, "styles"): sidebar.styles.width = target_width
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
    if panel == "input": app.query_one(app._chat_input_cls).focus()
    apply_focus_classes(app); app._update_topbar()
def _shift_focus(app: Any, direction: int) -> None:
    order = ["chat", "input"] if not app.query_one("#sidebar", Vertical).display else ["chat", "tree", "input"]
    current = order.index(app._focused_panel) if app._focused_panel in order else 0
    set_focused_panel(app, order[(current + direction) % len(order)])
def focus_next_panel(app: Any) -> None: _shift_focus(app, 1)
def focus_prev_panel(app: Any) -> None: _shift_focus(app, -1)
def _active_tree_ids(app: Any) -> list[str]:
    return [tag for _text, tag, _active in tree_rows(app) if tag in app.conv_tree.nodes] if app._focused_panel == "tree" else []

def _set_tree_cursor(app: Any, target_id: str) -> None:
    app._tree_cursor_id = target_id
    for update in (app._update_sidebar, app._update_topbar): update()

def _move_tree_cursor(app: Any, delta: int) -> None:
    ids = _active_tree_ids(app)
    if not ids: return
    current = ids.index(app._tree_cursor_id) if app._tree_cursor_id in ids else 0
    _set_tree_cursor(app, ids[max(0, min(len(ids) - 1, current + delta))])
def action_tree_down(app: Any) -> None: _move_tree_cursor(app, 1)
def action_tree_up(app: Any) -> None: _move_tree_cursor(app, -1)
def _move_tree_edge(app: Any, *, to_bottom: bool) -> None:
    ids = _active_tree_ids(app)
    if not ids: return
    _set_tree_cursor(app, ids[-1 if to_bottom else 0])
def action_tree_top(app: Any) -> None: _move_tree_edge(app, to_bottom=False)
def action_tree_bottom(app: Any) -> None: _move_tree_edge(app, to_bottom=True)
def move_tree_sibling(app: Any, direction: int) -> None:
    if app._focused_panel != "tree":
        return
    node = app.conv_tree.nodes.get(app._tree_cursor_id);
    if node is None or node.parent is None: return
    siblings = app.conv_tree.nodes[node.parent].children;
    if app._tree_cursor_id not in siblings: return
    idx = siblings.index(app._tree_cursor_id) + direction;
    if not 0 <= idx < len(siblings): return
    _set_tree_cursor(app, siblings[idx])
def action_tree_open(app: Any) -> None:
    if app._focused_panel != "tree" or app._tree_cursor_id not in app.conv_tree.nodes:
        return
    app.conv_tree.current_id = app._tree_cursor_id; app._save_active_session(); app._rebuild_viewport()
    for update in (app._update_sidebar, app._update_topbar): update()
