from __future__ import annotations

from core.conv_tree import ConvTree
from tui.sidebar import render_sidebar_markup
from tui.status import topbar_center, topbar_left, topbar_right


def test_topbar_helpers_include_workspace_branch_and_context() -> None:
    left = topbar_left("/Users/sohom/Desktop/Alphanus-Workspace")
    center = topbar_center(branch_name="root", memory_mode="hash", focus_panel="tree")
    right = topbar_right(
        endpoint="http://127.0.0.1:8080/v1/chat/completions",
        context_tokens=321,
        context_limit=2048,
    )

    assert "ALPHANUS" in left
    assert "Alphanus-Workspace" in left
    assert "branch:" in center
    assert "memory:" in center
    assert "focus:" in center
    assert "ctx:" in right
    assert "321" in right
    assert "2048" in right


def test_sidebar_markup_includes_inspector_for_selected_node() -> None:
    tree = ConvTree()
    turn = tree.add_turn("hello")
    tree.complete_turn(turn.id, "world")

    markup = render_sidebar_markup(tree, width=30, selected_id=turn.id)

    assert "Conversation Tree" in markup
    assert "Inspector" in markup
    assert turn.id in markup
    assert "assistant:" in markup
