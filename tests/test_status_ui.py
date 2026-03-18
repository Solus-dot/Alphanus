from __future__ import annotations

from core.conv_tree import ConvTree
from tui.sidebar import render_sidebar_markup
from tui.status import status_left_markup, topbar_center, topbar_left, topbar_right


def test_topbar_helpers_include_workspace_branch_and_context() -> None:
    left = topbar_left("/Users/sohom/Desktop/Alphanus-Workspace")
    center = topbar_center(branch_name="root", memory_mode="hash")
    right = topbar_right(
        endpoint="http://127.0.0.1:8080/v1/chat/completions",
        context_tokens=321,
    )

    assert "ALPHANUS" in left
    assert "Alphanus-Workspace" in left
    assert "branch:" in center
    assert "memory:" in center
    assert "ctx:" in right
    assert "321" in right
    assert "127.0.0.1:8080" in right


def test_topbar_right_handles_missing_model_usage() -> None:
    right = topbar_right(
        endpoint="http://127.0.0.1:8080/v1/chat/completions",
        context_tokens=None,
    )

    assert "ctx:" in right
    assert "—" in right


def test_sidebar_markup_includes_inspector_for_selected_node() -> None:
    tree = ConvTree()
    turn = tree.add_turn("hello")
    tree.complete_turn(turn.id, "world")
    tree.append_skill_exchange(turn.id, {"role": "tool", "name": "open_url", "content": "ok"})

    markup = render_sidebar_markup(tree, width=30, selected_id=turn.id)

    assert "Conversation Tree" in markup
    assert "Inspector" in markup
    assert turn.id in markup
    assert "assistant:" in markup
    assert "user:" in markup
    assert "calls:" in markup
    assert "open_url" in markup


def test_status_left_changes_with_focused_panel() -> None:
    tree_status = status_left_markup(
        await_shell_confirm=False,
        streaming=False,
        spinner_frame="x",
        stop_requested=False,
        esc_pending=False,
        auto_follow_stream=True,
        focus_panel="tree",
    )
    chat_status = status_left_markup(
        await_shell_confirm=False,
        streaming=False,
        spinner_frame="x",
        stop_requested=False,
        esc_pending=False,
        auto_follow_stream=True,
        focus_panel="chat",
    )

    assert "j/k move" in tree_status
    assert "enter open" in tree_status
    assert "pgup/dn scroll" in chat_status
