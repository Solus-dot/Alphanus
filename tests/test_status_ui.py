from __future__ import annotations

from core.conv_tree import ConvTree
from tui.sidebar import render_sidebar_markup
from tui.status import status_left_markup, status_right_markup, topbar_center, topbar_left, topbar_right


def test_topbar_helpers_include_workspace_branch_and_context() -> None:
    left = topbar_left("/Users/sohom/Desktop/Alphanus-Workspace", width=180)
    center = topbar_center(session_name="Session 1", branch_name="root", memory_mode="hash", width=180)
    right = topbar_right(
        endpoint="http://127.0.0.1:8080/v1/chat/completions",
        context_tokens=321,
        width=180,
    )

    assert "ALPHANUS" in left
    assert "Alphanus-Workspace" in left
    assert "session:" in center
    assert "branch:" in center
    assert "memory:" in center
    assert "ctx:" in right
    assert "321" in right
    assert "127.0.0.1:8080" in right


def test_topbar_right_handles_missing_model_usage() -> None:
    right = topbar_right(
        endpoint="http://127.0.0.1:8080/v1/chat/completions",
        context_tokens=None,
        width=180,
    )

    assert right.startswith("  ")
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
        width=180,
    )
    chat_status = status_left_markup(
        await_shell_confirm=False,
        streaming=False,
        spinner_frame="x",
        stop_requested=False,
        esc_pending=False,
        auto_follow_stream=True,
        focus_panel="chat",
        width=180,
    )

    assert "j/k move" in tree_status
    assert "enter open" in tree_status
    assert "pgup/dn scroll" in chat_status


def test_status_helpers_compact_at_small_width() -> None:
    left = topbar_left("/Users/sohom/Desktop/Alphanus-Workspace", width=90)
    center = topbar_center(
        session_name="Very Long Session Name",
        branch_name="very-long-branch-name",
        memory_mode="hash",
        width=90,
    )
    right = topbar_right(
        endpoint="http://127.0.0.1:8080/v1/chat/completions",
        context_tokens=321,
        width=90,
    )
    status_right = status_right_markup(
        model_name="llama-3.2-3b-instruct",
        branch_armed=True,
        branch_label="very-long-branch-name",
        thinking=True,
        width=80,
    )
    status_left = status_left_markup(
        await_shell_confirm=False,
        streaming=False,
        spinner_frame="x",
        stop_requested=False,
        esc_pending=False,
        auto_follow_stream=True,
        focus_panel="tree",
        width=90,
    )

    assert "Alphanus-Work…" in left
    assert "ss:" in center
    assert "br:" in center
    assert "memory:" not in center
    assert "127.0.0.1:8080" not in right
    assert "ctx:" in right
    assert "model:" in status_right
    assert "llama-3.2-3b-instruct" in status_right
    assert "files:" not in status_right
    assert "enter" in status_left


def test_status_right_markup_includes_model_label_and_value() -> None:
    status_right = status_right_markup(
        model_name="Meta-Llama-3.1-8B-Instruct-Q4_K_M",
        branch_armed=False,
        branch_label=None,
        thinking=True,
        width=180,
    )

    assert "model:" in status_right
    assert "Meta-Llama-3.1-8B-Instruct-Q4_K_M" in status_right
    assert "#6366f1" in status_right
