from __future__ import annotations

from core.conv_tree import ConvTree
from tui.sidebar import render_sidebar_inspector_markup, render_sidebar_tree_markup
from tui.status import context_usage_percent, status_left_markup, status_right_markup, topbar_center, topbar_left, topbar_right


def test_topbar_helpers_include_workspace_branch_and_context() -> None:
    left = topbar_left("/Users/sohom/Desktop/Alphanus-Workspace", width=180)
    center = topbar_center(session_name="Session 1", branch_name="root", width=180)
    right = topbar_right(
        endpoint="http://127.0.0.1:8080/v1/chat/completions",
        context_tokens=1612,
        context_window=40960,
        width=180,
        endpoint_state="online",
    )

    assert "ALPHANUS" in left
    assert "Alphanus-Workspace" not in left
    assert "session:" in center
    assert "branch:" in center
    assert "memory:" not in center
    assert "ctx:" in right
    assert "4%" in right
    assert "127.0.0.1:8080" in right


def test_topbar_right_uses_inference_engine_context_window() -> None:
    right = topbar_right(
        endpoint="http://127.0.0.1:8080/v1/chat/completions",
        context_tokens=40960,
        context_window=40960,
        width=180,
        endpoint_state="online",
    )

    assert "100%" in right


def test_context_usage_percent_handles_rounding_and_missing_values() -> None:
    assert context_usage_percent(1612, 40960) == 4
    assert context_usage_percent(1, 40960) == 1
    assert context_usage_percent(None, 40960) is None


def test_topbar_right_handles_missing_model_usage() -> None:
    right = topbar_right(
        endpoint="http://127.0.0.1:8080/v1/chat/completions",
        context_tokens=None,
        context_window=None,
        width=180,
        endpoint_state="unknown",
    )

    assert not right.startswith(" ")
    assert "ctx:" in right
    assert "—" in right


def test_sidebar_renderers_include_tree_and_inspector_details() -> None:
    tree = ConvTree()
    turn = tree.add_turn("hello")
    tree.complete_turn(turn.id, "world")
    tree.append_skill_exchange(turn.id, {"role": "tool", "name": "open_url", "content": "ok"})

    tree_markup = render_sidebar_tree_markup(tree, width=30, selected_id=turn.id)
    inspector_markup = render_sidebar_inspector_markup(tree, width=30, selected_id=turn.id)

    assert "hello" in tree_markup
    assert "assistant:" in inspector_markup
    assert "user:" in inspector_markup
    assert "calls:" in inspector_markup
    assert "open_url" in inspector_markup


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
        width=90,
    )
    right = topbar_right(
        endpoint="http://127.0.0.1:8080/v1/chat/completions",
        context_tokens=1612,
        context_window=40960,
        width=90,
        endpoint_state="offline",
    )
    status_right = status_right_markup(
        model_name="llama-3.2-3b-instruct",
        model_state="online",
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

    assert "Alphanus-Work…" not in left
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
        model_state="offline",
        branch_armed=False,
        branch_label=None,
        thinking=True,
        width=180,
    )

    assert "model:" in status_right
    assert "Meta-Llama-3.1-8B-Instruct-Q4_K_M" in status_right
    assert "#6366f1" in status_right
    assert "llm:" in status_right
