from __future__ import annotations

from types import SimpleNamespace

from rich.text import Text

from core.conv_tree import ConvTree
from tui.activity_runtime import ActivityState, render_activity_markup
from tui.sidebar import render_sidebar_inspector_markup, render_sidebar_tree_markup
from tui.status import context_usage_percent, status_left_markup, status_right_markup, topbar_center, topbar_left, topbar_right
from tui.view_runtime import widget_render_width


def _plain_lines(markup: str) -> list[str]:
    return Text.from_markup(markup).plain.splitlines()


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
    assert "mode:" not in right


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


def test_topbar_right_hides_backend_profile_but_keeps_integrity_warning() -> None:
    right = topbar_right(
        endpoint="http://127.0.0.1:8080/v1/chat/completions",
        context_tokens=512,
        context_window=8192,
        width=180,
        endpoint_state="online",
        backend_profile="mlx_vlm",
        model_integrity="violation",
    )

    assert "be:" not in right
    assert "mlx_vlm" not in right
    assert "int:" in right
    assert "fail" in right


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


def test_sidebar_inspector_truncates_to_visible_width() -> None:
    tree = ConvTree()
    turn = tree.add_turn("Tell me more about making a tetris game using html canvas and javascript")
    tree.complete_turn(turn.id, "plan")
    tree.append_skill_exchange(turn.id, {"role": "tool", "name": "very_long_tool_name_for_sidebar", "content": "ok"})

    lines = _plain_lines(render_sidebar_inspector_markup(tree, width=24, selected_id=turn.id))

    assert lines
    assert all(len(line) <= 24 for line in lines)


def test_activity_markup_truncates_to_visible_width() -> None:
    state = ActivityState()
    state.reset()
    state.start_tool("very_long_tool_name_for_sidebar", 'query="Tell me more about making a tetris game"')
    state.finish_tool("very_long_tool_name_for_sidebar", ok=False, message="a long failure message for narrow sidebars")

    lines = _plain_lines(render_activity_markup(state, width=24))

    assert lines
    assert all(len(line) <= 24 for line in lines)


def test_widget_render_width_prefers_content_region() -> None:
    widget = SimpleNamespace(
        content_region=SimpleNamespace(width=27),
        region=SimpleNamespace(width=40),
        size=SimpleNamespace(width=50),
    )

    assert widget_render_width(widget, fallback=30) == 27


def test_activity_markup_renders_live_tool_rows() -> None:
    state = ActivityState()
    state.reset()
    state.start_tool("git_status", 'path="repo"')
    state.finish_tool("git_status", ok=True, duration_ms=9)

    markup = render_activity_markup(state, width=30)

    assert "Turn activity" in markup
    assert "git_status" in markup
    assert "done" in markup


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


def test_status_left_input_focus_includes_file_shortcut_hint() -> None:
    status_left = status_left_markup(
        await_shell_confirm=False,
        streaming=False,
        spinner_frame="x",
        stop_requested=False,
        esc_pending=False,
        auto_follow_stream=True,
        focus_panel="input",
        width=180,
    )

    assert "ctrl+f file" in status_left


def test_status_left_streaming_copy_distinguishes_interrupt_states() -> None:
    normal = status_left_markup(
        await_shell_confirm=False,
        streaming=True,
        spinner_frame="x",
        stop_requested=False,
        esc_pending=False,
        auto_follow_stream=True,
        focus_panel="input",
        width=180,
    )
    armed = status_left_markup(
        await_shell_confirm=False,
        streaming=True,
        spinner_frame="x",
        stop_requested=False,
        esc_pending=True,
        auto_follow_stream=True,
        focus_panel="input",
        width=180,
    )
    stopping = status_left_markup(
        await_shell_confirm=False,
        streaming=True,
        spinner_frame="x",
        stop_requested=True,
        esc_pending=False,
        auto_follow_stream=True,
        focus_panel="input",
        width=180,
    )

    assert "esc stop" in normal
    assert "esc to confirm" in armed
    assert "stopping after current step" in stopping


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
    assert "mode:" in status_right
    assert "#6366f1" in status_right
