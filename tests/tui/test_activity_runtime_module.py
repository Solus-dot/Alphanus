from __future__ import annotations

from tui.activity_runtime import ActivityState, compact_tool_result_message, render_activity_markup, tool_result_duration_ms


def test_activity_state_tracks_tool_lifecycle() -> None:
    state = ActivityState()
    state.reset()
    state.start_tool("read_file", 'filepath="README.md"')
    state.finish_tool("read_file", ok=True, duration_ms=12)

    assert state.phase == "tools"
    assert len(state.rows) == 1
    assert state.rows[0].status == "done"
    assert state.rows[0].duration_ms == 12

    markup = render_activity_markup(state, width=30)
    assert "Turn activity" in markup
    assert "read_file" in markup
    assert "done" in markup


def test_activity_state_surfaces_failures_and_interrupts() -> None:
    state = ActivityState()
    state.reset()
    state.start_tool("shell_command", "pytest")
    state.finish_tool("shell_command", ok=False, message="exit 1", duration_ms=4)
    state.mark_interrupted()

    markup = render_activity_markup(state, width=30)

    assert state.interrupted is True
    assert "interrupted" in markup
    assert "failed" in markup
    assert "exit 1" in markup


def test_tool_result_helpers_extract_summary_fields() -> None:
    result = {"ok": False, "data": {"duration_ms": 7}, "error": {"message": "blocked"}, "meta": {"duration_ms": 5}}

    assert compact_tool_result_message(result) == "blocked"
    assert tool_result_duration_ms(result) == 5
