from __future__ import annotations

from datetime import datetime
from pathlib import Path

from agent.prompts import build_system_prompt


def test_system_prompt_includes_current_date_and_workspace(tmp_path: Path):
    ws = tmp_path / "workspace"
    expected_date = datetime.now().astimezone().date().isoformat()

    prompt = build_system_prompt(str(ws))

    assert f"- Current date: {expected_date}" in prompt
    assert f"- Primary workspace: {ws.resolve()}" in prompt


def test_system_prompt_prefers_localized_edit_file_calls(tmp_path: Path):
    prompt = build_system_prompt(str(tmp_path / "workspace"))

    assert "prefer localized `edit_file` calls with `old_string` and `new_string`" in prompt
    assert "use full `content` only when replacing most of the file" in prompt


def test_system_prompt_delegates_shell_confirmation_to_tool(tmp_path: Path):
    prompt = build_system_prompt(str(tmp_path / "workspace"))

    assert "`shell_command` tool's own confirmation prompt" in prompt
    assert "instead of asking the user for duplicate confirmation" in prompt
