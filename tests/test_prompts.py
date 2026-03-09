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
