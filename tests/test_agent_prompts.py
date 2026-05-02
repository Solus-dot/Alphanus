from __future__ import annotations

from agent.prompts import build_system_prompt


def test_system_prompt_treats_truncated_file_tool_output_as_display_limit(tmp_path):
    prompt = build_system_prompt(str(tmp_path))

    assert "truncated" in prompt
    assert "response/display limit" in prompt
    assert "read the file back first" in prompt
