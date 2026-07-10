from __future__ import annotations

from agent.prompts import build_system_prompt


def test_system_prompt_treats_truncated_file_tool_output_as_display_limit(tmp_path):
    prompt = build_system_prompt(str(tmp_path))

    assert "truncated" in prompt
    assert "response/display limit" in prompt
    assert "read the file back first" in prompt


def test_system_prompt_requires_structured_tool_calls(tmp_path):
    prompt = build_system_prompt(str(tmp_path))

    assert "real structured tool call" in prompt
    assert "one exposed function name" in prompt
    assert "one JSON object matching that function's schema" in prompt
    assert "Do not write tool calls in assistant text" in prompt
    assert "<|tool_call>" in prompt
    assert "call:name{...}" in prompt
    assert "current tool interface" in prompt


def test_system_prompt_guides_shell_safety_without_forbidding_shell_syntax(tmp_path):
    prompt = build_system_prompt(str(tmp_path))

    assert "Do not run malicious" in prompt
    assert "system-destructive" in prompt
    assert "normal shell syntax is allowed" in prompt


def test_system_prompt_requires_loading_matching_project_skill_before_declining_file_action(tmp_path):
    prompt = build_system_prompt(str(tmp_path))

    assert "matching project skill is listed" in prompt
    assert "load that skill with `skill_view`" in prompt
    assert "manual save/copy instructions" in prompt
