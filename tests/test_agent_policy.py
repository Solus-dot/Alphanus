from __future__ import annotations

from pathlib import Path

import pytest

from agent.core import Agent
from core.types import TurnClassification, TurnPolicySnapshot
from skills.runtime import SkillContext, SkillRuntime
from tests.support import build_skill_runtime

pytestmark = pytest.mark.usefixtures("disable_model_classification")

TEST_BASE_URL = "http://127.0.0.1:8080"
TEST_MODEL_ENDPOINT = f"{TEST_BASE_URL}/v1/chat/completions"
TEST_MODELS_ENDPOINT = f"{TEST_BASE_URL}/v1/models"
TEST_SLOTS_ENDPOINT = f"{TEST_BASE_URL}/slots"
TEST_PROPS_ENDPOINT = f"{TEST_BASE_URL}/props"


def agent_config(*, sections=None, **overrides):
    agent = {
        "model_endpoint": TEST_MODEL_ENDPOINT,
        "models_endpoint": TEST_MODELS_ENDPOINT,
        "request_timeout_s": 5,
        "readiness_timeout_s": 1,
        "readiness_poll_s": 0.05,
        "enable_thinking": True,
        "tls_verify": True,
        "max_tokens": 256,
    }
    agent.update(overrides)
    config = {"agent": agent}
    config.update(sections or {})
    return config


class FakeResponse:
    def __init__(self, lines):
        self.lines = [line.encode("utf-8") for line in lines]
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.lines)

    def read(self):
        return b"".join(self.lines)


@pytest.fixture
def runtime(tmp_path: Path) -> SkillRuntime:
    return build_skill_runtime(
        tmp_path,
        manifest="""
---
name: project-ops
description: project
version: 1.0.0
tools:
  allowed-tools:
    - create_directory
    - create_file
---
project
""",
        tools="""
TOOL_SPECS = {
  "create_directory": {
    "capability": "project_write",
    "description": "Create directory",
    "parameters": {
      "type": "object",
      "properties": {"path": {"type": "string"}},
      "required": ["path"]
    }
  },
  "create_file": {
    "capability": "project_write",
    "description": "Create file",
    "parameters": {
      "type": "object",
      "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}},
      "required": ["filepath", "content"]
    }
  },
}

def execute(tool_name, args, env):
    if tool_name == "create_directory":
        path = env.project.create_directory(args["path"])
        return {"ok": True, "data": {"filepath": path}, "error": None, "meta": {}}
    if tool_name == "create_file":
        path = env.project.create_file(args["filepath"], args["content"])
        return {"ok": True, "data": {"filepath": path}, "error": None, "meta": {}}
    return {"ok": False, "data": None, "error": {"code": "E_UNSUPPORTED", "message": "nope"}, "meta": {}}
""",
    )


def test_run_turn_allows_same_host_endpoints_with_different_ports(mocker, runtime: SkillRuntime):
    agent = Agent(
        {
            "agent": {
                "model_endpoint": TEST_MODEL_ENDPOINT,
                "models_endpoint": "http://127.0.0.1:9000/v1/models",
                "allow_cross_host_endpoints": False,
            },
        },
        runtime,
    )
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    mocker.patch.object(
        agent.llm_client,
        "call_with_retry",
        return_value=type(
            "R",
            (),
            {
                "finish_reason": "stop",
                "content": "ok",
                "reasoning": "",
                "tool_calls": [],
            },
        )(),
    )

    result = agent.run_turn(
        history_messages=[],
        user_input="hello",
        thinking=True,
    )

    assert result.status == "done"
    assert result.content == "ok"


def test_single_non_search_tool_runs_second_pass_even_if_search_skill_is_selected(mocker, runtime: SkillRuntime):
    utilities = runtime.skills_dir / "utilities"
    utilities.mkdir(parents=True)
    (utilities / "SKILL.md").write_text(
        """
---
name: utilities
description: utility tools
version: 1.0.0
tools:
  allowed-tools:
    - get_weather
---
utilities
""".strip(),
        encoding="utf-8",
    )
    (utilities / "tools.py").write_text(
        """
TOOL_SPECS = {
  "get_weather": {
    "capability": "utility_weather",
    "description": "Get weather",
    "parameters": {
      "type": "object",
      "properties": {"city": {"type": "string"}},
      "required": ["city"]
    }
  }
}

def execute(tool_name, args, env):
    return {"ok": True, "data": {"city": "London", "temp_c": 14, "desc": "Cloudy"}, "error": None, "meta": {}}
""".strip(),
        encoding="utf-8",
    )
    search_skill = runtime.skills_dir / "search-ops"
    search_skill.mkdir(parents=True)
    (search_skill / "SKILL.md").write_text(
        """
---
name: search-ops
description: web research
version: 1.0.0
---
search
""".strip(),
        encoding="utf-8",
    )
    runtime.load_skills()
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    selected = [runtime.get_skill("search-ops"), runtime.get_skill("utilities")]
    mocker.patch.object(agent.skill_runtime, "select_skills", return_value=selected)

    calls = []

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        calls.append(pass_id)
        if pass_id == "pass_1":
            return type(
                "R",
                (),
                {
                    "finish_reason": "tool_calls",
                    "content": "",
                    "reasoning": "",
                    "tool_calls": [
                        type(
                            "Call",
                            (),
                            {"stream_id": "1", "index": 0, "id": "call_1", "name": "get_weather", "arguments": {"city": "London"}},
                        )()
                    ],
                },
            )()
        if pass_id == "pass_2":
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "The current weather in London is 14 C with Cloudy.",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        raise AssertionError(f"Unexpected pass id: {pass_id}")

    mocker.patch.object(agent.llm_client, "call_with_retry", side_effect=fake_call_with_retry)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "what is the weather in London?"}],
        user_input="what is the weather in London?",
        thinking=True,
    )

    assert result.status == "done"
    assert result.content == "The current weather in London is 14 C with Cloudy."
    assert calls == ["pass_1", "pass_2"]


def test_compound_weather_question_does_not_stop_after_first_tool_result(mocker, runtime: SkillRuntime):
    utilities = runtime.skills_dir / "utilities"
    utilities.mkdir(parents=True)
    (utilities / "SKILL.md").write_text(
        """
---
name: utilities
description: utility tools
version: 1.0.0
tools:
  allowed-tools:
    - get_weather
---
utilities
""".strip(),
        encoding="utf-8",
    )
    runtime.load_skills()
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    selected = [runtime.get_skill("utilities")]
    mocker.patch.object(agent.skill_runtime, "select_skills", return_value=selected)

    calls = []

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        calls.append(pass_id)
        if pass_id == "pass_1":
            return type(
                "R",
                (),
                {
                    "finish_reason": "tool_calls",
                    "content": "",
                    "reasoning": "",
                    "tool_calls": [
                        type(
                            "Call",
                            (),
                            {"stream_id": "1", "index": 0, "id": "call_1", "name": "get_weather", "arguments": {"city": "Bengaluru"}},
                        )()
                    ],
                },
            )()
        if pass_id == "pass_2":
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "It is currently 21 C and clear in Bengaluru. You likely do not need an umbrella this evening unless the forecast changes later.",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        raise AssertionError(f"Unexpected pass id: {pass_id}")

    mocker.patch.object(agent.llm_client, "call_with_retry", side_effect=fake_call_with_retry)
    mocker.patch.object(
        runtime,
        "execute_tool_call",
        return_value={
            "ok": True,
            "data": {"city": "Bengaluru", "temp_c": 21, "desc": "Clear", "humidity": 73},
            "error": None,
            "meta": {},
        },
    )
    mocker.patch.object(runtime, "tools_for_turn", return_value=[{"type": "function", "function": {"name": "get_weather"}}])

    result = agent.run_turn(
        history_messages=[],
        user_input="What's the weather in Bengaluru right now, and should I carry an umbrella this evening?",
        thinking=True,
    )

    assert result.status == "done"
    assert "umbrella" in result.content.lower()
    assert calls == ["pass_1", "pass_2"]


def test_batch_project_delete_does_not_stop_after_first_successful_tool(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    selected = [runtime.get_skill("project-ops")]
    mocker.patch.object(agent.skill_runtime, "select_skills", return_value=selected)

    calls = []

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        calls.append(pass_id)
        if pass_id == "pass_1":
            return type(
                "R",
                (),
                {
                    "finish_reason": "tool_calls",
                    "content": "",
                    "reasoning": "",
                    "tool_calls": [
                        type(
                            "Call",
                            (),
                            {"stream_id": "1", "index": 0, "id": "call_1", "name": "delete_path", "arguments": {"path": "a.txt"}},
                        )()
                    ],
                },
            )()
        if pass_id == "pass_2":
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "Finished deleting all requested files.",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        raise AssertionError(f"Unexpected pass id: {pass_id}")

    mocker.patch.object(agent.llm_client, "call_with_retry", side_effect=fake_call_with_retry)
    mocker.patch.object(
        runtime,
        "execute_tool_call",
        return_value={"ok": True, "data": {"filepath": "/tmp/a.txt", "kind": "file"}, "error": None, "meta": {}},
    )
    mocker.patch.object(runtime, "tools_for_turn", return_value=[{"type": "function", "function": {"name": "delete_path"}}])

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "delete all files in project"}],
        user_input="delete all files in project",
        thinking=True,
    )

    assert result.status == "done"
    assert result.content == "Finished deleting all requested files."
    assert calls == ["pass_1", "pass_2"]


def test_local_project_tasks_prefer_project_tools_but_still_block_fetch_tools(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    mocker.patch.object(agent.classifier, "classify", return_value=TurnClassification(prefer_local_project_tools=True))
    selected = [runtime.get_skill("project-ops")]
    mocker.patch.object(agent.skill_runtime, "select_skills", return_value=selected)

    calls = []

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        calls.append(pass_id)
        if pass_id == "pass_1":
            return type(
                "R",
                (),
                {
                    "finish_reason": "tool_calls",
                    "content": "",
                    "reasoning": "",
                    "tool_calls": [
                        type(
                            "Call",
                            (),
                            {
                                "stream_id": "1",
                                "index": 0,
                                "id": "call_1",
                                "name": "shell_command",
                                "arguments": {"command": "mkdir -p 1738"},
                            },
                        )(),
                        type(
                            "Call",
                            (),
                            {"stream_id": "1", "index": 1, "id": "call_2", "name": "fetch_url", "arguments": {"url": "/tmp/index.html"}},
                        )(),
                    ],
                },
            )()
        if pass_id == "pass_2":
            return type(
                "R",
                (),
                {
                    "finish_reason": "tool_calls",
                    "content": "",
                    "reasoning": "",
                    "tool_calls": [
                        type(
                            "Call",
                            (),
                            {"stream_id": "2", "index": 0, "id": "call_3", "name": "create_directory", "arguments": {"path": "1738"}},
                        )(),
                        type(
                            "Call",
                            (),
                            {
                                "stream_id": "2",
                                "index": 1,
                                "id": "call_4",
                                "name": "create_file",
                                "arguments": {"filepath": "1738/index.html", "content": "<html></html>"},
                            },
                        )(),
                        type(
                            "Call",
                            (),
                            {
                                "stream_id": "2",
                                "index": 2,
                                "id": "call_5",
                                "name": "create_file",
                                "arguments": {"filepath": "1738/script.js", "content": "console.log(1)\n"},
                            },
                        )(),
                    ],
                },
            )()
        if pass_id == "pass_3":
            return type("R", (), {"finish_reason": "stop", "content": "Done.", "reasoning": "", "tool_calls": []})()
        raise AssertionError(f"Unexpected pass id: {pass_id}")

    mocker.patch.object(agent.llm_client, "call_with_retry", side_effect=fake_call_with_retry)
    executed = []
    real_execute = runtime.execute_tool_call

    def wrapped_execute(tool_name, args, selected, ctx, request_approval=None, **_kwargs):
        executed.append(tool_name)
        return real_execute(tool_name, args, selected=selected, ctx=ctx, request_approval=request_approval)

    mocker.patch.object(runtime, "execute_tool_call", side_effect=wrapped_execute)
    mocker.patch.object(
        runtime,
        "tools_for_turn",
        return_value=[
            {"type": "function", "function": {"name": "create_directory"}},
            {"type": "function", "function": {"name": "create_file"}},
            {"type": "function", "function": {"name": "shell_command"}},
            {"type": "function", "function": {"name": "fetch_url"}},
        ],
    )

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "create a bakery landing page in 1738 with html css js"}],
        user_input="Where is JS?",
        thinking=True,
    )

    assert result.status == "done"
    assert executed == ["shell_command", "create_directory", "create_file", "create_file"]
    assert calls == ["pass_1", "pass_2", "pass_3"]


def test_project_action_accepts_successful_mutating_shell_command(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    mocker.patch.object(
        agent.classifier,
        "classify",
        return_value=TurnClassification(requires_project_action=True, prefer_local_project_tools=True),
    )
    selected = [runtime.get_skill("project-ops")]
    mocker.patch.object(agent.skill_runtime, "select_skills", return_value=selected)

    calls = []

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        calls.append(pass_id)
        if pass_id == "pass_1":
            return type(
                "R",
                (),
                {
                    "finish_reason": "tool_calls",
                    "content": "",
                    "reasoning": "",
                    "tool_calls": [
                        type(
                            "Call",
                            (),
                            {
                                "stream_id": "1",
                                "index": 0,
                                "id": "call_1",
                                "name": "shell_command",
                                "arguments": {"command": "mkdir -p 1738"},
                            },
                        )(),
                    ],
                },
            )()
        if pass_id == "pass_2":
            return type("R", (), {"finish_reason": "stop", "content": "Done.", "reasoning": "", "tool_calls": []})()
        raise AssertionError(f"Unexpected pass id: {pass_id}")

    mocker.patch.object(agent.llm_client, "call_with_retry", side_effect=fake_call_with_retry)
    mocker.patch.object(
        runtime,
        "tools_for_turn",
        return_value=[
            {"type": "function", "function": {"name": "create_directory"}},
            {"type": "function", "function": {"name": "shell_command"}},
        ],
    )
    mocker.patch.object(
        runtime,
        "execute_tool_call",
        side_effect=lambda tool_name, args, selected, ctx, request_approval=None, **_kwargs: runtime.project.run_shell_command(
            args["command"]
        ),
    )

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "create a folder named 1738"}],
        user_input="yes",
        thinking=True,
        request_approval=lambda _command: True,
    )

    assert result.status == "done"
    assert result.content == "Done."
    assert (runtime.project.project_root / "1738").is_dir()
    assert calls == ["pass_1", "pass_2"]


def test_project_action_allows_snapshotting_around_shell_command(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    mocker.patch.object(
        agent.classifier,
        "classify",
        return_value=TurnClassification(requires_project_action=True, prefer_local_project_tools=True),
    )
    selected = [runtime.get_skill("project-ops")]
    mocker.patch.object(agent.skill_runtime, "select_skills", return_value=selected)
    mocker.patch.object(
        runtime,
        "tools_for_turn",
        return_value=[
            {"type": "function", "function": {"name": "create_directory"}},
            {"type": "function", "function": {"name": "shell_command"}},
        ],
    )

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        if pass_id == "pass_1":
            return type(
                "R",
                (),
                {
                    "finish_reason": "tool_calls",
                    "content": "",
                    "reasoning": "",
                    "tool_calls": [
                        type(
                            "Call",
                            (),
                            {
                                "stream_id": "1",
                                "index": 0,
                                "id": "call_1",
                                "name": "shell_command",
                                "arguments": {"command": "mkdir -p 1738"},
                            },
                        )(),
                    ],
                },
            )()
        if pass_id == "pass_2":
            return type("R", (), {"finish_reason": "stop", "content": "Done.", "reasoning": "", "tool_calls": []})()
        raise AssertionError(f"Unexpected pass id: {pass_id}")

    mocker.patch.object(agent.llm_client, "call_with_retry", side_effect=fake_call_with_retry)
    mocker.patch.object(
        runtime,
        "execute_tool_call",
        side_effect=lambda tool_name, args, selected, ctx, request_approval=None, **_kwargs: runtime.project.run_shell_command(
            args["command"]
        ),
    )

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "create a folder named 1738"}],
        user_input="yes",
        thinking=True,
        request_approval=lambda _command: True,
    )

    assert result.status == "done"
    assert (runtime.project.project_root / "1738").is_dir()


def test_explicit_external_path_disables_local_project_routing(runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    other_path = str(Path(runtime.project.project_root).parent / "other-project")
    ctx = SkillContext(
        user_input=f"Update the packages in {other_path}, it uses uv",
        branch_labels=[],
        attachments=[],
        project_root=str(runtime.project.project_root),
        memory_hits=[],
    )

    assert agent.classifier._explicit_path_outside_project(ctx.user_input) == other_path
    assert agent.classifier.classify(ctx).prefer_local_project_tools is False


def test_explicit_external_path_adds_prompt_rule_and_skips_local_project_rule(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    selected = [runtime.get_skill("project-ops")]
    mocker.patch.object(agent.skill_runtime, "select_skills", return_value=selected)
    mocker.patch.object(runtime, "tools_for_turn", return_value=[])
    other_path = str(Path(runtime.project.project_root).parent / "other-project")

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        system_text = payload["messages"][0]["content"]
        assert "Explicit path rule:" in system_text
        assert other_path in system_text
        assert "Local project tool rule:" not in system_text
        return type("R", (), {"finish_reason": "stop", "content": "Need confirmation.", "reasoning": "", "tool_calls": []})()

    mocker.patch.object(agent.llm_client, "call_with_retry", side_effect=fake_call_with_retry)

    result = agent.run_turn(
        history_messages=[],
        user_input=f"Update the packages in {other_path}, it uses uv",
        thinking=True,
    )

    assert result.status == "done"
    assert result.content == "Need confirmation."


def test_policy_rules_require_shell_tool_exposure_for_external_path_guidance(runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    rules = agent.prompt_renderer.render_policy_rules(
        TurnPolicySnapshot(
            explicit_external_path="/tmp/other-project",
            prefer_local_project_tools=True,
            shell_tool_exposed=False,
        )
    )

    assert "single shell command" not in rules
    assert "shell_command is still available" not in rules
    assert "no shell tool is exposed in this turn" in rules.lower()


def test_policy_rules_include_shell_guidance_only_when_shell_tool_is_exposed(runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    rules = agent.prompt_renderer.render_policy_rules(
        TurnPolicySnapshot(
            explicit_external_path="/tmp/other-project",
            prefer_local_project_tools=True,
            shell_tool_exposed=True,
        )
    )

    assert "use the exposed shell tool" in rules
    assert "A shell tool is exposed in this turn" in rules
    assert "shell_command is still available" not in rules


def test_explicit_external_path_ignores_urls(runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)

    assert agent.classifier._explicit_path_outside_project("Use https://example.com as inspiration for the landing page") == ""


def test_explicit_external_path_supports_quoted_paths_with_spaces(runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    other_path = str(Path(runtime.project.project_root).parent / "Other Project")
    ctx = SkillContext(
        user_input=f'Update the packages in "{other_path}", it uses uv',
        branch_labels=[],
        attachments=[],
        project_root=str(runtime.project.project_root),
        memory_hits=[],
    )

    assert agent.classifier._explicit_path_outside_project(ctx.user_input) == other_path
    assert agent.classifier.classify(ctx).prefer_local_project_tools is False
