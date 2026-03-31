from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from agent.core import Agent
from agent.types import TurnClassification
from core.memory import VectorMemory
from core.skills import SkillContext, SkillRuntime
from core.workspace import WorkspaceManager


class FakeResponse:
    def __init__(self, lines):
        self.lines = [l.encode("utf-8") for l in lines]
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self.lines)

    def read(self):
        return b"".join(self.lines)


@pytest.fixture
def runtime(tmp_path: Path) -> SkillRuntime:
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "workspace-ops").mkdir(parents=True)

    (skills / "workspace-ops" / "SKILL.md").write_text(
        """
---
name: workspace-ops
description: workspace
version: 1.0.0
tools:
  allowed-tools:
    - create_directory
    - create_file
    - create_files
---
workspace
""".strip(),
        encoding="utf-8",
    )
    (skills / "workspace-ops" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "create_directory": {
    "capability": "workspace_write",
    "description": "Create directory",
    "parameters": {
      "type": "object",
      "properties": {"path": {"type": "string"}},
      "required": ["path"]
    }
  },
  "create_file": {
    "capability": "workspace_write",
    "description": "Create file",
    "parameters": {
      "type": "object",
      "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}},
      "required": ["filepath", "content"]
    }
  },
  "create_files": {
    "capability": "workspace_write",
    "description": "Create files",
    "parameters": {
      "type": "object",
      "properties": {"files": {"type": "array"}},
      "required": ["files"]
    }
  }
}

def execute(tool_name, args, env):
    if tool_name == "create_directory":
        path = env.workspace.create_directory(args["path"])
        return {"ok": True, "data": {"filepath": path}, "error": None, "meta": {}}
    if tool_name == "create_file":
        path = env.workspace.create_file(args["filepath"], args["content"])
        return {"ok": True, "data": {"filepath": path}, "error": None, "meta": {}}
    if tool_name == "create_files":
        created = []
        for item in args.get("files", []):
            path = env.workspace.create_file(item["filepath"], item["content"])
            created.append({"filepath": path})
        return {"ok": True, "data": {"created": created, "count": len(created)}, "error": None, "meta": {}}
    return {"ok": False, "data": None, "error": {"code": "E_UNSUPPORTED", "message": "nope"}, "meta": {}}
""".strip(),
        encoding="utf-8",
    )

    return SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )


def test_agent_tool_call_loop(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
        }
    }
    agent = Agent(cfg, runtime)

    calls = []

    def fake_urlopen(req, timeout=None, context=None):
        calls.append(req)
        url = req.full_url
        if url.endswith("/v1/models"):
            return FakeResponse([])
        if len(calls) == 2:
            # First completion emits tool_calls.
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"create_file","arguments":"{\\"filepath\\": \\\"a.txt\\\", \\\"content\\\": \\\"hello\\\"}"}}]}}]}',
                    'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )
        # Second completion returns final answer.
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"Done"}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "write a file"}],
        user_input="write a file",
        thinking=True,
    )

    assert result.status == "done"
    assert "Done" in result.content
    assert any(msg.get("role") == "tool" for msg in result.skill_exchanges)

    # Ensure POST payload uses stream=true.
    post_req = calls[1]
    body = json.loads(post_req.data.decode("utf-8"))
    assert body["stream"] is True
    assert body["messages"][0]["role"] == "system"
    assert sum(1 for m in body["messages"] if m.get("role") == "system") == 1
    assert "temperature" not in body
    assert "top_p" not in body
    assert "top_k" not in body
    assert "min_p" not in body
    assert "repetition_penalty" not in body


def test_agent_records_model_usage_from_stream(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
        }
    }
    agent = Agent(cfg, runtime)

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"Done"}}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":321,"completion_tokens":12,"total_tokens":333}}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "say done"}],
        user_input="say done",
        thinking=True,
    )

    assert result.status == "done"
    assert result.journal["model_usage"]["prompt_tokens"] == 321
    assert result.journal["model_usage"]["completion_tokens"] == 12


def test_agent_requests_final_answer_if_post_tool_content_empty(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
        }
    }
    agent = Agent(cfg, runtime)

    chat_reqs = []
    events = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"create_file","arguments":"{\\"filepath\\": \\\"a.txt\\\", \\\"content\\\": \\\"hello\\\"}"}}]}}]}',
                    'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )
        if len(chat_reqs) == 2:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"reasoning_content":"I should now answer the user."}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        if len(chat_reqs) == 3:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"content":"Done"}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        raise AssertionError("Unexpected extra completion call")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "write a file"}],
        user_input="write a file",
        thinking=True,
        on_event=events.append,
    )

    assert result.status == "done"
    assert result.content == "Done"
    assert len(chat_reqs) == 3

    finalize_payload = json.loads(chat_reqs[2].data.decode("utf-8"))
    assert finalize_payload.get("chat_template_kwargs", {}).get("enable_thinking") is False
    assert "tools" not in finalize_payload
    assert "tool_choice" not in finalize_payload
    assert any(evt.get("type") == "reasoning_token" for evt in events)
    assert any(
        evt.get("type") == "pass_end"
        and evt.get("finish_reason") == "stop"
        and not evt.get("has_content")
        and not evt.get("has_tool_calls")
        for evt in events
    )


def test_agent_does_not_finish_after_helper_file_for_opaque_artifact_request(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
        }
    }
    agent = Agent(cfg, runtime)
    chat_reqs = []

    mocker.patch.object(agent, "_select_skills", return_value=runtime.enabled_skills())

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"create_file","arguments":"{\\"filepath\\": \\\"northstar-proposal.js\\\", \\\"content\\\": \\\"console.log(1)\\\"}"}}]}}]}',
                    'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )
        if len(chat_reqs) == 2:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"content":"Continuing with the DOCX workflow."}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        raise AssertionError("Unexpected extra completion call")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "create northstar-proposal.docx"}],
        user_input="create northstar-proposal.docx",
        thinking=True,
    )

    assert result.status == "done"
    assert result.content == "Continuing with the DOCX workflow."
    assert len(chat_reqs) == 2


def test_runtime_list_skills_refreshes_after_generation_changes(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "alpha").mkdir(parents=True)

    (skills / "alpha" / "SKILL.md").write_text(
        """
---
name: alpha
description: Alpha skill.
---
Alpha.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )
    assert [skill.id for skill in runtime.enabled_skills()] == ["alpha"]

    (skills / "beta").mkdir(parents=True)
    (skills / "beta" / "SKILL.md").write_text(
        """
---
name: beta
description: Beta skill.
---
Beta.
""".strip(),
        encoding="utf-8",
    )
    runtime.load_skills()

    assert runtime.generation >= 2
    assert [skill.id for skill in runtime.enabled_skills()] == ["alpha", "beta"]


def test_agent_reload_skills_returns_latest_generation(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "alpha").mkdir(parents=True)

    (skills / "alpha" / "SKILL.md").write_text(
        """
---
name: alpha
description: Alpha skill.
---
Alpha.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )
    agent = Agent({"agent": {}}, runtime)

    generation = agent.reload_skills()

    assert generation == runtime.generation
    assert [skill.id for skill in runtime.enabled_skills()] == ["alpha"]


def test_skill_toggle_bumps_generation_and_updates_enabled_skills(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "alpha").mkdir(parents=True)

    (skills / "alpha" / "SKILL.md").write_text(
        """
---
name: alpha
description: Alpha skill.
---
Alpha.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )
    generation_before = runtime.generation

    assert runtime.set_enabled("alpha", False) is True

    assert runtime.generation == generation_before + 1
    assert runtime.enabled_skills() == []


def test_confirmation_turn_reuses_immediate_prior_skill_context(mocker, runtime: SkillRuntime):
    runtime.config = {}
    agent = Agent({"agent": {}}, runtime)

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        return type("R", (), {"finish_reason": "stop", "content": '{"skills":["workspace-ops"]}'})()

    mocker.patch.object(agent, "_call_with_retry", side_effect=fake_call_with_retry)

    history_messages = [
        {"role": "user", "content": "delete all files in the workspace"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "create_file",
                        "arguments": '{"filepath":"a.txt","content":"hello"}',
                    }
                }
            ],
        },
        {"role": "tool", "name": "create_file", "content": '{"ok": true, "data": {"filepath": "a.txt"}}'},
    ]

    ctx = agent._build_skill_context("yes", [], [], history_messages)

    assert "previous user request: delete all files in the workspace" in ctx.recent_routing_hint
    assert "tools just used: create_file" in ctx.recent_routing_hint
    assert ctx.sticky_skill_ids == ["workspace-ops"]

    assert agent._select_skills(ctx, threading.Event()) == []


def test_contextual_followup_reuses_immediate_prior_skill_context(mocker, runtime: SkillRuntime):
    runtime.config = {}
    agent = Agent({"agent": {}}, runtime)

    history_messages = [
        {"role": "user", "content": "create a bakery landing page in a folder called 1738 with html css js"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "create_file",
                        "arguments": '{"filepath":"1738/index.html","content":"<html></html>"}',
                    }
                }
            ],
        },
        {"role": "tool", "name": "create_file", "content": '{"ok": true, "data": {"filepath": "1738/index.html"}}'},
    ]

    ctx = agent._build_skill_context("Where is JS?", [], [], history_messages)

    assert agent._select_skills(ctx, threading.Event()) == []


def test_contextual_followup_seed_reuses_immediate_prior_skill_context(runtime: SkillRuntime):
    runtime.config = {}
    agent = Agent({"agent": {}}, runtime)

    history_messages = [
        {"role": "user", "content": "create a bakery landing page in a folder called 1738 with html css js"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "create_file",
                        "arguments": '{"filepath":"1738/index.html","content":"<html></html>"}',
                    }
                }
            ],
        },
        {"role": "tool", "name": "create_file", "content": '{"ok": true, "data": {"filepath": "1738/index.html"}}'},
    ]

    ctx = agent._build_skill_context("Where is JS?", [], [], history_messages)

    assert agent._select_skills(ctx, threading.Event()) == []


def test_confirmation_workspace_action_retries_instead_of_accepting_manual_terminal_advice(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    mocker.patch.object(agent, "_classify_turn", return_value=TurnClassification(requires_workspace_action=True, followup_kind="confirmation"))

    calls = []

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        calls.append(pass_id)
        if pass_id == "pass_1_workspace_action_outcome":
            return type("R", (), {"finish_reason": "stop", "content": '{"outcome":"not_completed"}'})()
        if pass_id == "pass_1":
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "I cannot create files directly here. You can run touch notes.txt manually in the terminal.",
                    "reasoning": "",
                    "tool_calls": [],
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
                    "tool_calls": [type("Call", (), {"stream_id": "2", "index": 0, "id": "call_2", "name": "create_file", "arguments": {"filepath": "notes.txt", "content": "hello"}})()],
                },
            )()
        if pass_id == "pass_3":
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "Done with workspace tools.",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        raise AssertionError(f"Unexpected pass id: {pass_id}")

    def fake_execute_tool_call(tool_name, args, selected, ctx, confirm_shell=None, **_kwargs):
        assert tool_name == "create_file"
        return {"ok": True, "data": {"filepath": "notes.txt"}, "error": None, "meta": {}}

    mocker.patch.object(agent, "_call_with_retry", side_effect=fake_call_with_retry)
    mocker.patch.object(runtime, "execute_tool_call", side_effect=fake_execute_tool_call)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "create notes.txt in the workspace"}],
        user_input="yes",
        thinking=True,
    )

    assert result.status == "done"
    assert result.content == "Done with workspace tools."
    assert calls[-4:] == ["pass_1", "pass_1_workspace_action_outcome", "pass_2", "pass_3"]


def test_confirmation_workspace_action_rejects_manual_terminal_advice_after_retry(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    mocker.patch.object(agent, "_classify_turn", return_value=TurnClassification(requires_workspace_action=True, followup_kind="confirmation"))

    calls = []
    outcome_calls = {"pass_1_workspace_action_outcome": 0, "pass_2_workspace_action_outcome": 0}

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        calls.append(pass_id)
        if pass_id == "pass_1_workspace_action_outcome":
            outcome_calls[pass_id] += 1
            return type("R", (), {"finish_reason": "stop", "content": '{"outcome":"not_completed"}'})()
        if pass_id == "pass_2_workspace_action_outcome":
            outcome_calls[pass_id] += 1
            outcome = "not_completed" if outcome_calls[pass_id] == 1 else "declined_or_blocked"
            return type("R", (), {"finish_reason": "stop", "content": f'{{"outcome":"{outcome}"}}'})()
        if pass_id == "pass_2_final":
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "I couldn't complete that workspace action because no workspace tool actually ran.",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        return type(
            "R",
            (),
            {
                "finish_reason": "stop",
                "content": "I cannot delete files directly. Please run rm -rf manually in your terminal.",
                "reasoning": "",
                "tool_calls": [],
            },
        )()

    mocker.patch.object(agent, "_call_with_retry", side_effect=fake_call_with_retry)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "delete all files in the workspace"}],
        user_input="yes",
        thinking=True,
    )

    assert result.status == "done"
    assert "couldn't complete that workspace action" in result.content.lower()
    assert "rm -rf" not in result.content
    assert "pass_2_final" in calls


def test_confirmation_workspace_action_rejects_claimed_completion_without_tool_use(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    mocker.patch.object(agent, "_classify_turn", return_value=TurnClassification(requires_workspace_action=True, followup_kind="confirmation"))

    calls = []
    outcome_calls = {"pass_1_workspace_action_outcome": 0, "pass_2_workspace_action_outcome": 0}

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        calls.append(pass_id)
        if pass_id == "pass_1_workspace_action_outcome":
            outcome_calls[pass_id] += 1
            return type("R", (), {"finish_reason": "stop", "content": '{"outcome":"not_completed"}'})()
        if pass_id == "pass_2_workspace_action_outcome":
            outcome_calls[pass_id] += 1
            outcome = "not_completed" if outcome_calls[pass_id] == 1 else "declined_or_blocked"
            return type("R", (), {"finish_reason": "stop", "content": f'{{"outcome":"{outcome}"}}'})()
        if pass_id == "pass_2_final":
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "I couldn't complete that workspace action because no workspace tool actually ran.",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        return type(
            "R",
            (),
            {
                "finish_reason": "stop",
                "content": "I have deleted the following files from your workspace: a.txt and b.txt. The workspace is now empty.",
                "reasoning": "",
                "tool_calls": [],
            },
        )()

    mocker.patch.object(agent, "_call_with_retry", side_effect=fake_call_with_retry)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "delete all files in the workspace"}],
        user_input="yes",
        thinking=True,
    )

    assert result.status == "done"
    assert "couldn't complete that workspace action" in result.content.lower()
    assert "workspace is now empty" not in result.content.lower()
    assert "pass_2_final" in calls


def test_confirmation_workspace_action_requires_mutating_tool_before_accepting_success_claim(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    mocker.patch.object(agent, "_classify_turn", return_value=TurnClassification(requires_workspace_action=True, followup_kind="confirmation"))

    calls = []
    outcome_calls = {"pass_2_workspace_action_outcome": 0, "pass_3_workspace_action_outcome": 0}

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
                    "tool_calls": [type("Call", (), {"stream_id": "1", "index": 0, "id": "call_1", "name": "read_file", "arguments": {"filepath": "foo.txt"}})()],
                },
            )()
        if pass_id == "pass_2_workspace_action_outcome":
            outcome_calls[pass_id] += 1
            return type("R", (), {"finish_reason": "stop", "content": '{"outcome":"not_completed"}'})()
        if pass_id == "pass_2":
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "I renamed foo.txt to bar.txt.",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        if pass_id == "pass_3":
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "I renamed foo.txt to bar.txt.",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        if pass_id == "pass_3_workspace_action_outcome":
            outcome_calls[pass_id] += 1
            outcome = "not_completed" if outcome_calls[pass_id] == 1 else "declined_or_blocked"
            return type("R", (), {"finish_reason": "stop", "content": f'{{"outcome":"{outcome}"}}'})()
        if pass_id == "pass_3_final":
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "I couldn't complete that workspace action because no workspace tool actually ran.",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        raise AssertionError(f"Unexpected pass id: {pass_id}")

    def fake_execute_tool_call(tool_name, args, selected, ctx, confirm_shell=None, **_kwargs):
        assert tool_name == "read_file"
        return {"ok": True, "data": {"filepath": "foo.txt", "content": "alpha"}, "error": None, "meta": {}}

    mocker.patch.object(agent, "_call_with_retry", side_effect=fake_call_with_retry)
    mocker.patch.object(runtime, "execute_tool_call", side_effect=fake_execute_tool_call)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "rename foo.txt to bar.txt"}],
        user_input="yes",
        thinking=True,
    )

    assert result.status == "done"
    assert "couldn't complete that workspace action" in result.content.lower()
    assert "pass_3" in calls


def test_workspace_action_preserves_policy_blocked_reply_when_outcome_classifier_fails(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    mocker.patch.object(
        agent,
        "_classify_turn",
        return_value=TurnClassification(requires_workspace_action=True, prefer_local_workspace_tools=True, followup_kind="confirmation"),
    )

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
                    "tool_calls": [type("Call", (), {"stream_id": "1", "index": 0, "id": "call_1", "name": "shell_command", "arguments": {"command": "rm -rf ."}})()],
                },
            )()
        if pass_id in {"pass_2", "pass_3"}:
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "shell_command is not allowed for local workspace file tasks; use workspace tools instead.",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        if pass_id == "pass_2_workspace_action_outcome":
            raise RuntimeError("timeout")
        raise AssertionError(f"Unexpected pass id: {pass_id}")

    mocker.patch.object(agent, "_call_with_retry", side_effect=fake_call_with_retry)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "delete all files in the workspace"}],
        user_input="yes",
        thinking=True,
    )

    assert result.status == "done"
    assert result.content == "shell_command is not allowed for local workspace file tasks; use workspace tools instead."
    assert "no workspace tool actually ran" not in result.content.lower()
    assert calls == ["pass_1", "pass_2", "pass_2_workspace_action_outcome"]


def test_workspace_action_classifier_failure_does_not_accept_manual_shell_advice_after_block(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    mocker.patch.object(
        agent,
        "_classify_turn",
        return_value=TurnClassification(requires_workspace_action=True, prefer_local_workspace_tools=True, followup_kind="confirmation"),
    )

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
                    "tool_calls": [type("Call", (), {"stream_id": "1", "index": 0, "id": "call_1", "name": "shell_command", "arguments": {"command": "rm -rf ."}})()],
                },
            )()
        if pass_id == "pass_2":
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "shell_command is not allowed here. Please run rm -rf manually in your terminal.",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        if pass_id == "pass_3":
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "shell_command is not allowed here. Please run rm -rf manually in your terminal.",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        if pass_id in {"pass_2_workspace_action_outcome", "pass_3_workspace_action_outcome"}:
            raise RuntimeError("timeout")
        if pass_id == "pass_3_final":
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "I couldn't complete that workspace action because no workspace tool actually ran.",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        raise AssertionError(f"Unexpected pass id: {pass_id}")

    mocker.patch.object(agent, "_call_with_retry", side_effect=fake_call_with_retry)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "delete all files in the workspace"}],
        user_input="yes",
        thinking=True,
    )

    assert result.status == "done"
    assert "couldn't complete that workspace action" in result.content.lower()
    assert "rm -rf" not in result.content.lower()
    assert calls == ["pass_1", "pass_2", "pass_2_workspace_action_outcome", "pass_3", "pass_3_workspace_action_outcome", "pass_3_final", "pass_3_workspace_action_outcome"]






def test_run_turn_allows_same_host_endpoints_with_different_ports(mocker, runtime: SkillRuntime):
    agent = Agent(
        {
            "agent": {
                "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
                "models_endpoint": "http://127.0.0.1:9000/v1/models",
                "allow_cross_host_endpoints": False,
            },
        },
        runtime,
    )
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    mocker.patch.object(
        agent,
        "_call_with_retry",
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
    mocker.patch.object(agent, "_select_skills", return_value=selected)

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

    mocker.patch.object(agent, "_call_with_retry", side_effect=fake_call_with_retry)

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
    mocker.patch.object(agent, "_select_skills", return_value=selected)

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

    mocker.patch.object(agent, "_call_with_retry", side_effect=fake_call_with_retry)
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


def test_batch_workspace_delete_does_not_stop_after_first_successful_tool(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    selected = [runtime.get_skill("workspace-ops")]
    mocker.patch.object(agent, "_select_skills", return_value=selected)

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

    mocker.patch.object(agent, "_call_with_retry", side_effect=fake_call_with_retry)
    mocker.patch.object(
        runtime,
        "execute_tool_call",
        return_value={"ok": True, "data": {"filepath": "/tmp/a.txt", "kind": "file"}, "error": None, "meta": {}},
    )
    mocker.patch.object(runtime, "tools_for_turn", return_value=[{"type": "function", "function": {"name": "delete_path"}}])

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "delete all files in workspace"}],
        user_input="delete all files in workspace",
        thinking=True,
    )

    assert result.status == "done"
    assert result.content == "Finished deleting all requested files."
    assert calls == ["pass_1", "pass_2"]


def test_workspace_scaffold_does_not_stop_after_create_directory(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    selected = [runtime.get_skill("workspace-ops")]
    mocker.patch.object(agent, "_select_skills", return_value=selected)

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
                        type("Call", (), {"stream_id": "1", "index": 0, "id": "call_1", "name": "create_directory", "arguments": {"path": "arjun"}})(),
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
                        type("Call", (), {"stream_id": "2", "index": 0, "id": "call_2", "name": "create_files", "arguments": {"files": [{"filepath": "arjun/index.html", "content": "<html></html>"}, {"filepath": "arjun/styles.css", "content": "body{}"}, {"filepath": "arjun/script.js", "content": "console.log(1)\n"}]}})(),
                    ],
                },
            )()
        if pass_id == "pass_3":
            return type("R", (), {"finish_reason": "stop", "content": "Done.", "reasoning": "", "tool_calls": []})()
        raise AssertionError(f"Unexpected pass id: {pass_id}")

    mocker.patch.object(agent, "_call_with_retry", side_effect=fake_call_with_retry)
    executed = []
    real_execute = runtime.execute_tool_call

    def wrapped_execute(tool_name, args, selected, ctx, confirm_shell=None, **_kwargs):
        executed.append(tool_name)
        return real_execute(tool_name, args, selected=selected, ctx=ctx, confirm_shell=confirm_shell)

    mocker.patch.object(runtime, "execute_tool_call", side_effect=wrapped_execute)
    mocker.patch.object(
        runtime,
        "tools_for_turn",
        return_value=[
            {"type": "function", "function": {"name": "create_directory"}},
            {"type": "function", "function": {"name": "create_files"}},
        ],
    )

    result = agent.run_turn(
        history_messages=[],
        user_input="Make a landing page for a bakery using html css and javascript and save it in a folder called arjun",
        thinking=True,
    )

    assert result.status == "done"
    assert executed == ["create_directory", "create_files"]
    assert calls == ["pass_1", "pass_2", "pass_3"]


def test_workspace_folder_and_single_file_does_not_stop_after_create_directory(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    selected = [runtime.get_skill("workspace-ops")]
    mocker.patch.object(agent, "_select_skills", return_value=selected)

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
                        type("Call", (), {"stream_id": "1", "index": 0, "id": "call_1", "name": "create_directory", "arguments": {"path": "hvb"}})(),
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
                        type("Call", (), {"stream_id": "2", "index": 0, "id": "call_2", "name": "create_file", "arguments": {"filepath": "hvb/insertion_sort.py", "content": "def insertion_sort(arr):\n    return arr\n"}})(),
                    ],
                },
            )()
        if pass_id == "pass_3":
            return type("R", (), {"finish_reason": "stop", "content": "Done.", "reasoning": "", "tool_calls": []})()
        raise AssertionError(f"Unexpected pass id: {pass_id}")

    mocker.patch.object(agent, "_call_with_retry", side_effect=fake_call_with_retry)
    executed = []
    real_execute = runtime.execute_tool_call

    def wrapped_execute(tool_name, args, selected, ctx, confirm_shell=None, **_kwargs):
        executed.append(tool_name)
        return real_execute(tool_name, args, selected=selected, ctx=ctx, confirm_shell=confirm_shell)

    mocker.patch.object(runtime, "execute_tool_call", side_effect=wrapped_execute)
    mocker.patch.object(
        runtime,
        "tools_for_turn",
        return_value=[
            {"type": "function", "function": {"name": "create_directory"}},
            {"type": "function", "function": {"name": "create_file"}},
        ],
    )

    result = agent.run_turn(
        history_messages=[],
        user_input="make a folder called hvb and make a python file that does insertion sort",
        thinking=True,
    )

    assert result.status == "done"
    assert executed == ["create_directory", "create_file"]
    assert calls == ["pass_1", "pass_2", "pass_3"]


def test_workspace_readback_request_does_not_stop_after_create_file(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    selected = [runtime.get_skill("workspace-ops")]
    mocker.patch.object(agent, "_select_skills", return_value=selected)

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
                        type("Call", (), {"stream_id": "1", "index": 0, "id": "call_1", "name": "create_file", "arguments": {"filepath": "notes.txt", "content": "alpha"}})(),
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
                        type("Call", (), {"stream_id": "2", "index": 0, "id": "call_2", "name": "read_file", "arguments": {"filepath": "notes.txt"}})(),
                    ],
                },
            )()
        if pass_id == "pass_3":
            return type("R", (), {"finish_reason": "stop", "content": "notes.txt=alpha", "reasoning": "", "tool_calls": []})()
        raise AssertionError(f"Unexpected pass id: {pass_id}")

    mocker.patch.object(agent, "_call_with_retry", side_effect=fake_call_with_retry)
    executed = []

    def fake_execute(tool_name, args, selected, ctx, **_kwargs):
        executed.append(tool_name)
        if tool_name == "create_file":
            return {"ok": True, "data": {"filepath": "notes.txt", "basename": "notes.txt"}, "error": None, "meta": {}}
        if tool_name == "read_file":
            return {"ok": True, "data": {"filepath": "notes.txt", "content": "alpha"}, "error": None, "meta": {}}
        raise AssertionError(f"Unexpected tool: {tool_name}")

    mocker.patch.object(runtime, "execute_tool_call", side_effect=fake_execute)
    mocker.patch.object(
        runtime,
        "tools_for_turn",
        return_value=[
            {"type": "function", "function": {"name": "create_file"}},
            {"type": "function", "function": {"name": "read_file"}},
        ],
    )

    result = agent.run_turn(
        history_messages=[],
        user_input='Create notes.txt containing exactly the word alpha, then read it back and reply with exactly: "notes.txt=alpha".',
        thinking=True,
    )

    assert result.status == "done"
    assert result.content == "notes.txt=alpha"
    assert executed == ["create_file", "read_file"]
    assert calls == ["pass_1", "pass_2", "pass_3"]








def test_local_workspace_tasks_prefer_workspace_tools_but_still_block_fetch_tools(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    mocker.patch.object(agent, "_classify_turn", return_value=TurnClassification(prefer_local_workspace_tools=True))
    selected = [runtime.get_skill("workspace-ops")]
    mocker.patch.object(agent, "_select_skills", return_value=selected)

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
                        type("Call", (), {"stream_id": "1", "index": 0, "id": "call_1", "name": "shell_command", "arguments": {"command": "mkdir -p 1738"}})(),
                        type("Call", (), {"stream_id": "1", "index": 1, "id": "call_2", "name": "fetch_url", "arguments": {"url": "/tmp/index.html"}})(),
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
                        type("Call", (), {"stream_id": "2", "index": 0, "id": "call_3", "name": "create_directory", "arguments": {"path": "1738"}})(),
                        type("Call", (), {"stream_id": "2", "index": 1, "id": "call_4", "name": "create_files", "arguments": {"files": [{"filepath": "1738/index.html", "content": "<html></html>"}, {"filepath": "1738/script.js", "content": "console.log(1)\n"}]}})(),
                    ],
                },
            )()
        if pass_id == "pass_3":
            return type("R", (), {"finish_reason": "stop", "content": "Done.", "reasoning": "", "tool_calls": []})()
        raise AssertionError(f"Unexpected pass id: {pass_id}")

    mocker.patch.object(agent, "_call_with_retry", side_effect=fake_call_with_retry)
    executed = []
    real_execute = runtime.execute_tool_call

    def wrapped_execute(tool_name, args, selected, ctx, confirm_shell=None, **_kwargs):
        executed.append(tool_name)
        return real_execute(tool_name, args, selected=selected, ctx=ctx, confirm_shell=confirm_shell)

    mocker.patch.object(runtime, "execute_tool_call", side_effect=wrapped_execute)
    mocker.patch.object(
        runtime,
        "tools_for_turn",
        return_value=[
            {"type": "function", "function": {"name": "create_directory"}},
            {"type": "function", "function": {"name": "create_files"}},
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
    assert executed == ["shell_command", "create_directory", "create_files"]
    assert calls == ["pass_1", "pass_2", "pass_3"]


def test_workspace_action_accepts_successful_mutating_shell_command(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    mocker.patch.object(
        agent,
        "_classify_turn",
        return_value=TurnClassification(requires_workspace_action=True, prefer_local_workspace_tools=True),
    )
    selected = [runtime.get_skill("workspace-ops")]
    mocker.patch.object(agent, "_select_skills", return_value=selected)

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
                        type("Call", (), {"stream_id": "1", "index": 0, "id": "call_1", "name": "shell_command", "arguments": {"command": "mkdir -p 1738"}})(),
                    ],
                },
            )()
        if pass_id == "pass_2":
            return type("R", (), {"finish_reason": "stop", "content": "Done.", "reasoning": "", "tool_calls": []})()
        raise AssertionError(f"Unexpected pass id: {pass_id}")

    mocker.patch.object(agent, "_call_with_retry", side_effect=fake_call_with_retry)
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
        side_effect=lambda tool_name, args, selected, ctx, confirm_shell=None, **_kwargs: runtime.workspace.run_shell_command(
            args["command"]
        ),
    )

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "create a folder named 1738"}],
        user_input="yes",
        thinking=True,
        confirm_shell=lambda _command: True,
    )

    assert result.status == "done"
    assert result.content == "Done."
    assert (runtime.workspace.workspace_root / "1738").is_dir()
    assert calls == ["pass_1", "pass_2"]


def test_workspace_action_does_not_fingerprint_around_shell_command(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    mocker.patch.object(
        agent,
        "_classify_turn",
        return_value=TurnClassification(requires_workspace_action=True, prefer_local_workspace_tools=True),
    )
    selected = [runtime.get_skill("workspace-ops")]
    mocker.patch.object(agent, "_select_skills", return_value=selected)
    mocker.patch.object(
        runtime,
        "tools_for_turn",
        return_value=[
            {"type": "function", "function": {"name": "create_directory"}},
            {"type": "function", "function": {"name": "shell_command"}},
        ],
    )
    mocker.patch.object(runtime.workspace, "workspace_state_fingerprint", side_effect=AssertionError("unexpected fingerprint call"))

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
                        type("Call", (), {"stream_id": "1", "index": 0, "id": "call_1", "name": "shell_command", "arguments": {"command": "mkdir -p 1738"}})(),
                    ],
                },
            )()
        if pass_id == "pass_2":
            return type("R", (), {"finish_reason": "stop", "content": "Done.", "reasoning": "", "tool_calls": []})()
        raise AssertionError(f"Unexpected pass id: {pass_id}")

    mocker.patch.object(agent, "_call_with_retry", side_effect=fake_call_with_retry)
    mocker.patch.object(
        runtime,
        "execute_tool_call",
        side_effect=lambda tool_name, args, selected, ctx, confirm_shell=None, **_kwargs: runtime.workspace.run_shell_command(
            args["command"]
        ),
    )

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "create a folder named 1738"}],
        user_input="yes",
        thinking=True,
        confirm_shell=lambda _command: True,
    )

    assert result.status == "done"
    assert (runtime.workspace.workspace_root / "1738").is_dir()


def test_explicit_external_path_disables_local_workspace_routing(runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    selected = [runtime.get_skill("workspace-ops")]
    other_path = str(Path(runtime.workspace.home_root) / "other-project")
    ctx = SkillContext(
        user_input=f"Update the packages in {other_path}, it uses uv",
        branch_labels=[],
        attachments=[],
        workspace_root=str(runtime.workspace.workspace_root),
        memory_hits=[],
    )

    assert agent._explicit_path_outside_workspace(ctx.user_input) == other_path
    assert agent._prefers_local_workspace_tools(ctx, selected) is False


def test_explicit_external_path_adds_prompt_rule_and_skips_local_workspace_rule(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    selected = [runtime.get_skill("workspace-ops")]
    mocker.patch.object(agent, "_select_skills", return_value=selected)
    mocker.patch.object(runtime, "tools_for_turn", return_value=[])
    other_path = str(Path(runtime.workspace.home_root) / "other-project")

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        system_text = payload["messages"][0]["content"]
        assert "Explicit path rule:" in system_text
        assert other_path in system_text
        assert "Local workspace tool rule:" not in system_text
        return type("R", (), {"finish_reason": "stop", "content": "Need confirmation.", "reasoning": "", "tool_calls": []})()

    mocker.patch.object(agent, "_call_with_retry", side_effect=fake_call_with_retry)

    result = agent.run_turn(
        history_messages=[],
        user_input=f"Update the packages in {other_path}, it uses uv",
        thinking=True,
    )

    assert result.status == "done"
    assert result.content == "Need confirmation."


def test_explicit_external_path_ignores_urls(runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)

    assert agent._explicit_path_outside_workspace("Use https://example.com as inspiration for the landing page") == ""


def test_explicit_external_path_supports_quoted_paths_with_spaces(runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    selected = [runtime.get_skill("workspace-ops")]
    other_path = str(Path(runtime.workspace.home_root) / "Other Project")
    ctx = SkillContext(
        user_input=f'Update the packages in "{other_path}", it uses uv',
        branch_labels=[],
        attachments=[],
        workspace_root=str(runtime.workspace.workspace_root),
        memory_hits=[],
    )

    assert agent._explicit_path_outside_workspace(ctx.user_input) == other_path
    assert agent._prefers_local_workspace_tools(ctx, selected) is False


def test_finalization_retries_when_model_leaks_tool_markup(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
        }
    }
    agent = Agent(cfg, runtime)

    chat_reqs = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"reasoning_content":"Need a plain answer."}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        if len(chat_reqs) == 2:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"content":"<tool_call>\\n<function=web_search>\\n<parameter=query>meta</parameter>\\n</function>\\n</tool_call>"}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        if len(chat_reqs) == 3:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"content":"I could not verify a clean answer from the available evidence."}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        raise AssertionError("Unexpected extra completion call")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "tell me about meta"}],
        user_input="tell me about meta",
        thinking=True,
    )

    assert result.status == "done"
    assert "<tool_call>" not in result.content
    assert result.content == "I could not verify a clean answer from the available evidence."
    assert len(chat_reqs) == 3


def test_finalization_returns_error_when_markup_repeats(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
        }
    }
    agent = Agent(cfg, runtime)

    history = [
        {
            "role": "tool",
            "name": "web_search",
            "content": json.dumps(
                {
                    "ok": True,
                    "data": {
                        "results": [
                            {
                                "title": "Meta Newsroom",
                                "url": "https://about.fb.com/news/",
                                "domain": "about.fb.com",
                                "snippet": "Meta newsroom updates.",
                            }
                        ]
                    },
                    "error": None,
                    "meta": {},
                }
            ),
        }
    ]

    chat_reqs = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"reasoning_content":"Need a plain answer."}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"<tool_call>\\n<function=web_search>\\n<parameter=query>meta</parameter>\\n</function>\\n</tool_call>"}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=history,
        user_input="tell me about meta",
        thinking=True,
    )

    assert result.status == "error"
    assert "Finalization failed to produce a clean user-facing answer" in (result.error or "")
    assert len(chat_reqs) == 3


def test_finalization_strips_think_tags_from_model_output(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
        }
    }
    agent = Agent(cfg, runtime)

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"<think>internal</think>\\n\\nVisible answer"}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "say visible answer"}],
        user_input="say visible answer",
        thinking=True,
    )

    assert result.status == "done"
    assert "<think>" not in result.content
    assert "</think>" not in result.content
    assert "Visible answer" in result.content


def test_main_pass_tool_markup_forces_clean_finalization(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
        }
    }
    agent = Agent(cfg, runtime)

    chat_reqs = []
    events = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"content":"I should check that.\\n\\n<tool_call>\\n<function=shell>\\n<parameter=command>git branch</parameter>\\n</function>\\n</tool_call>"}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"I misformatted the previous reply. Please tell me what you want to do with the branch."}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "This is a branch!"}],
        user_input="This is a branch!",
        thinking=True,
        on_event=events.append,
    )

    assert result.status == "done"
    assert "<tool_call>" not in result.content
    assert "misformatted the previous reply" in result.content
    assert len(chat_reqs) == 2
    assert not any(
        evt.get("type") == "content_token" and "<tool_call>" in str(evt.get("text", ""))
        for evt in events
    )


def test_search_failures_return_safe_non_speculative_answer(mocker, runtime: SkillRuntime):
    search_skill = runtime.skills_dir / "search-ops"
    search_skill.mkdir(parents=True)
    (search_skill / "SKILL.md").write_text(
        """
---
name: search-ops
description: web research
version: 1.1.0
allowed-tools: web_search
metadata:
  tags: [web, latest]
  triggers:
    keywords: [latest, web]
---
search
""".strip(),
        encoding="utf-8",
    )
    (search_skill / "tools.py").write_text(
        """
TOOL_SPECS = {
  "web_search": {
    "capability": "web_search",
    "description": "Search web",
    "parameters": {
      "type": "object",
      "properties": {"query": {"type": "string"}},
      "required": ["query"]
    }
  }
}

def execute(tool_name, args, env):
    return {"ok": False, "data": None, "error": {"code": "E_IO", "message": "Search returned no parsable results"}, "meta": {}}
""".strip(),
        encoding="utf-8",
    )
    runtime.load_skills()

    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
        }
    }
    agent = Agent(cfg, runtime)

    chat_reqs = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        chat_reqs.append(req)
        if len(chat_reqs) <= 2:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"web_search","arguments":"{\\"query\\": \\\"meta latest acquisitions\\\"}"}}]}}]}',
                    'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"I could not verify the latest acquisitions from reliable web results in this turn."}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "tell me the latest acquisitions"}],
        user_input="tell me the latest acquisitions",
        thinking=True,
    )

    assert result.status == "done"
    assert "could not verify" in result.content.lower()


def test_search_tool_call_cap_forces_finalization_after_two_searches(mocker, runtime: SkillRuntime):
    search_skill = runtime.skills_dir / "search-ops"
    search_skill.mkdir(parents=True)
    (search_skill / "SKILL.md").write_text(
        """
---
name: search-ops
description: web research
version: 1.2.0
allowed-tools: web_search
metadata:
  tags: [web, latest]
  triggers:
    keywords: [latest, web]
---
search
""".strip(),
        encoding="utf-8",
    )
    (search_skill / "tools.py").write_text(
        """
TOOL_SPECS = {
  "web_search": {
    "capability": "web_search",
    "description": "Search web",
    "parameters": {
      "type": "object",
      "properties": {"query": {"type": "string"}},
      "required": ["query"]
    }
  }
}

def execute(tool_name, args, env):
    return {
      "ok": True,
      "data": {
        "results": [
          {"title": "Example Source", "url": "https://example.com", "domain": "example.com", "snippet": "Example"}
        ]
      },
      "error": None,
      "meta": {}
    }
""".strip(),
        encoding="utf-8",
    )
    runtime.load_skills()

    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
        }
    }
    agent = Agent(cfg, runtime)

    chat_reqs = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        chat_reqs.append(req)
        if len(chat_reqs) in {1, 2}:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"web_search","arguments":"{\\"query\\": \\\"meta latest acquisitions\\\"}"}}]}}]}',
                    'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"Use the verified sources already gathered."}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "tell me the latest acquisitions"}],
        user_input="tell me the latest acquisitions",
        thinking=True,
    )

    assert result.status == "done"
    assert result.content == "Use the verified sources already gathered."
    assert len(chat_reqs) == 3


def test_time_sensitive_query_forces_search_before_accepting_answer(mocker, runtime: SkillRuntime):
    search_skill = runtime.skills_dir / "search-ops"
    search_skill.mkdir(parents=True)
    (search_skill / "SKILL.md").write_text(
        """
---
name: search-ops
description: Search the web for recent information.
allowed-tools: web_search
metadata:
  tags: [web, latest, recent, current, news]
---
Search the internet.
""".strip(),
        encoding="utf-8",
    )
    (search_skill / "tools.py").write_text(
        """
TOOL_SPECS = {
  "web_search": {
    "capability": "web_search",
    "description": "Search web",
    "parameters": {
      "type": "object",
      "properties": {"query": {"type": "string"}},
      "required": ["query"]
    }
  }
}

def execute(tool_name, args, env):
    return {"ok": True, "data": {"results": [{"title": "Example", "url": "https://example.com", "domain": "example.com", "snippet": "Verified"}]}, "error": None, "meta": {}}
""".strip(),
        encoding="utf-8",
    )
    runtime.load_skills()

    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
        }
    }
    agent = Agent(cfg, runtime)
    mocker.patch.object(
        agent,
        "_classify_turn",
        return_value=TurnClassification(time_sensitive=True),
    )

    chat_reqs = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"content":"As of my knowledge, Meta has not announced major new acquisitions."}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        if len(chat_reqs) == 2:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"web_search","arguments":"{\\"query\\": \\\"meta latest acquisitions\\\"}"}}]}}]}',
                    'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"I could not verify the latest open source models from reliable current web results in this turn."}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "tell me about the latest open source models"}],
        user_input="tell me about the latest open source models",
        thinking=True,
    )

    assert result.status == "done"
    assert "could not verify" in result.content.lower()
    assert len(chat_reqs) == 4


def test_time_sensitive_query_without_model_classification_does_not_force_search(mocker, runtime: SkillRuntime):
    search_skill = runtime.skills_dir / "search-ops"
    search_skill.mkdir(parents=True)
    (search_skill / "SKILL.md").write_text(
        """
---
name: search-ops
description: Search the web for recent information.
allowed-tools: web_search
metadata:
  tags: [web, latest, recent, current, news]
---
Search the internet.
""".strip(),
        encoding="utf-8",
    )
    (search_skill / "tools.py").write_text(
        """
TOOL_SPECS = {
  "web_search": {
    "capability": "web_search",
    "description": "Search web",
    "parameters": {
      "type": "object",
      "properties": {"query": {"type": "string"}},
      "required": ["query"]
    }
  }
}

def execute(tool_name, args, env):
    return {"ok": True, "data": {"results": [{"title": "Example", "url": "https://example.com", "domain": "example.com", "snippet": "Verified"}]}, "error": None, "meta": {}}
""".strip(),
        encoding="utf-8",
    )
    runtime.load_skills()

    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
        }
    }
    agent = Agent(cfg, runtime)

    chat_reqs = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"content":"As of my knowledge, Meta has not announced major new acquisitions."}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        if len(chat_reqs) == 2:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"web_search","arguments":"{\\"query\\": \\\"meta latest acquisitions\\\"}"}}]}}]}',
                    'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"I could not verify the latest open source models from reliable current web results in this turn."}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "tell me about the latest open source models"}],
        user_input="tell me about the latest open source models",
        thinking=True,
    )

    assert result.status == "done"
    assert "as of my knowledge" in result.content.lower()
    assert len(chat_reqs) == 1


def test_all_active_skills_expose_search_tool_on_first_turn(mocker, runtime: SkillRuntime):
    search_skill = runtime.skills_dir / "search-ops"
    search_skill.mkdir(parents=True)
    (search_skill / "SKILL.md").write_text(
        """
---
name: search-ops
description: Search the web and fetch page content for research and up-to-date information.
allowed-tools: web_search
metadata:
  tags: [web, latest, recent, current, news, lookup]
---
Search the internet.
""".strip(),
        encoding="utf-8",
    )
    (search_skill / "tools.py").write_text(
        """
TOOL_SPECS = {
  "web_search": {
    "capability": "web_search",
    "description": "Search web",
    "parameters": {
      "type": "object",
      "properties": {"query": {"type": "string"}},
      "required": ["query"]
    }
  }
}

def execute(tool_name, args, env):
    return {"ok": True, "data": {"results": [{"title": "Meta News", "url": "https://example.com", "domain": "example.com", "snippet": "Verified"}]}, "error": None, "meta": {}}
""".strip(),
        encoding="utf-8",
    )
    runtime.load_skills()

    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
        },
    }
    agent = Agent(cfg, runtime)

    chat_reqs = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            payload = json.loads(req.data.decode("utf-8"))
            tool_names = [tool["function"]["name"] for tool in payload.get("tools", [])]
            assert "web_search" in tool_names
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"web_search","arguments":"{\\"query\\": \\\"meta latest acquisitions\\\"}"}}]}}]}',
                    'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"I checked the selected search skill before answering."}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "tell me about the latest acquisitions"}],
        user_input="tell me about the latest acquisitions",
        thinking=True,
    )

    assert result.status == "done"


def test_skill_index_keeps_full_catalog_available_until_a_skill_is_loaded(runtime: SkillRuntime):
    hidden_skill = runtime.skills_dir / "hidden-tool"
    hidden_skill.mkdir(parents=True)
    (hidden_skill / "SKILL.md").write_text(
        """
---
name: hidden-tool
description: General build helper.
allowed-tools: artifact_forge
---
Use artifact_forge when needed.
""".strip(),
        encoding="utf-8",
    )
    (hidden_skill / "tools.py").write_text(
        """
TOOL_SPECS = {
  "artifact_forge": {
    "capability": "workspace_write",
    "description": "Forge an artifact",
    "parameters": {
      "type": "object",
      "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}},
      "required": ["filepath", "content"]
    }
  }
}

def execute(tool_name, args, env):
    path = env.workspace.create_file(args["filepath"], args["content"])
    return {"ok": True, "data": {"filepath": path}, "error": None, "meta": {}}
""".strip(),
        encoding="utf-8",
    )
    runtime.load_skills()
    runtime.config = {}

    agent = Agent({"agent": {}}, runtime)
    ctx = SkillContext(
        user_input="make an artifact for me",
        branch_labels=[],
        attachments=[],
        workspace_root=str(runtime.workspace.workspace_root),
        memory_hits=[],
    )
    selected = agent.skill_runtime.select_skills(ctx)
    system_content = agent.prompt_renderer.compose_system_content(selected, ctx)
    assert "hidden-tool" in system_content
    assert "artifact_forge" not in system_content

    load_result = runtime.skill_view("hidden-tool", "", ctx)
    assert load_result["skill_id"] == "hidden-tool"
    assert load_result["loaded"] is True
    selected = agent.skill_runtime.select_skills(ctx)
    loaded_system_content = agent.prompt_renderer.compose_system_content(selected, ctx)
    assert "hidden-tool" in loaded_system_content
    assert "artifact_forge" in loaded_system_content


def test_large_tool_call_args_are_compacted_in_history(runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    large = "x" * 5000
    compacted = agent._tool_call_args_for_history({"filepath": "a.txt", "content": large})

    assert compacted["filepath"] == "a.txt"
    assert compacted["content"].startswith("x" * 1200)
    assert "[truncated 3800 chars]" in compacted["content"]
    assert len(compacted["content"]) < len(large)


def test_agent_transport_error_marks_error(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
        }
    }
    agent = Agent(cfg, runtime)

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        raise urllib.error.URLError("boom")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "hi"}],
        user_input="hi",
        thinking=True,
    )

    assert result.status == "error"


def test_agent_infers_tool_calls_when_finish_reason_missing(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
        }
    }
    agent = Agent(cfg, runtime)

    chat_reqs = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            # Backend bug: emits tool_calls but final reason is "stop".
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"create_file","arguments":"{\\"filepath\\": \\\"a.txt\\\", \\\"content\\\": \\\"hello\\\"}"}}]}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        if len(chat_reqs) == 2:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"content":"Done"}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        raise AssertionError("Unexpected extra completion call")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "write a file"}],
        user_input="write a file",
        thinking=True,
    )

    assert result.status == "done"
    assert any(msg.get("role") == "tool" for msg in result.skill_exchanges)


def test_agent_emits_unique_tool_stream_ids_per_pass(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
        }
    }
    agent = Agent(cfg, runtime)

    chat_reqs = []
    events = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"create_file","arguments":"{\\"filepath\\": \\\"a.txt\\\", \\\"content\\\": \\\"hello\\\"}"}}]}}]}',
                    'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )
        if len(chat_reqs) == 2:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_2","type":"function","function":{"name":"create_file","arguments":"{\\"filepath\\": \\\"b.txt\\\", \\\"content\\\": \\\"world\\\"}"}}]}}]}',
                    'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )
        if len(chat_reqs) == 3:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"content":"Done"}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        raise AssertionError("Unexpected extra completion call")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "write files"}],
        user_input="write files",
        thinking=True,
        on_event=events.append,
    )

    assert result.status == "done"
    stream_ids = [evt.get("stream_id") for evt in events if evt.get("type") == "tool_call"]
    assert len(stream_ids) == 2
    assert len(set(stream_ids)) == 2


def test_tool_result_history_not_compacted_by_default(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
        }
    }
    agent = Agent(cfg, runtime)
    huge_text = "x" * 24000

    chat_reqs = []

    def fake_execute_tool_call(tool_name, args, selected, ctx, confirm_shell=None, **_kwargs):
        return {"ok": True, "data": {"blob": huge_text}, "error": None, "meta": {}}

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"create_file","arguments":"{\\"filepath\\": \\\"a.txt\\\", \\\"content\\\": \\\"hello\\\"}"}}]}}]}',
                    'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )
        if len(chat_reqs) == 2:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"content":"Done"}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        raise AssertionError("Unexpected extra completion call")

    mocker.patch.object(runtime, "execute_tool_call", side_effect=fake_execute_tool_call)
    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "write"}],
        user_input="write",
        thinking=True,
    )

    assert result.status == "done"
    second_payload = json.loads(chat_reqs[1].data.decode("utf-8"))
    tool_msgs = [msg for msg in second_payload["messages"] if msg.get("role") == "tool"]
    assert tool_msgs
    assert huge_text in tool_msgs[-1]["content"]


def test_tool_result_history_compaction_can_be_gated_by_tool_name(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
            "compact_tool_results_in_history": True,
            "compact_tool_result_tools": ["create_file"],
            "max_tool_result_chars": 200,
        }
    }
    agent = Agent(cfg, runtime)
    huge_text = "y" * 6000

    chat_reqs = []

    def fake_execute_tool_call(tool_name, args, selected, ctx, confirm_shell=None, **_kwargs):
        return {"ok": True, "data": {"blob": huge_text}, "error": None, "meta": {}}

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"create_file","arguments":"{\\"filepath\\": \\\"a.txt\\\", \\\"content\\\": \\\"hello\\\"}"}}]}}]}',
                    'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )
        if len(chat_reqs) == 2:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"content":"Done"}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        raise AssertionError("Unexpected extra completion call")

    mocker.patch.object(runtime, "execute_tool_call", side_effect=fake_execute_tool_call)
    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "write"}],
        user_input="write",
        thinking=True,
    )

    assert result.status == "done"
    second_payload = json.loads(chat_reqs[1].data.decode("utf-8"))
    tool_msgs = [msg for msg in second_payload["messages"] if msg.get("role") == "tool"]
    assert tool_msgs
    tool_content = tool_msgs[-1]["content"]
    assert "truncated" in tool_content
    assert huge_text not in tool_content


def test_agent_can_cancel_while_waiting_for_readiness(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 5,
            "readiness_poll_s": 0.5,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
        }
    }
    agent = Agent(cfg, runtime)
    stop_event = threading.Event()

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            stop_event.set()
            raise urllib.request.URLError("refused")
        raise AssertionError("chat endpoint should not be reached")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "hi"}],
        user_input="hi",
        thinking=True,
        stop_event=stop_event,
    )

    assert result.status == "cancelled"


def test_doctor_report_uses_env_auth_header(mocker, runtime: SkillRuntime, monkeypatch):
    monkeypatch.setenv("ALPHANUS_AUTH_HEADER", "Authorization: Bearer test")
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
        },
        "search": {"provider": "tavily"},
    }
    agent = Agent(cfg, runtime)

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        raise AssertionError("unexpected endpoint")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)
    report = agent.doctor_report()
    assert report["agent"]["auth_header_source"] == "env"


def test_doctor_report_supports_brave_provider(mocker, runtime: SkillRuntime, monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-test-key")
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
        },
        "search": {"provider": "brave"},
    }
    agent = Agent(cfg, runtime)

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        raise AssertionError("unexpected endpoint")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)
    report = agent.doctor_report()
    assert report["search"]["provider"] == "brave"
    assert report["search"]["ready"] is True
    assert report["search"]["reason"] == ""


def test_fetch_model_name_reads_first_model_id(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
        }
    }
    agent = Agent(cfg, runtime)

    def fake_urlopen(req, timeout=None, context=None):
        assert req.full_url.endswith("/v1/models")
        return FakeResponse(['{"data":[{"id":"llama-3.2-3b-instruct"}]}'])

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    assert agent.fetch_model_name() == "llama-3.2-3b-instruct"


def test_fetch_model_metadata_reads_model_id_and_context_window(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
        }
    }
    agent = Agent(cfg, runtime)

    def fake_urlopen(req, timeout=None, context=None):
        assert req.full_url.endswith("/v1/models")
        return FakeResponse(['{"data":[{"id":"llama-3.2-3b-instruct","metadata":{"n_ctx_slot":40960}}]}'])

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    assert agent.fetch_model_metadata() == ("llama-3.2-3b-instruct", 40960)


def test_select_skills_returns_only_explicitly_loaded_skills(runtime: SkillRuntime):
    runtime.config = {}
    agent = Agent({"agent": {}}, runtime)
    ctx = SkillContext(
        user_input="write a file",
        branch_labels=[],
        attachments=[],
        workspace_root=str(runtime.workspace.workspace_root),
        memory_hits=[],
    )
    assert agent._select_skills(ctx, threading.Event()) == []


def test_fetch_model_metadata_falls_back_to_props_for_context_window(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
        }
    }
    agent = Agent(cfg, runtime)

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse(['{"data":[{"id":"llama-3.2-3b-instruct"}]}'])
        if req.full_url.endswith("/props"):
            return FakeResponse(['{"default_generation_settings":{"n_ctx":40960}}'])
        raise AssertionError(f"unexpected endpoint: {req.full_url}")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    assert agent.fetch_model_metadata() == ("llama-3.2-3b-instruct", 40960)


def test_fetch_model_name_accepts_top_level_model_field(runtime: SkillRuntime):
    assert Agent._extract_model_name({"model": "qwen2.5-coder-7b"}) == "qwen2.5-coder-7b"


def test_extract_model_context_window_accepts_nested_metadata(runtime: SkillRuntime):
    assert Agent._extract_model_context_window({"data": [{"metadata": {"n_ctx_slot": 40960}}]}) == 40960


def test_extract_model_context_window_accepts_recursive_nested_fields(runtime: SkillRuntime):
    assert Agent._extract_model_context_window({"default_generation_settings": {"nested": {"n_ctx": 40960}}}) == 40960


def test_reload_config_resets_readiness_state(runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
        },
        "context": {"context_limit": 8192, "keep_last_n": 10, "safety_margin": 500},
    }
    agent = Agent(cfg, runtime)
    agent._ready_checked = True

    agent.reload_config(
        {
            "agent": {
                **cfg["agent"],
                "model_endpoint": "http://127.0.0.1:8081/v1/chat/completions",
                "models_endpoint": "http://127.0.0.1:8081/v1/models",
            },
            "context": cfg["context"],
        }
    )

    assert agent.model_endpoint == "http://127.0.0.1:8081/v1/chat/completions"
    assert agent.models_endpoint == "http://127.0.0.1:8081/v1/models"
    assert agent._ready_checked is False


def test_reload_config_rebuilds_context_manager(runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
        },
        "context": {"context_limit": 8192, "keep_last_n": 10, "safety_margin": 500},
    }
    agent = Agent(cfg, runtime)

    agent.reload_config(
        {
            "agent": cfg["agent"],
            "context": {"context_limit": 4096, "keep_last_n": 4, "safety_margin": 200},
        }
    )

    assert agent.context_mgr.context_limit == 4096
    assert agent.context_mgr.keep_last_n == 4
    assert agent.context_mgr.safety_margin == 200


def test_time_sensitive_search_without_fetch_evidence_declines(mocker, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "search-ops").mkdir(parents=True)
    (skills / "search-ops" / "SKILL.md").write_text(
        """
---
name: search-ops
description: web search
allowed-tools: web_search fetch_url
metadata:
  tags: [web, latest, current]
---
Search
""".strip(),
        encoding="utf-8",
    )
    (skills / "search-ops" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "web_search": {
    "capability": "web_search",
    "description": "Search web",
    "parameters": {
      "type": "object",
      "properties": {"query": {"type": "string"}},
      "required": ["query"]
    }
  },
  "fetch_url": {
    "capability": "web_fetch",
    "description": "Fetch url",
    "parameters": {
      "type": "object",
      "properties": {"url": {"type": "string"}},
      "required": ["url"]
    }
  }
}

def execute(tool_name, args, env):
    if tool_name == "web_search":
        return {"ok": True, "data": {"results": [{"title": "Iran update", "url": "https://example.com", "domain": "example.com", "snippet": "snippet"}]}, "error": None, "meta": {}}
    if tool_name == "fetch_url":
        return {"ok": False, "data": None, "error": {"code": "E_IO", "message": "blocked"}, "meta": {}}
    raise ValueError(tool_name)
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
        },
    }
    agent = Agent(cfg, runtime)
    chat_reqs = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"web_search","arguments":"{\\"query\\": \\\"iran current situation\\\"}"}}]}}]}',
                    'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"I could not verify the current situation in Iran from reliable fetched sources in this turn."}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)
    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "what is the current situation in iran"}],
        user_input="what is the current situation in iran",
        thinking=True,
    )
    assert result.status == "done"
    assert "could not verify" in result.content.lower()


def test_plain_text_skill_mention_does_not_auto_load_skill_without_skill_view(mocker, tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "docx").mkdir(parents=True)
    (skills / "docx" / "scripts").mkdir(parents=True)
    (skills / "docx" / "scripts" / "convert.py").write_text("print('ok')\n", encoding="utf-8")
    (skills / "docx" / "SKILL.md").write_text(
        """
---
name: docx
description: convert documents to docx
version: 1.0.0
triggers:
  keywords:
    - docx
---
Detailed internal prompt for DOCX workflows.
Use scripts/convert.py when needed.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
        },
    }
    agent = Agent(cfg, runtime)
    chat_reqs: list[urllib.request.Request] = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        chat_reqs.append(req)
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"Loaded and ready."}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "use skill docx"}],
        user_input="use skill docx",
        thinking=True,
    )

    assert result.status == "done"
    assert len(chat_reqs) == 1
    first_payload = json.loads(chat_reqs[0].data.decode("utf-8"))
    first_system = first_payload["messages"][0]["content"]
    tool_names = [tool["function"]["name"] for tool in first_payload.get("tools", [])]
    assert "Detailed internal prompt for DOCX workflows." not in first_system
    assert "Loaded skill guidance:" not in first_system
    assert "docx: convert documents to docx" in first_system
    assert "skill_view" in tool_names


def test_reasoning_tokens_strip_think_markers_in_journal_and_output(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
        }
    }
    agent = Agent(cfg, runtime)

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"reasoning_content":"<think>internal chain</think>"}}]}',
                'data: {"choices":[{"delta":{"content":"Visible answer"}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "say visible answer"}],
        user_input="say visible answer",
        thinking=True,
    )

    assert result.status == "done"
    assert "<think>" not in result.reasoning
    assert "</think>" not in result.reasoning
    assert "internal chain" in result.reasoning


def test_request_user_input_tool_returns_direct_follow_up_to_user(mocker, tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "asker").mkdir(parents=True)
    (skills / "asker" / "SKILL.md").write_text(
        """
---
name: asker
description: ask for clarification
version: 1.0.0
---
Ask a follow-up before continuing.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
        }
    }
    agent = Agent(cfg, runtime)
    chat_reqs: list[urllib.request.Request] = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        chat_reqs.append(req)
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"request_user_input","arguments":"{\\"question\\": \\\"Pick a format\\\", \\\"options\\\": [\\\"pdf\\\", \\\"docx\\\"]}"}}]}}]}',
                'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "use skill asker"}],
        user_input="use skill asker",
        thinking=True,
    )

    assert result.status == "done"
    assert result.content == "Pick a format\nOptions: pdf | docx"
    assert len(chat_reqs) == 1


def test_request_user_input_halts_turn_before_later_tool_calls(mocker, tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "asker").mkdir(parents=True)
    (skills / "asker" / "SKILL.md").write_text(
        """
---
name: asker
description: ask for clarification
version: 1.0.0
---
Ask a follow-up before continuing.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    cfg = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:8080/v1/models",
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
        }
    }
    agent = Agent(cfg, runtime)
    calls_seen: list[str] = []
    real_execute = runtime.execute_tool_call

    def wrapped_execute(tool_name, args, selected, ctx, **kwargs):
        calls_seen.append(tool_name)
        return real_execute(tool_name, args, selected, ctx, **kwargs)

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"request_user_input","arguments":"{\\"question\\": \\\"Pick a format\\\", \\\"options\\\": [\\\"pdf\\\", \\\"docx\\\"]}"}},{"index":1,"id":"call_2","type":"function","function":{"name":"read_skill_resource","arguments":"{\\"skill_id\\": \\\"asker\\\", \\\"path\\": \\\"LICENSE.txt\\"}"}}]}}]}',
                'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(runtime, "execute_tool_call", side_effect=wrapped_execute)
    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "use skill asker"}],
        user_input="use skill asker",
        thinking=True,
    )

    assert result.status == "done"
    assert result.content == "Pick a format\nOptions: pdf | docx"
    assert calls_seen == ["request_user_input"]



def test_run_turn_allows_shell_command_for_environment_question(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    mocker.patch.object(agent, "_classify_turn", return_value=TurnClassification(prefer_local_workspace_tools=False))

    pass_calls = []
    executed = []

    def fake_turn_call(payload, stop_event, on_event, pass_id):
        pass_calls.append(pass_id)
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
                                "arguments": {"command": "go version"},
                            },
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
                    "content": "Go version: go1.24.0",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        raise AssertionError(f"Unexpected pass id: {pass_id}")

    def fake_execute_tool_call(tool_name, args, selected, ctx, confirm_shell=None, **_kwargs):
        executed.append((tool_name, args))
        assert tool_name == "shell_command"
        assert args["command"] == "go version"
        return {
            "ok": True,
            "data": {"stdout": "go version go1.24.0 linux/amd64\n", "stderr": "", "returncode": 0},
            "error": None,
            "meta": {},
        }

    mocker.patch.object(agent, "_call_with_retry", side_effect=fake_turn_call)
    mocker.patch.object(runtime, "execute_tool_call", side_effect=fake_execute_tool_call)

    result = agent.run_turn(
        history_messages=[],
        user_input="What version of go am I using?",
        thinking=True,
        confirm_shell=lambda _command: True,
    )

    assert result.status == "done"
    assert result.content == "Go version: go1.24.0"
    assert executed == [("shell_command", {"command": "go version"})]
    assert "local workspace file tasks" not in result.content
    assert pass_calls == ["pass_1", "pass_2"]
