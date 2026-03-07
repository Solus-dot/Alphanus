from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from agent.core import Agent
from core.memory import VectorMemory
from core.skills import SkillRuntime
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


@pytest.fixture
def runtime(tmp_path: Path) -> SkillRuntime:
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "workspace-ops").mkdir(parents=True)

    (skills / "workspace-ops" / "skill.toml").write_text(
        """
schema_version = "1.0.0"
id = "workspace-ops"
name = "Workspace"
version = "1.0.0"
description = "workspace"
enabled = true
priority = 90

[triggers]
keywords = ["write"]
file_ext = [".py"]
capabilities = ["workspace_write", "workspace_read"]
""".strip(),
        encoding="utf-8",
    )
    (skills / "workspace-ops" / "prompt.md").write_text("workspace", encoding="utf-8")
    (skills / "workspace-ops" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "create_file": {
    "capability": "workspace_write",
    "description": "Create file",
    "parameters": {
      "type": "object",
      "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}},
      "required": ["filepath", "content"]
    }
  }
}

def execute(tool_name, args, env):
    if tool_name == "create_file":
        path = env.workspace.create_file(args["filepath"], args["content"])
        return {"ok": True, "data": {"filepath": path}, "error": None, "meta": {}}
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
