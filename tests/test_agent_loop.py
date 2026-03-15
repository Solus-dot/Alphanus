from __future__ import annotations

import json
import threading
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

    (skills / "workspace-ops" / "SKILL.md").write_text(
        """
---
name: workspace-ops
description: workspace
version: 1.0.0
tools:
  allowed-tools:
    - create_file
---
workspace
""".strip(),
        encoding="utf-8",
    )
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


def test_skill_snapshot_refreshes_only_after_runtime_generation_changes(tmp_path: Path):
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
        config={"skills": {"selection_mode": "model", "max_active_skills": 2}},
    )
    agent = Agent({"agent": {}}, runtime)

    snap1 = agent._get_skill_snapshot()
    snap2 = agent._get_skill_snapshot()
    assert snap1 is snap2
    assert [skill.id for skill in snap1.skills] == ["alpha"]

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

    snap3 = agent._get_skill_snapshot()
    assert snap3 is not snap1
    assert snap3.generation == runtime.generation
    assert [skill.id for skill in snap3.skills] == ["alpha", "beta"]


def test_finalization_falls_back_immediately_when_model_leaks_tool_markup(mocker, runtime: SkillRuntime):
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
        raise AssertionError("Unexpected extra completion call")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "tell me about meta"}],
        user_input="tell me about meta",
        thinking=True,
    )

    assert result.status == "done"
    assert "<tool_call>" not in result.content
    assert "I couldn't turn the available tool and model output into a clean user-facing answer" in result.content
    assert len(chat_reqs) == 2


def test_finalization_falls_back_to_sources_when_markup_repeats(mocker, runtime: SkillRuntime):
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

    assert result.status == "done"
    assert "Relevant sources to check:" in result.content
    assert "https://about.fb.com/news/" in result.content
    assert len(chat_reqs) == 2


def test_finalization_falls_back_to_generic_tool_summary_without_search_contamination(mocker, runtime: SkillRuntime):
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
            "name": "get_weather",
            "content": json.dumps(
                {
                    "ok": True,
                    "data": {
                        "city": "London",
                        "temperature_c": 14,
                        "condition": "Cloudy",
                    },
                    "error": None,
                    "meta": {},
                }
            ),
        }
    ]

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"<tool_call>\\n<function=get_weather>\\n<parameter=city>London</parameter>\\n</function>\\n</tool_call>"}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent._run_finalization_pass(
        "system",
        history,
        "",
        history[:],
        threading.Event(),
        None,
        "pass",
        allow_search_fallback=False,
    )

    assert result.status == "done"
    assert "get_weather" in result.content
    assert "temperature_c" in result.content
    assert "I couldn't verify the answer from reliable web results" not in result.content


def test_search_fallback_requires_actual_search_activity(runtime: SkillRuntime):
    agent = Agent(
        {
            "agent": {
                "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
                "models_endpoint": "http://127.0.0.1:8080/v1/models",
            }
        },
        runtime,
    )

    assert agent._search_fallback_allowed(
        {"web_search": 0, "fetch_url": 0},
        search_failure_count=0,
        search_has_success=False,
        search_has_fetch_content=False,
    ) is False
    assert agent._search_fallback_allowed(
        {"web_search": 1, "fetch_url": 0},
        search_failure_count=0,
        search_has_success=False,
        search_has_fetch_content=False,
    ) is True


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
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"web_search","arguments":"{\\"query\\": \\\"meta latest acquisitions\\\"}"}}]}}]}',
                'data: {"choices":[{"finish_reason":"tool_calls"}]}',
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
    assert "I couldn't verify" in result.content


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
                'data: {"choices":[{"delta":{"content":"I checked current web results before answering."}}]}',
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
    assert result.content == "I checked current web results before answering."
    assert len(chat_reqs) == 3


def test_model_skill_router_selects_skill_before_tool_turn(mocker, runtime: SkillRuntime):
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
        "skills": {
            "selection_mode": "model",
            "max_active_skills": 2,
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
                    'data: {"choices":[{"delta":{"content":"{\\"skills\\": [\\"search-ops\\"]}"}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        if len(chat_reqs) == 2:
            payload = json.loads(req.data.decode("utf-8"))
            tool_names = [tool["function"]["name"] for tool in payload.get("tools", [])]
            assert tool_names == ["web_search"]
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
    assert result.content == "I checked the selected search skill before answering."
    assert len(chat_reqs) == 3


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

    def fake_execute_tool_call(tool_name, args, selected, ctx, confirm_shell=None):
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

    def fake_execute_tool_call(tool_name, args, selected, ctx, confirm_shell=None):
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
