from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

import pytest

from agent.core import Agent
from agent.policies import PromptPolicyRenderer
from core.types import TurnClassification
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

    cfg = agent_config()
    agent = Agent(cfg, runtime)
    mocker.patch.object(
        agent.classifier,
        "classify",
        return_value=TurnClassification(time_sensitive=True),
    )

    chat_reqs = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"n_ctx":40960}'])
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
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"web_search","arguments":"{\\"query\\": \\"meta latest acquisitions\\"}"}}]}}]}',
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
        loaded_skill_ids=["search-ops"],
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

    cfg = agent_config()
    agent = Agent(cfg, runtime)

    chat_reqs = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"n_ctx":40960}'])
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
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"web_search","arguments":"{\\"query\\": \\"meta latest acquisitions\\"}"}}]}}]}',
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


def test_session_loaded_search_skill_exposes_search_tool_on_first_turn(mocker, runtime: SkillRuntime):
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

    cfg = agent_config()
    agent = Agent(cfg, runtime)

    chat_reqs = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"n_ctx":40960}'])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            payload = json.loads(req.data.decode("utf-8"))
            tool_names = [tool["function"]["name"] for tool in payload.get("tools", [])]
            assert "web_search" in tool_names
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"web_search","arguments":"{\\"query\\": \\"meta latest acquisitions\\"}"}}]}}]}',
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
        loaded_skill_ids=["search-ops"],
    )

    assert result.status == "done"


def test_time_sensitive_turn_recomputes_search_safeguards_after_skill_view(mocker, runtime: SkillRuntime):
    search_skill = runtime.skills_dir / "search-ops"
    search_skill.mkdir(parents=True)
    (search_skill / "SKILL.md").write_text(
        """
---
name: search-ops
description: Search the web for current information.
allowed-tools: web_search
---
Search the web.
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

    cfg = agent_config()
    agent = Agent(cfg, runtime)
    mocker.patch.object(agent.classifier, "classify", return_value=TurnClassification(time_sensitive=True))

    chat_reqs: list[urllib.request.Request] = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"n_ctx":40960}'])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"skill_view","arguments":"{\\"name\\": \\"search-ops\\", \\"file_path\\": \\"\\"}"}}]}}]}',
                    'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )
        if len(chat_reqs) == 2:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"content":"As of my knowledge, no major updates."}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        if len(chat_reqs) == 3:
            payload = json.loads(req.data.decode("utf-8"))
            system_message = payload["messages"][0]["content"]
            assert "You must call web_search before answering." in system_message
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_2","type":"function","function":{"name":"web_search","arguments":"{\\"query\\": \\"latest acquisitions\\"}"}}]}}]}',
                    'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"I checked the web before answering."}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "latest acquisitions"}],
        user_input="latest acquisitions",
        thinking=True,
    )

    assert result.status == "done"
    assert "checked the web" in result.content
    assert len(chat_reqs) >= 4


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
    "capability": "project_write",
    "description": "Forge an artifact",
    "parameters": {
      "type": "object",
      "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}},
      "required": ["filepath", "content"]
    }
  }
}

def execute(tool_name, args, env):
    path = env.project.create_file(args["filepath"], args["content"])
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
        project_root=str(runtime.project.project_root),
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


def test_prompt_policy_renderer_uses_configured_context_limit_for_skill_budget(mocker, runtime: SkillRuntime):
    skill_runtime = mocker.Mock()
    skill_runtime.compose_skill_index.return_value = ""
    skill_runtime.compose_skill_block.return_value = "loaded guidance"
    renderer = PromptPolicyRenderer("system", skill_runtime, context_limit=4096)
    selected = cast(Any, [object()])
    ctx = SkillContext(
        user_input="hello",
        branch_labels=[],
        attachments=[],
        project_root=str(runtime.project.project_root),
        memory_hits=[],
    )

    rendered = renderer.compose_system_content(selected, ctx)

    assert "loaded guidance" in rendered
    skill_runtime.compose_skill_block.assert_called_once_with(selected, context_limit=4096)


def test_large_tool_call_args_are_compacted_in_history(runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    large = "x" * 5000
    compacted = cast(Any, agent.orchestrator.history.arguments({"filepath": "a.txt", "content": large}))

    assert compacted["filepath"] == "a.txt"
    assert compacted["content"].startswith("x" * 1200)
    assert "[history excerpt; 3800 chars omitted]" in compacted["content"]
    assert "truncated" not in compacted["content"]


def test_large_non_content_tool_call_args_still_use_generic_truncation(runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    large = "x" * 5000
    compacted = cast(Any, agent.orchestrator.history.arguments({"query": large}))

    assert compacted["query"].startswith("x" * 1200)
    assert "[truncated 3800 chars]" in compacted["query"]
    assert len(compacted["query"]) < len(large)


def test_tool_result_compaction_preserves_nested_scalars(runtime: SkillRuntime):
    agent = Agent({"agent": {"max_tool_result_chars": 100}}, runtime)
    result = {
        "ok": True,
        "data": {"a": {"b": {"c": {"d": {"value": "keep me"}}}}},
        "error": None,
        "meta": {},
    }

    compacted = cast(Any, agent.orchestrator.history.compact_result(result))

    assert compacted["data"]["a"]["b"]["c"]["d"]["value"] == "keep me"


def test_tool_result_compaction_middle_truncates_long_strings(runtime: SkillRuntime):
    agent = Agent({"agent": {"max_tool_result_chars": 120}}, runtime)
    value = "BEGIN-" + ("x" * 500) + "-END"

    compacted = cast(
        Any,
        agent.orchestrator.history.compact_result({"ok": True, "data": {"content": value}, "error": None, "meta": {}}),
    )

    content = compacted["data"]["content"]
    assert content.startswith("BEGIN-")
    assert content.endswith("-END")
    assert "chars truncated" in content
    assert len(content) < len(value)


def test_memory_tool_history_compaction_keeps_recalled_text(runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    result = {
        "ok": True,
        "data": {
            "hits": [
                {
                    "id": 42,
                    "text": "The user's name is Sohom.",
                    "type": "preference",
                    "score": 0.91,
                    "timestamp": 123.0,
                    "metadata": {"source": "test", "nested": {"kept": "simple"}},
                }
            ]
        },
        "error": None,
        "meta": {},
    }

    compacted = cast(Any, agent.orchestrator.history.result("recall_memory", result))

    hit = compacted["data"]["hits"][0]
    assert hit["id"] == 42
    assert hit["text"] == "The user's name is Sohom."
    assert hit["text_truncated"] is False
    assert hit["score"] == 0.91
    assert hit["metadata"]["nested"]["kept"] == "simple"


def test_memory_tool_history_compaction_uses_memory_cap_before_generic_cap(runtime: SkillRuntime):
    agent = Agent({"agent": {"max_tool_result_chars": 100}}, runtime)
    memory_text = "name: " + ("Sohom " * 200) + "done"
    result = {
        "ok": True,
        "data": {"hits": [{"id": 7, "text": memory_text, "type": "fact", "timestamp": 123.0}]},
        "error": None,
        "meta": {},
    }

    compacted = cast(Any, agent.orchestrator.history.result("recall_memory", result))

    text = compacted["data"]["hits"][0]["text"]
    assert text.startswith("name: ")
    assert text.endswith("done")
    assert len(text) > 100


def test_memory_tool_history_compaction_clips_long_memory_text(runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    long_text = "name: " + ("Sohom " * 1000) + "done"
    result = {
        "ok": True,
        "data": {"memories": [{"id": 1, "text": long_text, "type": "conversation", "timestamp": 123.0}]},
        "error": None,
        "meta": {},
    }

    compacted = cast(Any, agent.orchestrator.history.result("list_memories", result))

    memory = compacted["data"]["memories"][0]
    assert memory["id"] == 1
    assert memory["text"].startswith("name: ")
    assert memory["text"].endswith("done")
    assert memory["text_truncated"] is True
    assert len(memory["text"]) < len(long_text)


def test_read_tool_history_compaction_keeps_large_bounded_content(runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    content = "BEGIN\n" + ("x" * 70000) + "\nEND"
    result = {
        "ok": True,
        "data": {"content": content, "line_count": 3, "content_truncated": False},
        "error": None,
        "meta": {},
    }

    compacted = cast(Any, agent.orchestrator.history.result("read_file", result))

    compacted_content = compacted["data"]["content"]
    assert compacted_content.startswith("BEGIN\n")
    assert compacted_content.endswith("\nEND")
    assert "chars truncated" in compacted_content
    assert compacted["data"]["content_truncated"] is True
    assert compacted["data"]["content_omitted_chars"] > 0
    assert result["data"]["content"] == content
    assert "content_omitted_chars" not in result["data"]


def test_read_files_history_compaction_uses_read_cap_before_generic_cap(runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    content = "BEGIN\n" + ("x" * 30000) + "\nEND"
    result = {
        "ok": True,
        "data": {
            "files": [
                {
                    "filepath": "large.txt",
                    "content": content,
                    "content_truncated": False,
                    "line_count": 3,
                }
            ],
            "count": 1,
        },
        "error": None,
        "meta": {},
    }

    compacted = cast(Any, agent.orchestrator.history.result("read_files", result))

    file_result = compacted["data"]["files"][0]
    assert file_result["content"] == content
    assert file_result["content_truncated"] is False
    assert result["data"]["files"][0]["content"] == content


def test_write_tool_history_compaction_keeps_evidence_not_full_content(runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    result = {
        "ok": True,
        "data": {
            "filepath": "big.py",
            "created": True,
            "write_verified": True,
            "sha256": "abc123",
            "bytes_written": 50000,
            "line_count": 1000,
            "content_preview": "start\n" + ("x" * 5000) + "\nend",
            "content_preview_truncated": True,
            "unexpected_full_content": "y" * 50000,
        },
        "error": None,
        "meta": {},
    }

    compacted = cast(Any, agent.orchestrator.history.result("create_file", result))

    assert compacted["data"]["sha256"] == "abc123"
    assert compacted["data"]["bytes_written"] == 50000
    assert "unexpected_full_content" not in compacted["data"]
    assert len(compacted["data"]["content_preview"]) < 5000


def test_agent_transport_error_marks_error(mocker, runtime: SkillRuntime):
    cfg = agent_config()
    agent = Agent(cfg, runtime)

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"n_ctx":40960}'])
        raise urllib.error.URLError("boom")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "hi"}],
        user_input="hi",
        thinking=True,
    )

    assert result.status == "error"


def test_agent_infers_tool_calls_when_finish_reason_missing(mocker, runtime: SkillRuntime):
    cfg = agent_config()
    agent = Agent(cfg, runtime)

    chat_reqs = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"n_ctx":40960}'])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            # Backend bug: emits tool_calls but final reason is "stop".
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"create_file","arguments":"{\\"filepath\\": \\"a.txt\\", \\"content\\": \\"hello\\"}"}}]}}]}',
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
    cfg = agent_config()
    agent = Agent(cfg, runtime)

    chat_reqs = []
    events = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"n_ctx":40960}'])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"create_file","arguments":"{\\"filepath\\": \\"a.txt\\", \\"content\\": \\"hello\\"}"}}]}}]}',
                    'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )
        if len(chat_reqs) == 2:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_2","type":"function","function":{"name":"create_file","arguments":"{\\"filepath\\": \\"b.txt\\", \\"content\\": \\"world\\"}"}}]}}]}',
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


def test_tool_result_history_compacted_by_default(mocker, runtime: SkillRuntime):
    cfg = agent_config()
    agent = Agent(cfg, runtime)
    huge_text = "x" * 24000

    chat_reqs = []

    def fake_execute_tool_call(tool_name, args, selected, ctx, request_approval=None, **_kwargs):
        return {"ok": True, "data": {"blob": huge_text}, "error": None, "meta": {}}

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"n_ctx":40960}'])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"create_file","arguments":"{\\"filepath\\": \\"a.txt\\", \\"content\\": \\"hello\\"}"}}]}}]}',
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
    assert huge_text not in tool_content
    assert "blob" not in tool_content


def test_tool_result_history_compaction_can_be_disabled(mocker, runtime: SkillRuntime):
    cfg = agent_config(compact_tool_results_in_history=False)
    agent = Agent(cfg, runtime)
    huge_text = "x" * 24000

    chat_reqs = []

    def fake_execute_tool_call(tool_name, args, selected, ctx, request_approval=None, **_kwargs):
        return {"ok": True, "data": {"blob": huge_text}, "error": None, "meta": {}}

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"n_ctx":40960}'])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"create_file","arguments":"{\\"filepath\\": \\"a.txt\\", \\"content\\": \\"hello\\"}"}}]}}]}',
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
            "model_endpoint": TEST_MODEL_ENDPOINT,
            "models_endpoint": TEST_MODELS_ENDPOINT,
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

    def fake_execute_tool_call(tool_name, args, selected, ctx, request_approval=None, **_kwargs):
        return {"ok": True, "data": {"blob": huge_text}, "error": None, "meta": {}}

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"n_ctx":40960}'])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"create_file","arguments":"{\\"filepath\\": \\"a.txt\\", \\"content\\": \\"hello\\"}"}}]}}]}',
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
    assert huge_text not in tool_content
    assert "blob" not in tool_content


def test_agent_summarizes_history_when_prompt_exceeds_context_budget(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": TEST_MODEL_ENDPOINT,
            "models_endpoint": TEST_MODELS_ENDPOINT,
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
            "max_tokens": 256,
            "context_budget_max_tokens": 128,
        },
        "context": {"context_limit": 260, "keep_last_n": 2, "safety_margin": 0},
    }
    agent = Agent(cfg, runtime)
    chat_reqs = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"n_ctx":40960}'])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"content":"Summary: user wants the context system optimized."}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        if len(chat_reqs) == 2:
            payload = json.loads(req.data.decode("utf-8"))
            system_content = payload["messages"][0]["content"]
            assert "Conversation summary" in system_content
            assert "context system optimized" in system_content
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"content":"Done"}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        raise AssertionError("Unexpected extra completion call")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)
    history = cast(
        Any,
        [
            {"role": "user", "content": "old request " + ("x" * 800)},
            {"role": "assistant", "content": "old answer " + ("y" * 800)},
            {"role": "user", "content": "middle request " + ("z" * 800)},
            {"role": "assistant", "content": "middle answer " + ("q" * 800)},
            {"role": "user", "content": "what next?"},
        ],
    )

    result = agent.run_turn(history_messages=history, user_input="what next?", thinking=True)

    assert result.status == "done"
    assert result.journal["context_summary"] == "Summary: user wants the context system optimized."
    report = cast(Any, result.journal["context_report"])
    assert isinstance(report, dict)
    assert report["summary_status"] == "model"
    assert int(report["final_prompt_tokens_estimate"]) > 0


def test_agent_can_cancel_while_waiting_for_readiness(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": TEST_MODEL_ENDPOINT,
            "models_endpoint": TEST_MODELS_ENDPOINT,
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
            raise urllib.error.URLError("refused")
        raise AssertionError("chat endpoint should not be reached")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=cast(Any, [{"role": "user", "content": "hi"}]),
        user_input="hi",
        thinking=True,
        stop_event=stop_event,
    )

    assert result.status == "cancelled"
