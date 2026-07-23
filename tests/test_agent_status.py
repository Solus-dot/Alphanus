from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

import pytest

from agent.core import Agent
from agent.provider_metadata import ProviderMetadataExtractor
from core.memory import LexicalMemory
from core.project import ProjectRuntime
from core.types import ModelStatus
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


def test_doctor_report_uses_env_auth_header(mocker, runtime: SkillRuntime, monkeypatch):
    monkeypatch.setenv("ALPHANUS_AUTH_HEADER", "Authorization: Bearer test")
    cfg = {
        "agent": {
            "model_endpoint": TEST_MODEL_ENDPOINT,
            "models_endpoint": TEST_MODELS_ENDPOINT,
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
        },
        "search": {"provider": "searxng", "fallback_provider": "tavily", "searxng_base_url": "http://127.0.0.1:8888"},
    }
    agent = Agent(cfg, runtime)

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        raise AssertionError("unexpected endpoint")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)
    report = cast(Any, agent.doctor_report())
    assert report["agent"]["auth_header_source"] == "env"
    assert report["agent"]["permission_mode"] == "project-write"
    assert report["harness_metrics"]["turns_total"] == 0


def test_typed_config_provider_uses_resolved_auth_header(runtime: SkillRuntime, monkeypatch):
    monkeypatch.setenv("CUSTOM_ALPHANUS_KEY", "secret-token")

    agent = Agent(
        {
            "agent": {
                "api_key": "env:CUSTOM_ALPHANUS_KEY",
                "api_key_env": "CUSTOM_ALPHANUS_KEY",
                "auth_header_template": "Authorization: Bearer {api_key}",
            },
        },
        runtime,
    )

    assert agent.llm_client.provider_config.auth_header == "Authorization: Bearer secret-token"
    assert agent.llm_client.provider_config.auth_header == "Authorization: Bearer secret-token"


def test_doctor_report_handles_non_object_runtime_and_capabilities_sections(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": TEST_MODEL_ENDPOINT,
            "models_endpoint": TEST_MODELS_ENDPOINT,
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
        },
        "runtime": "not-a-dict",
        "permissions": ["not", "a", "dict"],
    }
    agent = Agent(cfg, runtime)

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        raise AssertionError("unexpected endpoint")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)
    report = cast(Any, agent.doctor_report())
    assert report["agent"]["permission_mode"] == "project-write"
    assert report["agent"]["approvals"] == "on-boundary"


def test_doctor_report_can_skip_readiness_probe(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": TEST_MODEL_ENDPOINT,
            "models_endpoint": TEST_MODELS_ENDPOINT,
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
        },
    }
    agent = Agent(cfg, runtime)
    mocker.patch.object(agent.llm_client, "get_model_status", return_value=ModelStatus(state="online"))
    ensure_ready = mocker.patch.object(agent, "ensure_ready", side_effect=AssertionError("should not probe readiness"))

    report = cast(Any, agent.doctor_report(probe_ready=False))

    assert report["agent"]["ready"] is True
    ensure_ready.assert_not_called()


def test_doctor_report_supports_searxng_provider(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": TEST_MODEL_ENDPOINT,
            "models_endpoint": TEST_MODELS_ENDPOINT,
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
        },
        "search": {"provider": "searxng", "searxng_base_url": "http://127.0.0.1:8888"},
    }
    agent = Agent(cfg, runtime)

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        raise AssertionError("unexpected endpoint")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)
    report = cast(Any, agent.doctor_report())
    assert report["search"]["provider"] == "searxng"
    assert report["search"]["fallback_provider"] == "tavily"
    assert report["search"]["ready"] is True
    assert report["search"]["searxng_base_url"] == "http://127.0.0.1:8888"
    assert report["search"]["reason"] == ""
    assert report["agent"]["backend_profile_requested"] == "auto"
    assert report["agent"]["backend_profile_selected"] in {"unknown", "auto", "mlx_vlm", "llamacpp", "ollama", "vllm", "lmstudio"}


def test_refresh_model_status_reads_first_model_id(mocker, runtime: SkillRuntime):
    cfg = agent_config(max_tokens=None)
    agent = Agent(cfg, runtime)

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"n_ctx":8192}'])
        assert req.full_url.endswith("/v1/models")
        return FakeResponse(['{"data":[{"id":"llama-3.2-3b-instruct"}]}'])

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    assert agent.refresh_model_status(force=True).model_name == "llama-3.2-3b-instruct"


def test_refresh_model_status_reads_model_id_and_context_window(mocker, runtime: SkillRuntime):
    cfg = agent_config(max_tokens=None)
    agent = Agent(cfg, runtime)

    seen_urls: list[str] = []

    def fake_urlopen(req, timeout=None, context=None):
        seen_urls.append(req.full_url)
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"n_ctx":24576}'])
        assert req.full_url.endswith("/v1/models")
        return FakeResponse(['{"data":[{"id":"llama-3.2-3b-instruct","metadata":{"n_ctx_slot":40960}}]}'])

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    status = agent.refresh_model_status(force=True)
    assert (status.model_name, status.context_window) == ("llama-3.2-3b-instruct", 24576)
    assert seen_urls == [
        TEST_SLOTS_ENDPOINT,
        TEST_MODELS_ENDPOINT,
    ]


def test_select_skills_returns_only_explicitly_loaded_skills(runtime: SkillRuntime):
    runtime.config = {}
    agent = Agent({"agent": {}}, runtime)
    ctx = SkillContext(
        user_input="write a file",
        branch_labels=[],
        attachments=[],
        project_root=str(runtime.project.project_root),
        memory_hits=[],
    )
    assert agent.skill_runtime.select_skills(ctx) == []


def test_refresh_model_status_falls_back_to_slots_for_context_window(mocker, runtime: SkillRuntime):
    cfg = agent_config(max_tokens=None)
    agent = Agent(cfg, runtime)
    seen_urls: list[str] = []

    def fake_urlopen(req, timeout=None, context=None):
        seen_urls.append(req.full_url)
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"n_ctx":40960,"params":{"seed":1}}'])
        if req.full_url.endswith("/v1/models"):
            return FakeResponse(['{"data":[{"id":"llama-3.2-3b-instruct"}]}'])
        raise AssertionError(f"unexpected endpoint: {req.full_url}")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    status = agent.refresh_model_status(force=True)
    assert (status.model_name, status.context_window) == ("llama-3.2-3b-instruct", 40960)
    assert seen_urls == [
        TEST_SLOTS_ENDPOINT,
        TEST_MODELS_ENDPOINT,
    ]


def test_refresh_model_status_falls_back_to_props_after_slots_miss(mocker, runtime: SkillRuntime):
    cfg = agent_config(max_tokens=None)
    agent = Agent(cfg, runtime)
    seen_urls: list[str] = []

    def fake_urlopen(req, timeout=None, context=None):
        seen_urls.append(req.full_url)
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"params":{"seed":1}}'])
        if req.full_url.endswith("/v1/models"):
            return FakeResponse(['{"data":[{"id":"llama-3.2-3b-instruct"}]}'])
        if req.full_url.endswith("/props"):
            return FakeResponse(['{"default_generation_settings":{"n_ctx":40960}}'])
        raise AssertionError(f"unexpected endpoint: {req.full_url}")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    status = agent.refresh_model_status(force=True)
    assert (status.model_name, status.context_window) == ("llama-3.2-3b-instruct", 40960)
    assert seen_urls == [
        TEST_SLOTS_ENDPOINT,
        TEST_MODELS_ENDPOINT,
        TEST_PROPS_ENDPOINT,
    ]


def test_refresh_model_status_reports_online_and_context_window(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"n_ctx":16384}'])
        assert req.full_url.endswith("/v1/models")
        return FakeResponse(['{"data":[{"id":"qwen-3","metadata":{"n_ctx_slot":16384}}]}'])

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    status = agent.refresh_model_status(force=True)

    assert status.state == "online"
    assert status.model_name == "qwen-3"
    assert status.context_window == 16384
    assert agent.get_model_status().state == "online"
    assert agent.llm_client._ready_checked is True


def test_refresh_model_status_shares_timeout_budget_across_probes(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    provider = agent.llm_client
    probe_timeouts: list[tuple[str, float]] = []

    mocker.patch(
        "agent.provider.time.monotonic",
        side_effect=[100.0, 100.0, 100.4, 100.8],
    )

    def fake_fetch_json(url: str, timeout_s=None):
        assert timeout_s is not None
        if url.endswith("/slots"):
            probe_timeouts.append(("slots", float(timeout_s)))
            return {"id": 0, "params": {"seed": 1}}
        if url.endswith("/props"):
            probe_timeouts.append(("props", float(timeout_s)))
            return {"default_generation_settings": {"n_ctx": 40960}}
        raise AssertionError(f"unexpected endpoint: {url}")

    def fake_list_models(timeout_s=None):
        assert timeout_s is not None
        probe_timeouts.append(("models", float(timeout_s)))
        return {"data": [{"id": "qwen-3"}]}

    mocker.patch.object(provider, "fetch_json", side_effect=fake_fetch_json)
    mocker.patch.object(provider, "list_models", side_effect=fake_list_models)

    status = provider.refresh_model_status(timeout_s=1.0, force=True)

    assert status.model_name == "qwen-3"
    assert status.context_window == 40960
    assert probe_timeouts == [
        ("slots", pytest.approx(1.0)),
        ("models", pytest.approx(0.6)),
        ("props", pytest.approx(0.2)),
    ]


def test_reload_config_resets_model_status_cache(runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    agent.llm_client._store_model_status(
        ModelStatus(
            state="online",
            model_name="qwen-old",
            context_window=8192,
            last_checked_at=123.0,
            last_success_at=123.0,
            endpoint=TEST_MODELS_ENDPOINT,
        )
    )

    agent.reload_config(
        {
            "agent": {
                "model_endpoint": "http://127.0.0.1:8081/v1/chat/completions",
                "models_endpoint": "http://127.0.0.1:8081/v1/models",
            }
        }
    )

    status = agent.get_model_status()

    assert status.state == "unknown"
    assert status.endpoint == "http://127.0.0.1:8081/v1/models"
    assert status.model_name is None
    assert agent.llm_client._ready_checked is False


def test_fetch_model_name_accepts_top_level_model_field(runtime: SkillRuntime):
    assert ProviderMetadataExtractor.extract_model_name({"model": "qwen2.5-coder-7b"}) == "qwen2.5-coder-7b"


def test_fetch_model_name_accepts_nested_model_id_field(runtime: SkillRuntime):
    assert (
        ProviderMetadataExtractor.extract_model_name({"models": [{"model_id": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"}]})
        == "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    )


def test_fetch_model_name_returns_none_when_models_list_is_empty(runtime: SkillRuntime):
    assert ProviderMetadataExtractor.extract_model_name({"object": "list", "data": []}) is None


def test_extract_model_context_window_accepts_nested_metadata(runtime: SkillRuntime):
    assert ProviderMetadataExtractor.extract_model_context_window({"data": [{"metadata": {"n_ctx_slot": 40960}}]}) == 40960


def test_extract_model_context_window_accepts_recursive_nested_fields(runtime: SkillRuntime):
    assert ProviderMetadataExtractor.extract_model_context_window({"default_generation_settings": {"nested": {"n_ctx": 40960}}}) == 40960


def test_reload_config_resets_readiness_state(runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": TEST_MODEL_ENDPOINT,
            "models_endpoint": TEST_MODELS_ENDPOINT,
            "request_timeout_s": 5,
            "readiness_timeout_s": 1,
            "readiness_poll_s": 0.01,
            "enable_thinking": True,
            "tls_verify": True,
        },
        "context": {"context_limit": 8192, "keep_last_n": 10, "safety_margin": 500},
    }
    agent = Agent(cfg, runtime)
    agent.llm_client._ready_checked = True

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

    assert agent.llm_client.model_endpoint == "http://127.0.0.1:8081/v1/chat/completions"
    assert agent.llm_client.models_endpoint == "http://127.0.0.1:8081/v1/models"
    assert agent.llm_client._ready_checked is False


def test_reload_config_rebuilds_context_manager(runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": TEST_MODEL_ENDPOINT,
            "models_endpoint": TEST_MODELS_ENDPOINT,
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


def test_reload_config_coerces_invalid_numeric_values(runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": TEST_MODEL_ENDPOINT,
            "models_endpoint": TEST_MODELS_ENDPOINT,
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
            "agent": {
                **cfg["agent"],
                "max_action_depth": "invalid",
                "max_tool_result_chars": None,
                "max_reasoning_chars": "-20",
                "context_budget_max_tokens": "not-an-int",
                "max_classifier_tokens": "bad",
                "tool_budgets": {
                    "web_search": "bad-value",
                    "fetch_url": 0,
                    "recall_memory": -4,
                    "custom_lookup": "7",
                },
            },
            "context": {"context_limit": "bad", "keep_last_n": 0, "safety_margin": "-3"},
        }
    )

    assert agent.context_mgr.context_limit == 8192
    assert agent.context_mgr.keep_last_n == 1
    assert agent.context_mgr.safety_margin == 0
    assert agent.llm_client.max_classifier_tokens == 256
    assert agent.orchestrator.max_action_depth == 10
    assert agent.orchestrator.history.max_chars == 12000
    assert agent.orchestrator.max_reasoning_chars == 0
    assert agent.orchestrator.context_budget_max_tokens == 2048
    assert agent.orchestrator.default_tool_budgets["web_search"] == 2
    assert agent.orchestrator.default_tool_budgets["fetch_url"] == 1
    assert agent.orchestrator.default_tool_budgets["recall_memory"] == 1
    assert agent.orchestrator.default_tool_budgets["custom_lookup"] == 7


def test_run_turn_fails_fast_when_cached_status_is_freshly_offline(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    agent.llm_client._store_model_status(
        ModelStatus(
            state="offline",
            model_name="qwen-down",
            last_checked_at=time.monotonic(),
            last_success_at=time.monotonic(),
            last_error="[Errno 61] Connection refused",
            endpoint=agent.llm_client.models_endpoint,
        )
    )
    run = mocker.patch.object(agent.orchestrator, "run_turn")
    refresh = mocker.patch.object(agent, "refresh_model_status", wraps=agent.refresh_model_status)

    result = agent.run_turn(history_messages=[], user_input="hello", thinking=True)

    assert result.status == "error"
    assert "offline" in (result.error or "").lower()
    assert "Is the local model server running" in (result.error or "")
    assert "Errno 61" not in (result.error or "")
    run.assert_not_called()
    refresh.assert_not_called()


def test_run_turn_waits_for_cold_local_model_despite_fresh_offline_status(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    agent.llm_client._store_model_status(
        ModelStatus(
            state="offline",
            model_name=None,
            last_checked_at=time.monotonic(),
            last_success_at=0.0,
            last_error="[Errno 61] Connection refused",
            endpoint=agent.llm_client.models_endpoint,
        )
    )
    ready = mocker.patch.object(agent, "ensure_ready", return_value=True)
    run = mocker.patch.object(
        agent.orchestrator,
        "run_turn",
        return_value=type("Result", (), {"status": "done", "content": "", "reasoning": "", "skill_exchanges": []})(),
    )

    result = agent.run_turn(history_messages=[], user_input="hello", thinking=True)

    assert result.status == "done"
    ready.assert_called_once()
    run.assert_called_once()


def test_run_turn_probes_when_status_is_stale(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    agent.llm_client._store_model_status(
        ModelStatus(
            state="online",
            model_name="qwen-stale",
            context_window=8192,
            last_checked_at=1.0,
            last_success_at=1.0,
            endpoint=agent.llm_client.models_endpoint,
        )
    )
    refreshed = ModelStatus(
        state="online",
        model_name="qwen-fresh",
        context_window=16384,
        last_checked_at=time.monotonic(),
        last_success_at=time.monotonic(),
        endpoint=agent.llm_client.models_endpoint,
    )
    refresh = mocker.patch.object(agent, "refresh_model_status", return_value=refreshed)
    run = mocker.patch.object(
        agent.orchestrator,
        "run_turn",
        return_value=type("Result", (), {"status": "done", "content": "", "reasoning": "", "skill_exchanges": []})(),
    )

    result = agent.run_turn(history_messages=[], user_input="hello", thinking=True)

    assert result.status == "done"
    refresh.assert_called_once()
    run.assert_called_once()


def test_run_turn_uses_configured_readiness_timeout_before_first_turn(mocker, runtime: SkillRuntime):
    cfg = {
        "agent": {
            "model_endpoint": TEST_MODEL_ENDPOINT,
            "models_endpoint": TEST_MODELS_ENDPOINT,
            "readiness_timeout_s": 30,
        }
    }
    agent = Agent(cfg, runtime)
    agent.llm_client._store_model_status(ModelStatus(state="unknown", endpoint=agent.llm_client.models_endpoint))
    ready = mocker.patch.object(agent, "ensure_ready", return_value=True)
    run = mocker.patch.object(
        agent.orchestrator,
        "run_turn",
        return_value=type("Result", (), {"status": "done", "content": "", "reasoning": "", "skill_exchanges": []})(),
    )

    result = agent.run_turn(history_messages=[], user_input="hello", thinking=True)

    assert result.status == "done"
    ready.assert_called_once()
    assert ready.call_args.kwargs.get("timeout_s") is None
    run.assert_called_once()


def test_local_connection_refused_is_not_retried(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    agent.llm_client._store_model_status(
        ModelStatus(
            state="online",
            model_name="qwen-3",
            context_window=8192,
            last_checked_at=1.0,
            last_success_at=1.0,
            endpoint=agent.llm_client.models_endpoint,
        )
    )
    events: list[dict] = []

    def boom(*_args, **_kwargs):
        raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))

    mocker.patch.object(
        agent.llm_client,
        "_status_allows_immediate_send",
        return_value=ModelStatus(state="online", endpoint=agent.llm_client.models_endpoint),
    )
    mocker.patch("agent.provider.stream_chat_completions", side_effect=boom)

    with pytest.raises(Exception):
        agent.llm_client.call_with_retry({"messages": []}, None, events.append, pass_id="pass_1")

    assert not any("Retrying request" in event.get("text", "") for event in events if isinstance(event, dict))
    assert agent.get_model_status().state == "offline"
    assert agent.get_model_status().model_name == "qwen-3"
    assert "Is the local model server running" in agent.get_model_status().last_error
    assert "Errno 61" not in agent.get_model_status().last_error


def test_refresh_model_status_formats_connection_refused_for_users(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(
        urllib.request,
        "urlopen",
        side_effect=urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")),
    )

    status = agent.refresh_model_status(force=True)

    assert status.state == "offline"
    assert "Connection refused by model endpoint" in status.last_error
    assert "Is the local model server running" in status.last_error
    assert "Errno 61" not in status.last_error


def test_retryable_transport_error_still_runs_readiness_poll_after_offline_probe(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    events: list[dict] = []
    mocker.patch.object(
        agent.llm_client,
        "_status_allows_immediate_send",
        return_value=ModelStatus(state="online", endpoint=agent.llm_client.models_endpoint),
    )
    stream = mocker.patch(
        "agent.provider.stream_chat_completions",
        side_effect=urllib.error.URLError(TimeoutError("timed out")),
    )
    mocker.patch.object(
        agent.llm_client,
        "refresh_model_status",
        return_value=ModelStatus(state="offline", last_error="timed out", endpoint=agent.llm_client.models_endpoint),
    )
    ready = mocker.patch.object(agent.llm_client, "check_ready", return_value=False)

    with pytest.raises(Exception):
        agent.llm_client.call_with_retry({"messages": []}, None, events.append, pass_id="pass_1")

    assert stream.called
    ready.assert_called_once()
    assert ready.call_args.kwargs["timeout_s"] == min(agent.llm_client.readiness_timeout_s, 5.0)


def test_refresh_model_status_preserves_model_name_when_endpoint_goes_offline(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    agent.llm_client._store_model_status(
        ModelStatus(
            state="online",
            model_name="qwen-3",
            context_window=8192,
            last_checked_at=1.0,
            last_success_at=1.0,
            endpoint=agent.llm_client.models_endpoint,
        )
    )
    mocker.patch.object(urllib.request, "urlopen", side_effect=urllib.error.URLError("offline"))

    status = agent.refresh_model_status(force=True)

    assert status.state == "offline"
    assert status.model_name == "qwen-3"
    assert status.context_window == 8192


def test_refresh_model_status_detects_hot_swapped_model(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    refresh_index = {"value": 0}

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/slots"):
            refresh_index["value"] += 1
            if refresh_index["value"] == 1:
                return FakeResponse(['{"id":0,"n_ctx":8192}'])
            return FakeResponse(['{"id":0,"n_ctx":16384}'])
        if req.full_url.endswith("/v1/models"):
            if refresh_index["value"] == 1:
                return FakeResponse(['{"data":[{"id":"qwen-3"}]}'])
            return FakeResponse(['{"data":[{"id":"qwen-4"}]}'])
        raise AssertionError(f"unexpected endpoint: {req.full_url}")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    first = agent.refresh_model_status(force=True)
    second = agent.refresh_model_status(force=True)

    assert first.model_name == "qwen-3"
    assert second.model_name == "qwen-4"
    assert second.context_window == 16384


def test_time_sensitive_search_without_fetch_evidence_declines(mocker, tmp_path: Path):
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
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    cfg = {
        "agent": {
            "model_endpoint": TEST_MODEL_ENDPOINT,
            "models_endpoint": TEST_MODELS_ENDPOINT,
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
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"web_search","arguments":"{\\"query\\": \\"iran current situation\\"}"}}]}}]}',
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
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    cfg = {
        "agent": {
            "model_endpoint": TEST_MODEL_ENDPOINT,
            "models_endpoint": TEST_MODELS_ENDPOINT,
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
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"n_ctx":40960}'])
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
    first_payload = cast(Any, json.loads(cast(bytes, chat_reqs[0].data).decode("utf-8")))
    first_system = first_payload["messages"][0]["content"]
    tool_names = [tool["function"]["name"] for tool in first_payload.get("tools", [])]
    assert "Detailed internal prompt for DOCX workflows." not in first_system
    assert "Loaded skill guidance:" not in first_system
    assert "docx: convert documents to docx" in first_system
    assert "skill_view" in tool_names


def test_reasoning_tokens_strip_think_markers_in_journal_and_output(mocker, runtime: SkillRuntime):
    cfg = agent_config()
    agent = Agent(cfg, runtime)

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"n_ctx":40960}'])
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
