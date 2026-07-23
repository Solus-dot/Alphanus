from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from agent.classifier import TurnClassifier
from agent.context import ContextWindowManager
from agent.orchestrator import TurnOrchestrator
from agent.policies import PromptPolicyRenderer
from agent.provider import LLMClient
from core.config_model import default_config
from core.retrieval import SQLiteRetrievalStore
from core.streaming import StreamError
from core.types import AgentTurnResult, ModelStatus, StreamPassResult, ToolCall, TurnClassification
from skills.runtime import SkillContext, SkillRuntime
from tests.support import build_skill_runtime


def _runtime(tmp_path: Path) -> SkillRuntime:
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
  }
}

def execute(tool_name, args, env):
    return {"ok": True, "data": {"filepath": env.project.create_directory(args["path"])}, "error": None, "meta": {}}
""",
    )


def _orchestrator_runtime(tmp_path: Path) -> tuple[SkillRuntime, TurnClassifier, TurnOrchestrator]:
    runtime = _runtime(tmp_path)
    cfg = default_config()
    llm_client = LLMClient(cfg)
    classifier = TurnClassifier(cfg, runtime, llm_client)
    orchestrator = TurnOrchestrator(
        skill_runtime=runtime,
        context_mgr=ContextWindowManager(),
        llm_client=llm_client,
        classifier=classifier,
        prompt_renderer=PromptPolicyRenderer("system", runtime),
    )
    return runtime, classifier, orchestrator


def _turn_state(tmp_path: Path, *, user_input: str, time_sensitive: bool, project_action: bool):
    runtime, classifier, orchestrator = _orchestrator_runtime(tmp_path)
    ctx = classifier.build_skill_context(user_input, [], [], [])
    classification = TurnClassification(
        time_sensitive=time_sensitive,
        requires_project_action=project_action,
        prefer_local_project_tools=project_action,
        source="rules",
    )
    state = orchestrator.policy_engine.build_turn_state(ctx, [], [], classification)
    return runtime, orchestrator, state


def _reload_retrieval(orchestrator: TurnOrchestrator, **updates: object) -> None:
    config = orchestrator.config.model_dump()
    config["retrieval"].update(updates)
    orchestrator.reload_config(config)


def test_fetch_tool_effects_do_not_count_unusable_text_as_fetch_evidence(tmp_path: Path) -> None:
    _runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="latest status",
        time_sensitive=True,
        project_action=False,
    )
    state.search_tools_enabled = True
    call = ToolCall(
        stream_id="call_fetch",
        index=0,
        id="call_fetch",
        name="fetch_url",
        arguments={"url": "https://example.com/tiny"},
    )

    orchestrator.record_tool_effects(
        state,
        call,
        {
            "ok": True,
            "data": {"url": "https://example.com/tiny", "final_url": "https://example.com/tiny", "usable_text": False},
            "error": None,
            "meta": {},
        },
    )

    assert state.completion.search_has_success is True
    assert state.completion.search_has_fetch_content is False
    assert "https://example.com/tiny" in state.completion.fetched_urls


def test_policy_retrieval_injects_time_sensitive_context(tmp_path: Path) -> None:
    db_path = tmp_path / "retrieval.sqlite"
    SQLiteRetrievalStore(db_path).upsert_record(
        record_type="web_page",
        source="https://example.com/status",
        canonical_source="https://example.com/status",
        title="Status update",
        text="Current release status says alpha is available today.",
    )
    _runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="What is the current alpha release status?",
        time_sensitive=True,
        project_action=False,
    )
    _reload_retrieval(orchestrator, store_path=str(db_path), pre_context_top_k=2)
    events = cast(Any, [])

    orchestrator.inject_policy_retrieval_context(state, on_event=events.append)
    system_content = orchestrator.prompt_renderer.compose_system_content(state.selected, state.ctx)

    assert state.ctx.retrieval_hits
    assert "Retrieved context:" in system_content
    assert "Status update" in system_content
    assert any("Retrieved 1 local context" in str(event.get("text", "")) for event in events)


def test_policy_retrieval_skips_non_time_sensitive_turns(tmp_path: Path) -> None:
    db_path = tmp_path / "retrieval.sqlite"
    SQLiteRetrievalStore(db_path).upsert_record(
        record_type="memory_fact",
        source="memory:1",
        canonical_source="memory:1",
        title="preference",
        text="User prefers compact answers.",
    )
    _runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="Say hello",
        time_sensitive=False,
        project_action=False,
    )
    _reload_retrieval(orchestrator, store_path=str(db_path), pre_context_top_k=2)

    orchestrator.inject_policy_retrieval_context(state)

    assert state.ctx.retrieval_hits == []


def test_auto_memory_capture_stores_safe_preference(tmp_path: Path) -> None:
    db_path = tmp_path / "retrieval.sqlite"
    runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="I prefer concise answers for this project.",
        time_sensitive=False,
        project_action=False,
    )
    _reload_retrieval(orchestrator, store_path=str(db_path))

    orchestrator.maybe_auto_capture_memory(
        state,
        AgentTurnResult(status="done", content="Noted.", reasoning="", skill_exchanges=[]),
    )

    memories = runtime.memory.list_recent(5)
    assert memories[0]["type"] == "preference"
    assert "prefer concise answers" in memories[0]["text"]
    hits = SQLiteRetrievalStore(db_path).search("concise answers", top_k=1)
    assert hits and hits[0]["record_type"] == "memory_fact"


def test_auto_memory_capture_respects_disabled_retrieval(tmp_path: Path) -> None:
    db_path = tmp_path / "retrieval.sqlite"
    runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="I prefer concise answers for this project.",
        time_sensitive=False,
        project_action=False,
    )
    _reload_retrieval(orchestrator, enabled=False, store_path=str(db_path))

    orchestrator.maybe_auto_capture_memory(
        state,
        AgentTurnResult(status="done", content="Noted.", reasoning="", skill_exchanges=[]),
    )

    assert runtime.memory.list_recent(5)
    assert not db_path.exists()


def test_auto_memory_capture_ignores_secret_like_text(tmp_path: Path) -> None:
    runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="I prefer api key abc123 for tests.",
        time_sensitive=False,
        project_action=False,
    )

    orchestrator.maybe_auto_capture_memory(
        state,
        AgentTurnResult(status="done", content="Ok.", reasoning="", skill_exchanges=[]),
    )

    assert runtime.memory.list_recent(5) == []


def test_successful_tool_outcome_is_indexed_for_retrieval(tmp_path: Path) -> None:
    db_path = tmp_path / "retrieval.sqlite"
    _runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="read README",
        time_sensitive=False,
        project_action=False,
    )
    _reload_retrieval(orchestrator, store_path=str(db_path))
    call = ToolCall(stream_id="call_1", index=0, id="call_1", name="read_file", arguments={"filepath": "README.md"})

    orchestrator.record_tool_effects(
        state,
        call,
        {"ok": True, "data": {"stdout": "all tests passed"}, "error": None, "meta": {}},
    )

    hits = SQLiteRetrievalStore(db_path).search("tests passed", top_k=1, sources=["tool_outcome"])
    assert hits
    assert hits[0]["record_type"] == "tool_outcome"


def test_failed_tool_outcome_is_not_indexed(tmp_path: Path) -> None:
    db_path = tmp_path / "retrieval.sqlite"
    _runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="read README",
        time_sensitive=False,
        project_action=False,
    )
    _reload_retrieval(orchestrator, store_path=str(db_path))
    call = ToolCall(stream_id="call_1", index=0, id="call_1", name="read_file", arguments={"filepath": "README.md"})

    orchestrator.record_tool_effects(
        state,
        call,
        {"ok": False, "data": None, "error": {"message": "failed"}, "meta": {}},
    )

    assert SQLiteRetrievalStore(db_path).search("failed", top_k=1, sources=["tool_outcome"]) == []


def test_project_action_coercion_preserves_clarification_reply(mocker, tmp_path: Path) -> None:
    _runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="edit the config",
        time_sensitive=False,
        project_action=True,
    )
    mocker.patch.object(orchestrator, "project_action_outcome", return_value="needs_clarification")
    result = AgentTurnResult(
        status="done",
        content="Which config file should I edit?",
        reasoning="",
        skill_exchanges=[],
    )

    coerced = orchestrator.coerce_project_action_failure(state, result, stop_event=None, pass_id="pass_1")

    assert coerced is result


def test_project_action_coercion_preserves_declined_or_blocked_reply(mocker, tmp_path: Path) -> None:
    _runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="delete all project files",
        time_sensitive=False,
        project_action=True,
    )
    mocker.patch.object(orchestrator, "project_action_outcome", return_value="declined_or_blocked")
    result = AgentTurnResult(
        status="done",
        content="No project tool actually ran because the requested operation was blocked.",
        reasoning="",
        skill_exchanges=[],
    )

    coerced = orchestrator.coerce_project_action_failure(state, result, stop_event=None, pass_id="pass_1")

    assert coerced is result


def test_retrieval_replacement_and_forget_cleanup_embeddings(tmp_path: Path) -> None:
    db_path = tmp_path / "retrieval.sqlite"
    store = SQLiteRetrievalStore(db_path)
    first = store.upsert_record(
        record_type="web_page",
        source="https://example.com/first",
        canonical_source="https://example.com/doc",
        title="First",
        text="alpha release notes " * 100,
    )
    assert first is not None
    for chunk in store.chunk_texts_for_record(first.id):
        store.set_chunk_embedding(chunk_id=int(chunk["chunk_id"]), model="test", vector=[1.0, 0.0])
    assert store.stats()["embeddings"] > 0

    second = store.upsert_record(
        record_type="web_page",
        source="https://example.com/second",
        canonical_source="https://example.com/doc",
        title="Second",
        text="beta release notes " * 20,
    )

    assert second is not None
    assert second.id == first.id
    assert store.stats()["embeddings"] == 0
    for chunk in store.chunk_texts_for_record(second.id):
        store.set_chunk_embedding(chunk_id=int(chunk["chunk_id"]), model="test", vector=[0.0, 1.0])
    assert store.forget(second.id) is True
    assert store.stats()["records"] == 0
    assert store.stats()["chunks"] == 0
    assert store.stats()["embeddings"] == 0


def test_retrieval_open_cleans_existing_orphans(tmp_path: Path) -> None:
    db_path = tmp_path / "retrieval.sqlite"
    store = SQLiteRetrievalStore(db_path)
    record = store.upsert_record(
        record_type="web_page",
        source="https://example.com/orphan",
        canonical_source="https://example.com/orphan",
        title="Orphan",
        text="orphan cleanup release notes " * 80,
    )
    assert record is not None
    for chunk in store.chunk_texts_for_record(record.id):
        store.set_chunk_embedding(chunk_id=int(chunk["chunk_id"]), model="test", vector=[1.0])

    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM records WHERE id = ?", (record.id,))
    raw_store = SQLiteRetrievalStore(db_path)

    assert raw_store.stats()["records"] == 0
    assert raw_store.stats()["chunks"] == 0
    assert raw_store.stats()["embeddings"] == 0
    assert raw_store.search("orphan cleanup", top_k=1) == []


def test_plan_mode_blocks_mutating_tool_calls(mocker, tmp_path: Path) -> None:
    runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="create src folder",
        time_sensitive=False,
        project_action=True,
    )
    state.collaboration_mode = "plan"
    execute_call = mocker.patch.object(runtime, "execute_tool_call")
    stream_result = type(
        "R",
        (),
        {
            "tool_calls": [
                ToolCall(
                    stream_id="call_1",
                    index=0,
                    id="call_1",
                    name="create_directory",
                    arguments={"path": "src"},
                )
            ]
        },
    )()

    action, result = orchestrator.tool_loop.execute_tool_calls(
        system_content="system",
        state=state,
        pass_id="pass_1",
        stream_result=stream_result,
    )

    assert action == "continue"
    assert result is None
    execute_call.assert_not_called()
    assert state.completion.tool_counts.get("create_directory") == 1
    assert state.evidence
    blocked = state.evidence[-1]
    assert blocked.name == "create_directory"
    assert blocked.policy_blocked is True


def test_plan_mode_allows_project_tree_read_only_tool(tmp_path: Path) -> None:
    runtime, _classifier, orchestrator = _orchestrator_runtime(tmp_path)
    cast(Any, runtime).tool_registration = lambda name: (
        type("Reg", (), {"capability": "project_tree"})() if name == "project_tree" else None
    )
    cast(Any, runtime).tool_is_mutating = lambda _name: True

    assert orchestrator._tool_allowed_in_plan_mode("project_tree") is True


def test_finalize_response_skips_project_action_enforcement_in_plan_mode(tmp_path: Path) -> None:
    _runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="delete temp files",
        time_sensitive=False,
        project_action=True,
    )
    state.collaboration_mode = "plan"
    stream_result = type(
        "R",
        (),
        {
            "content": "Here is a step-by-step implementation plan.",
            "finish_reason": "stop",
        },
    )()

    action, result = orchestrator.finalization_engine.finalize_response(
        system_content="system",
        state=state,
        pass_id="pass_1",
        stream_result=stream_result,
    )

    assert action == "result"
    assert result is not None
    assert result.status == "done"
    assert "implementation plan" in result.content.lower()
    assert state.forced_action_retry is False


def test_finalize_response_accepts_read_only_shell_version_evidence(tmp_path: Path) -> None:
    runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="what go version am i running?",
        time_sensitive=False,
        project_action=True,
    )
    original_tool_registration = runtime.tool_registration
    shell_registration = SimpleNamespace(capability="run_shell_command", actions=("run",), mutates=True)
    cast(Any, runtime).tool_registration = lambda name: shell_registration if name == "shell_command" else original_tool_registration(name)
    orchestrator.classifier.llm_client.enable_structured_classification = False
    call = ToolCall(stream_id="call_1", index=0, id="call_1", name="shell_command", arguments={"command": "go version"})
    orchestrator.record_tool_effects(
        state,
        call,
        {
            "ok": True,
            "data": {
                "command": "go version",
                "stdout": "go version go1.26.4 darwin/arm64\n",
                "stderr": "",
                "returncode": 0,
            },
            "error": None,
            "meta": {"project_changed": False},
        },
    )
    stream_result = type(
        "R",
        (),
        {
            "content": "You are running Go version `go1.26.4` on `darwin/arm64`.",
            "finish_reason": "stop",
        },
    )()

    action, result = orchestrator.finalization_engine.finalize_response(
        system_content="system",
        state=state,
        pass_id="pass_1",
        stream_result=stream_result,
    )

    assert action == "result"
    assert result is not None
    assert result.status == "done"
    assert result.content == "You are running Go version `go1.26.4` on `darwin/arm64`."
    assert state.forced_action_retry is False


def test_finalize_response_preserves_shell_rejection_explanation(tmp_path: Path) -> None:
    _runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="what go version am i using?",
        time_sensitive=False,
        project_action=True,
    )
    orchestrator.classifier.llm_client.enable_structured_classification = False
    call = ToolCall(stream_id="call_1", index=0, id="call_1", name="shell_command", arguments={"command": "go version"})
    orchestrator.record_tool_effects(
        state,
        call,
        {
            "ok": False,
            "data": None,
            "error": {"code": "E_POLICY", "message": "Shell command rejected by user"},
            "meta": {"duration_ms": 60008},
        },
        policy_blocked=True,
    )
    stream_result = type(
        "R",
        (),
        {
            "content": "I attempted to run `go version` to check your Go version, but the command was rejected.",
            "finish_reason": "stop",
        },
    )()

    action, result = orchestrator.finalization_engine.finalize_response(
        system_content="system",
        state=state,
        pass_id="pass_1",
        stream_result=stream_result,
    )

    assert action == "result"
    assert result is not None
    assert result.status == "done"
    assert "command was rejected" in result.content
    assert result.error is None
    assert state.forced_action_retry is False


def test_skill_runtime_only_exposes_custom_tools_after_skill_view_load(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    skill_ctx = SkillContext(
        user_input="create",
        branch_labels=[],
        attachments=[],
        project_root=str(runtime.project.project_root),
        memory_hits=[],
    )

    selected_before = runtime.select_skills(skill_ctx)
    names_before = set(runtime.allowed_tool_names(selected_before, ctx=skill_ctx))
    assert "create_directory" not in names_before

    load_result = runtime.skill_view("project-ops", "", skill_ctx)
    assert load_result.get("loaded") is True

    selected_after = runtime.select_skills(skill_ctx)
    names_after = set(runtime.allowed_tool_names(selected_after, ctx=skill_ctx))
    assert "create_directory" in names_after


def test_llm_client_call_with_retry_delegates_to_provider_stream(mocker) -> None:
    llm_client = LLMClient({"agent": {}})
    expected = StreamPassResult(finish_reason="stop", content="ok")
    mocker.patch.object(llm_client, "_status_allows_immediate_send", return_value=ModelStatus(state="online"))
    stream = mocker.patch.object(llm_client, "stream_completion", return_value=expected)

    result = llm_client.call_with_retry({"messages": []}, stop_event=None, on_event=None, pass_id="pass_1")

    assert result == expected
    stream.assert_called_once()


def test_llm_client_call_with_retry_retries_once_on_retryable_failure(mocker) -> None:
    llm_client = LLMClient({"agent": {}})
    llm_client.per_turn_retries = 1
    llm_client.retry_backoff_s = 0.0
    events = cast(Any, [])
    mocker.patch.object(llm_client, "_status_allows_immediate_send", return_value=ModelStatus(state="online"))
    mocker.patch.object(llm_client, "_should_retry_exception", return_value=True)
    mocker.patch.object(llm_client, "refresh_model_status", return_value=ModelStatus(state="online"))
    mocker.patch.object(llm_client, "check_ready", return_value=True)
    stream = mocker.patch.object(
        llm_client,
        "stream_completion",
        side_effect=[
            RuntimeError("temporary"),
            StreamPassResult(finish_reason="stop", content="ok"),
        ],
    )

    result = llm_client.call_with_retry({"messages": []}, stop_event=None, on_event=events.append, pass_id="pass_1")

    assert result.finish_reason == "stop"
    assert result.content == "ok"
    assert stream.call_count == 2
    assert any(event.get("type") == "info" and "Retrying request (1/1)" in str(event.get("text", "")) for event in events)


def test_llm_client_call_with_retry_injects_status_model_when_payload_omits_model(mocker) -> None:
    llm_client = LLMClient({"agent": {}})
    expected = StreamPassResult(finish_reason="stop", content="ok")
    mocker.patch.object(
        llm_client,
        "_status_allows_immediate_send",
        return_value=ModelStatus(state="online", model_name="mlx-community/Qwen2.5-VL-7B-Instruct-4bit"),
    )
    stream = mocker.patch.object(llm_client, "stream_completion", return_value=expected)

    result = llm_client.call_with_retry({"messages": []}, stop_event=None, on_event=None, pass_id="pass_1")

    assert result == expected
    sent_payload = stream.call_args.args[0]
    assert sent_payload["model"] == "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"


def test_llm_client_call_with_retry_keeps_explicit_model_override(mocker) -> None:
    llm_client = LLMClient({"agent": {}})
    expected = StreamPassResult(finish_reason="stop", content="ok")
    mocker.patch.object(
        llm_client,
        "_status_allows_immediate_send",
        return_value=ModelStatus(state="online", model_name="mlx-community/Qwen2.5-VL-7B-Instruct-4bit"),
    )
    stream = mocker.patch.object(llm_client, "stream_completion", return_value=expected)

    result = llm_client.call_with_retry({"messages": [], "model": "custom-model"}, stop_event=None, on_event=None, pass_id="pass_1")

    assert result == expected
    sent_payload = stream.call_args.args[0]
    assert sent_payload["model"] == "custom-model"


def test_llm_client_build_payload_uses_responses_shape_when_forced() -> None:
    llm_client = LLMClient({"agent": {"endpoint_mode": "responses"}})

    payload = cast(
        Any,
        llm_client.build_payload(
            model_messages=[{"role": "user", "content": "hi"}],
            thinking=True,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "list_files",
                        "description": "List files",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        ),
    )

    assert "input" in payload
    assert "messages" not in payload
    assert payload["stream"] is True
    assert payload["tools"][0]["name"] == "list_files"
    assert payload["tool_choice"] == "auto"


def test_llm_client_auto_mode_falls_back_from_responses_to_chat(mocker) -> None:
    llm_client = LLMClient({"agent": {"endpoint_mode": "auto", "per_turn_retries": 0}})
    calls: list[str] = []
    payload = llm_client.build_payload(model_messages=[{"role": "user", "content": "hello"}], thinking=True)
    assert "input" in payload

    def fake_stream(
        *,
        endpoint,
        payload,
        timeout_s,
        headers,
        ssl_context,
        stop_event,
        on_debug_event,
    ):
        calls.append(endpoint)
        if endpoint.endswith("/v1/responses"):
            raise StreamError("HTTP 404: Not Found", status_code=404, retryable=False)
        yield {"choices": [{"delta": {"content": "ok"}}]}
        yield {"choices": [{"finish_reason": "stop"}]}

    mocker.patch.object(llm_client, "_status_allows_immediate_send", return_value=ModelStatus(state="online"))
    llm_client._stream_chat_completions = fake_stream

    result = llm_client.call_with_retry(payload, stop_event=None, on_event=None, pass_id="pass_1")

    assert result.finish_reason == "stop"
    assert result.content == "ok"
    assert calls[0].endswith("/v1/responses")
    assert calls[1].endswith("/v1/chat/completions")


def test_llm_client_detects_backend_profile_from_models_payload() -> None:
    llm_client = LLMClient({"agent": {"backend_profile": "auto"}})

    llm_client._refresh_backend_profile({"data": [{"id": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"}]})
    info = llm_client.backend_profile_info()

    assert info["requested"] == "auto"
    assert info["detected"] == "mlx_vlm"
    assert info["selected"] == "mlx_vlm"


def test_llm_client_rewrites_mlx_vlm_multimodal_payload(mocker) -> None:
    llm_client = LLMClient({"agent": {"backend_profile": "mlx_vlm"}})
    payload = llm_client.build_payload(
        model_messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZQ=="}},
                ],
            }
        ],
        thinking=True,
    )
    expected = StreamPassResult(finish_reason="stop", content="ok")
    mocker.patch.object(
        llm_client,
        "_status_allows_immediate_send",
        return_value=ModelStatus(state="online", model_name="mlx-community/Qwen2.5-VL-7B-Instruct-4bit"),
    )
    stream = mocker.patch.object(llm_client, "stream_completion", return_value=expected)

    result = llm_client.call_with_retry(payload, stop_event=None, on_event=None, pass_id="pass_1")

    assert result == expected
    sent = stream.call_args.args[0]
    assert "chat_template_kwargs" not in sent
    assert "stream_options" not in sent
    assert sent["messages"][0]["content"][1]["image_url"] == "data:image/png;base64,ZmFrZQ=="


def test_llm_client_preserves_disabled_thinking_for_template_backends(mocker) -> None:
    llm_client = LLMClient({"agent": {"backend_profile": "mlx_vlm"}})
    payload = llm_client.build_payload(
        model_messages=[{"role": "user", "content": "hello"}],
        thinking=False,
    )
    expected = StreamPassResult(finish_reason="stop", content="ok")
    mocker.patch.object(
        llm_client,
        "_status_allows_immediate_send",
        return_value=ModelStatus(state="online", model_name="mlx-community/Qwen2.5-VL-7B-Instruct-4bit"),
    )
    stream = mocker.patch.object(llm_client, "stream_completion", return_value=expected)

    result = llm_client.call_with_retry(payload, stop_event=None, on_event=None, pass_id="pass_1")

    assert result == expected
    sent = stream.call_args.args[0]
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert "stream_options" not in sent


def test_llm_client_fails_fast_on_local_backend_model_mismatch(mocker) -> None:
    llm_client = LLMClient({"agent": {"backend_profile": "llamacpp"}})
    mocker.patch.object(
        llm_client,
        "_status_allows_immediate_send",
        return_value=ModelStatus(state="online", model_name="qwen-3"),
    )
    mocker.patch.object(llm_client, "stream_completion")

    try:
        llm_client.call_with_retry(
            {"messages": [{"role": "user", "content": "hello"}], "model": "llava-1.5b"},
            stop_event=None,
            on_event=None,
            pass_id="pass_1",
        )
        raise AssertionError("Expected model integrity mismatch error")
    except RuntimeError as exc:
        assert "Backend model mismatch" in str(exc)
