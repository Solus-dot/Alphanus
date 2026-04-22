from __future__ import annotations

import json
from pathlib import Path

from agent.classifier import TurnClassifier
from agent.context import ContextWindowManager
from agent.llm_client import LLMClient
from agent.orchestrator import TurnOrchestrator
from agent.policies import PromptPolicyRenderer
from agent.telemetry import TelemetryEmitter, configure_logging
from core.memory import VectorMemory
from core.skills import SkillContext, SkillRuntime
from core.types import ModelStatus, StreamPassResult, ToolCall, TurnClassification
from core.workspace import WorkspaceManager


def _runtime(tmp_path: Path) -> SkillRuntime:
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
  }
}

def execute(tool_name, args, env):
    return {"ok": True, "data": {"filepath": env.workspace.create_directory(args["path"])}, "error": None, "meta": {}}
""".strip(),
        encoding="utf-8",
    )
    return SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )


def test_telemetry_emits_json_lines_to_configured_log_file(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    logger = configure_logging({"logging": {"format": "json", "path": str(log_path), "level": "INFO"}})

    TelemetryEmitter(logger).emit("turn_classified", source="model", time_sensitive=True)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "turn_classified"
    assert payload["payload"]["source"] == "model"
    assert payload["payload"]["time_sensitive"] is True


def test_console_logging_suppresses_info_telemetry_but_keeps_file_events(tmp_path: Path, capsys) -> None:
    log_path = tmp_path / "events.jsonl"
    logger = configure_logging({"logging": {"format": "plain", "path": str(log_path), "level": "INFO"}})

    TelemetryEmitter(logger).emit("http_stream", phase="start")
    logger.warning("visible warning")

    captured = capsys.readouterr()
    assert "http_stream" not in captured.err
    assert "visible warning" in captured.err

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2
    events = [json.loads(line) for line in lines]
    assert any(item.get("event") == "http_stream" for item in events)

def test_classifier_uses_model_for_local_workspace_task(mocker, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": True}}
    llm_client = LLMClient(cfg)
    classifier = TurnClassifier(cfg, runtime, llm_client)
    ctx = classifier.build_skill_context(
        "Make a landing page for a bakery using html css and javascript and save it in a folder called arjun",
        [],
        [],
        [],
    )

    mocker.patch.object(TurnClassifier, "_should_model_classify", return_value=True)

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        assert pass_id == "turn_classify"
        return type(
            "R",
            (),
            {
                "finish_reason": "stop",
                "content": json.dumps(
                    {
                        "prefer_local_workspace_tools": True,
                    }
                ),
            },
        )()

    llm_client.call_with_retry = fake_call_with_retry
    classifier.call_with_retry = fake_call_with_retry

    classification = classifier.classify(ctx)

    assert classification.used_model is True
    assert classification.prefer_local_workspace_tools is True


def test_classifier_seed_keeps_explicit_external_path_without_model(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": True}}
    llm_client = LLMClient(cfg)
    classifier = TurnClassifier(cfg, runtime, llm_client)
    ctx = classifier.build_skill_context(
        "Read /tmp/proposal.docx",
        [],
        [],
        [],
    )

    classification = classifier.classify(ctx)

    assert classification.used_model is False
    assert classification.explicit_external_path == str(Path("/tmp/proposal.docx").resolve(strict=False))


def test_classifier_seed_keeps_time_sensitive_flag_without_model(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": True}}
    llm_client = LLMClient(cfg)
    classifier = TurnClassifier(cfg, runtime, llm_client)
    ctx = classifier.build_skill_context(
        "latest OpenAI news",
        [],
        [],
        [],
    )

    classification = classifier.classify(ctx)

    assert classification.used_model is False
    assert classification.time_sensitive is False


def test_classifier_seed_does_not_infer_workspace_flags_without_model(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": True}}
    llm_client = LLMClient(cfg)
    classifier = TurnClassifier(cfg, runtime, llm_client)
    history_messages = [
        {"role": "user", "content": "delete all files in the workspace"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "create_file",
                        "arguments": '{"filepath":"a.txt","content":"alpha"}',
                    }
                }
            ],
        },
        {"role": "tool", "name": "create_file", "content": '{"ok": true, "data": {"filepath": "a.txt"}}'},
    ]
    ctx = classifier.build_skill_context("yes", [], [], history_messages)

    classification = classifier.classify(ctx)

    assert classification.used_model is False
    assert classification.requires_workspace_action is False
    assert classification.prefer_local_workspace_tools is False
    assert classification.followup_kind == "new_request"


def test_classifier_seed_does_not_infer_contextual_followup_without_model(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": True}}
    llm_client = LLMClient(cfg)
    classifier = TurnClassifier(cfg, runtime, llm_client)
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
    ctx = classifier.build_skill_context("Where is JS?", [], [], history_messages)

    classification = classifier.classify(ctx)

    assert classification.used_model is False
    assert classification.followup_kind == "new_request"


def test_classifier_uses_model_for_contextual_followup(mocker, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": True}}
    llm_client = LLMClient(cfg)
    classifier = TurnClassifier(cfg, runtime, llm_client)
    mocker.patch.object(TurnClassifier, "_should_model_classify", return_value=True)
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
    ctx = classifier.build_skill_context("Where is JS?", [], [], history_messages)
    calls = []

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        calls.append(pass_id)
        return type(
            "R",
            (),
            {
                "finish_reason": "stop",
                "content": json.dumps({"followup_kind": "contextual_followup"}),
            },
        )()

    llm_client.call_with_retry = fake_call_with_retry
    classifier.call_with_retry = fake_call_with_retry

    classification = classifier.classify(ctx)

    assert classification.used_model is True
    assert classification.followup_kind == "contextual_followup"
    assert "turn_classify" in calls


def test_workspace_action_outcome_skips_model_when_structured_classification_disabled(mocker, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    llm_client = LLMClient(cfg)
    classifier = TurnClassifier(cfg, runtime, llm_client)
    call = mocker.patch.object(classifier, "call_with_retry")

    outcome = classifier.classify_workspace_action_outcome(
        current_user_input="yes",
        recent_routing_hint="",
        assistant_reply="shell_command is not allowed for local workspace file tasks; use workspace tools instead.",
        evidence={
            "has_successful_mutation": False,
            "policy_blocked_tools": ["shell_command"],
            "recent_tools": [
                {
                    "name": "shell_command",
                    "ok": False,
                    "mutating": False,
                    "policy_blocked": True,
                    "error_code": "E_POLICY",
                    "error_message": "shell_command is not allowed",
                }
            ],
        },
        pass_id="pass_1",
    )

    assert outcome == "declined_or_blocked"
    call.assert_not_called()


def test_workspace_action_outcome_falls_back_to_blocked_when_classifier_call_fails(mocker, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": True}}
    llm_client = LLMClient(cfg)
    classifier = TurnClassifier(cfg, runtime, llm_client)
    mocker.patch.object(classifier, "call_with_retry", side_effect=RuntimeError("timeout"))

    outcome = classifier.classify_workspace_action_outcome(
        current_user_input="yes",
        recent_routing_hint="",
        assistant_reply="shell_command is not allowed for local workspace file tasks; use workspace tools instead.",
        evidence={
            "has_successful_mutation": False,
            "policy_blocked_tools": ["shell_command"],
            "recent_tools": [
                {
                    "name": "shell_command",
                    "ok": False,
                    "mutating": False,
                    "policy_blocked": True,
                    "error_code": "E_POLICY",
                    "error_message": "shell_command is not allowed",
                }
            ],
        },
        pass_id="pass_1",
    )

    assert outcome == "declined_or_blocked"


def test_workspace_action_outcome_fallback_rejects_user_delegation_even_with_blocked_evidence(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    llm_client = LLMClient(cfg)
    classifier = TurnClassifier(cfg, runtime, llm_client)

    outcome = classifier.classify_workspace_action_outcome(
        current_user_input="yes",
        recent_routing_hint="",
        assistant_reply="The tool is blocked here. Please delete the files yourself.",
        evidence={
            "has_successful_mutation": False,
            "policy_blocked_tools": ["shell_command"],
            "recent_tools": [
                {
                    "name": "shell_command",
                    "ok": False,
                    "mutating": False,
                    "policy_blocked": True,
                    "error_code": "E_POLICY",
                    "error_message": "shell_command is not allowed",
                }
            ],
        },
        pass_id="pass_1",
    )

    assert outcome == "not_completed"


def test_workspace_action_outcome_fallback_rejects_false_success_claim_even_with_blocked_evidence(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    llm_client = LLMClient(cfg)
    classifier = TurnClassifier(cfg, runtime, llm_client)

    outcome = classifier.classify_workspace_action_outcome(
        current_user_input="yes",
        recent_routing_hint="",
        assistant_reply="shell_command is not allowed here, but I deleted the files for you.",
        evidence={
            "has_successful_mutation": False,
            "policy_blocked_tools": ["shell_command"],
            "recent_tools": [
                {
                    "name": "shell_command",
                    "ok": False,
                    "mutating": False,
                    "policy_blocked": True,
                    "error_code": "E_POLICY",
                    "error_message": "shell_command is not allowed",
                }
            ],
        },
        pass_id="pass_1",
    )

    assert outcome == "not_completed"


def test_classifier_clears_local_workspace_preference_for_environment_question(mocker, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": True}}
    llm_client = LLMClient(cfg)
    classifier = TurnClassifier(cfg, runtime, llm_client)
    ctx = classifier.build_skill_context(
        "What version of go am I using?",
        [],
        [],
        [],
    )

    mocker.patch.object(TurnClassifier, "_should_model_classify", return_value=True)

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        assert pass_id == "turn_classify"
        return type(
            "R",
            (),
            {
                "finish_reason": "stop",
                "content": json.dumps(
                    {
                        "prefer_local_workspace_tools": True,
                    }
                ),
            },
        )()

    llm_client.call_with_retry = fake_call_with_retry
    classifier.call_with_retry = fake_call_with_retry

    classification = classifier.classify(ctx)

    assert classification.used_model is True
    assert classification.prefer_local_workspace_tools is False


def test_classifier_clears_local_workspace_preference_for_shell_confirmation_followup(mocker, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": True}}
    llm_client = LLMClient(cfg)
    classifier = TurnClassifier(cfg, runtime, llm_client)
    history_messages = [
        {"role": "user", "content": "how do i check my go version"},
        {"role": "assistant", "content": "I can run `go version` in the workspace if you want."},
    ]
    ctx = classifier.build_skill_context("yes check it", [], [], history_messages)

    mocker.patch.object(TurnClassifier, "_should_model_classify", return_value=True)

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        assert pass_id == "turn_classify"
        return type(
            "R",
            (),
            {
                "finish_reason": "stop",
                "content": json.dumps(
                    {
                        "followup_kind": "confirmation",
                        "requires_workspace_action": True,
                        "prefer_local_workspace_tools": True,
                    }
                ),
            },
        )()

    llm_client.call_with_retry = fake_call_with_retry
    classifier.call_with_retry = fake_call_with_retry

    classification = classifier.classify(ctx)

    assert classification.used_model is True
    assert classification.followup_kind == "confirmation"
    assert classification.requires_workspace_action is True
    assert classification.prefer_local_workspace_tools is False


def _orchestrator_runtime(tmp_path: Path) -> tuple[SkillRuntime, TurnClassifier, TurnOrchestrator]:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {}}
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


def _turn_state(tmp_path: Path, *, user_input: str, time_sensitive: bool, workspace_action: bool):
    runtime, classifier, orchestrator = _orchestrator_runtime(tmp_path)
    ctx = classifier.build_skill_context(user_input, [], [], [])
    classification = TurnClassification(
        time_sensitive=time_sensitive,
        requires_workspace_action=workspace_action,
        prefer_local_workspace_tools=workspace_action,
        source="fallback",
    )
    state = orchestrator.build_turn_state(ctx, [], [], classification)
    return runtime, orchestrator, state


def test_orchestrator_policy_snapshot_captures_forced_flags_and_shell_exposure(mocker, tmp_path: Path) -> None:
    runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="latest updates",
        time_sensitive=True,
        workspace_action=True,
    )
    state.search_tools_enabled = True
    state.forced_search_retry = True
    state.forced_action_retry = True
    mocker.patch.object(
        runtime,
        "allowed_tool_names",
        return_value=["shell_command", "web_search", "request_user_input"],
    )

    snapshot = orchestrator.build_policy_snapshot(state)

    assert snapshot.search_mode is True
    assert snapshot.time_sensitive_query is True
    assert snapshot.forced_search_retry is True
    assert snapshot.requires_workspace_action is True
    assert snapshot.forced_action_retry is True
    assert snapshot.shell_tool_exposed is True


def test_orchestrator_records_workspace_evidence_and_policy_blocks(tmp_path: Path) -> None:
    _runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="create a file",
        time_sensitive=False,
        workspace_action=True,
    )

    mutating_shell_call = ToolCall(
        stream_id="call_1",
        index=0,
        id="call_1",
        name="shell_command",
        arguments={"command": "touch notes.txt"},
    )
    shell_call = ToolCall(
        stream_id="call_2",
        index=1,
        id="call_2",
        name="shell_command",
        arguments={"command": "rm -rf /tmp/nope"},
    )

    orchestrator.record_tool_effects(
        state,
        mutating_shell_call,
        {
            "ok": True,
            "data": {"stdout": "", "stderr": ""},
            "error": None,
            "meta": {"workspace_changed": True},
        },
    )
    orchestrator.record_tool_effects(
        state,
        shell_call,
        {
            "ok": False,
            "data": None,
            "error": {"code": "E_POLICY", "message": "shell blocked"},
            "meta": {},
        },
        policy_blocked=True,
    )

    evidence = orchestrator.workspace_action_evidence(state)

    assert state.completion.tool_counts["shell_command"] == 2
    assert evidence["has_successful_mutation"] is True
    assert "shell_command" in evidence["successful_mutating_tools"]
    assert "shell_command" in evidence["policy_blocked_tools"]


def test_orchestrator_search_budget_reason_is_explicit_for_time_sensitive_turns(tmp_path: Path) -> None:
    _runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="latest status",
        time_sensitive=True,
        workspace_action=False,
    )
    state.search_tools_enabled = True
    state.tool_budgets["web_search"] = 1
    state.completion.tool_counts["web_search"] = 1
    web_search_call = ToolCall(
        stream_id="call_search",
        index=0,
        id="call_search",
        name="web_search",
        arguments={"query": "latest status"},
    )

    reason = orchestrator.tool_budget_reason(state, web_search_call)

    assert reason is not None
    assert "search-attempt budget is exhausted" in reason


def test_skill_runtime_only_exposes_custom_tools_after_skill_view_load(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    skill_ctx = SkillContext(
        user_input="create",
        branch_labels=[],
        attachments=[],
        workspace_root=str(runtime.workspace.workspace_root),
        memory_hits=[],
    )

    selected_before = runtime.select_skills(skill_ctx)
    names_before = set(runtime.allowed_tool_names(selected_before, ctx=skill_ctx))
    assert "create_directory" not in names_before

    load_result = runtime.skill_view("workspace-ops", "", skill_ctx)
    assert load_result.get("loaded") is True

    selected_after = runtime.select_skills(skill_ctx)
    names_after = set(runtime.allowed_tool_names(selected_after, ctx=skill_ctx))
    assert "create_directory" in names_after


def test_llm_client_call_with_retry_delegates_to_provider_stream(mocker) -> None:
    llm_client = LLMClient({"agent": {}})
    expected = StreamPassResult(finish_reason="stop", content="ok")
    mocker.patch.object(llm_client, "_status_allows_immediate_send", return_value=ModelStatus(state="online"))
    stream = mocker.patch.object(llm_client.provider, "stream_completion", return_value=expected)

    result = llm_client.call_with_retry({"messages": []}, stop_event=None, on_event=None, pass_id="pass_1")

    assert result == expected
    stream.assert_called_once()


def test_llm_client_call_with_retry_retries_once_on_retryable_failure(mocker) -> None:
    llm_client = LLMClient({"agent": {}})
    llm_client.per_turn_retries = 1
    llm_client.retry_backoff_s = 0.0
    events: list[dict[str, object]] = []
    mocker.patch.object(llm_client, "_status_allows_immediate_send", return_value=ModelStatus(state="online"))
    mocker.patch.object(llm_client, "_should_retry_exception", return_value=True)
    mocker.patch.object(llm_client, "refresh_model_status", return_value=ModelStatus(state="online"))
    mocker.patch.object(llm_client, "ensure_ready", return_value=True)
    stream = mocker.patch.object(
        llm_client.provider,
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
