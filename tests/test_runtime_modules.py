from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportIndexIssue=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportCallIssue=false
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from agent.classifier import TurnClassifier
from agent.context import ContextWindowManager
from agent.llm_client import LLMClient
from agent.orchestrator import TurnOrchestrator
from agent.policies import PromptPolicyRenderer
from agent.telemetry import TelemetryEmitter, configure_logging
from core.memory import LexicalMemory
from core.retrieval import SQLiteRetrievalStore
from core.streaming import StreamError
from core.types import AgentTurnResult, ModelStatus, StreamPassResult, ToolCall, TurnClassification
from core.workspace import WorkspaceManager
from skills.runtime import SkillContext, SkillRuntime


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
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
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


def test_malformed_auth_header_template_emits_telemetry(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "auth-events.jsonl"
    logger = configure_logging({"logging": {"format": "json", "path": str(log_path), "level": "INFO"}})
    monkeypatch.setenv("ALPHANUS_API_KEY", "secret-token")

    client = LLMClient(
        {"agent": {"api_key": "env:ALPHANUS_API_KEY", "auth_header_template": "Authorization: Bearer {api_key.missing}"}},
        telemetry=TelemetryEmitter(logger),
    )
    client.reload_config(
        {"agent": {"api_key": "env:ALPHANUS_API_KEY", "auth_header_template": "Authorization: Bearer {api_key.missing}"}}
    )

    assert client.auth_header == "Authorization: Bearer secret-token"
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert any(item.get("event") == "auth_header_template_invalid" for item in events)


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


def test_classifier_rule_path_reports_rules_source(mocker, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    llm_client = LLMClient(cfg)
    classifier = TurnClassifier(cfg, runtime, llm_client)
    call = mocker.patch.object(classifier, "call_with_retry")
    ctx = classifier.build_skill_context("hello", [], [], [])

    classification = classifier.classify(ctx)

    assert classification.source == "rules"
    assert classification.used_model is False
    call.assert_not_called()


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


def test_workspace_action_outcome_rules_rejects_user_delegation_even_with_blocked_evidence(tmp_path: Path) -> None:
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


def test_workspace_action_outcome_rules_rejects_false_success_claim_even_with_blocked_evidence(tmp_path: Path) -> None:
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
        source="rules",
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
    assert snapshot.collaboration_mode == "execute"


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
    assert evidence["has_successful_tool"] is True
    assert "shell_command" in evidence["successful_tools"]
    assert evidence["has_successful_mutation"] is True
    assert "shell_command" in evidence["successful_mutating_tools"]
    assert "shell_command" in evidence["policy_blocked_tools"]


def test_workspace_action_outcome_accepts_successful_non_mutating_open_action(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))

    outcome = classifier.classify_workspace_action_outcome(
        current_user_input="open it using the open command please",
        recent_routing_hint="",
        assistant_reply="The website has been opened in your default browser.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["shell_command"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "policy_blocked_tools": [],
            "recent_tools": [{"name": "shell_command", "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "completed_with_evidence"


def test_workspace_action_outcome_accepts_open_followup_after_mutating_hint(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))

    outcome = classifier.classify_workspace_action_outcome(
        current_user_input="open it",
        recent_routing_hint="previous user request: create notes.txt",
        assistant_reply="The website has been opened in your default browser.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["shell_command"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "policy_blocked_tools": [],
            "recent_tools": [{"name": "shell_command", "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "completed_with_evidence"


def test_workspace_action_outcome_accepts_show_request_with_list_evidence(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))

    outcome = classifier.classify_workspace_action_outcome(
        current_user_input="show me the files in the workspace",
        recent_routing_hint="",
        assistant_reply="I listed the files in the workspace.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["list_files"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "policy_blocked_tools": [],
            "recent_tools": [{"name": "list_files", "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "completed_with_evidence"


def test_workspace_action_outcome_accepts_display_tree_with_workspace_tree_evidence(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))

    outcome = classifier.classify_workspace_action_outcome(
        current_user_input="display the directory tree",
        recent_routing_hint="",
        assistant_reply="I displayed the directory tree.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["workspace_tree"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "policy_blocked_tools": [],
            "recent_tools": [{"name": "workspace_tree", "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "completed_with_evidence"


def test_workspace_action_outcome_rejects_ack_followup_to_mutating_request_without_mutation(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))

    outcome = classifier.classify_workspace_action_outcome(
        current_user_input="yes, do it",
        recent_routing_hint="previous user request: create notes.txt",
        assistant_reply="I checked notes.txt.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["read_file"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "policy_blocked_tools": [],
            "recent_tools": [{"name": "read_file", "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "not_completed"


def test_workspace_action_outcome_overrides_structured_ack_followup_without_mutation(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": True}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))
    classifier.call_with_retry = lambda *_args, **_kwargs: SimpleNamespace(content='{"outcome":"completed_with_evidence"}')

    outcome = classifier.classify_workspace_action_outcome(
        current_user_input="do it",
        recent_routing_hint="previous user request: delete old.txt",
        assistant_reply="I checked old.txt.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["read_file"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "policy_blocked_tools": [],
            "recent_tools": [{"name": "read_file", "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "not_completed"


def test_workspace_action_outcome_rejects_open_claim_with_only_read_evidence(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))

    outcome = classifier.classify_workspace_action_outcome(
        current_user_input="open the app",
        recent_routing_hint="",
        assistant_reply="I opened the app.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["read_file"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "policy_blocked_tools": [],
            "recent_tools": [{"name": "read_file", "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "not_completed"


def test_workspace_action_outcome_overrides_structured_open_claim_with_only_read_evidence(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": True}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))
    classifier.call_with_retry = lambda *_args, **_kwargs: SimpleNamespace(content='{"outcome":"completed_with_evidence"}')

    outcome = classifier.classify_workspace_action_outcome(
        current_user_input="open the app",
        recent_routing_hint="",
        assistant_reply="I opened the app.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["read_file"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "policy_blocked_tools": [],
            "recent_tools": [{"name": "read_file", "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "not_completed"


def test_workspace_action_outcome_still_rejects_file_creation_claim_without_mutation(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))

    outcome = classifier.classify_workspace_action_outcome(
        current_user_input="create notes.txt",
        recent_routing_hint="",
        assistant_reply="I created notes.txt.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["read_file"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "policy_blocked_tools": [],
            "recent_tools": [{"name": "read_file", "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "not_completed"


def test_workspace_action_outcome_rejects_non_mutating_completion_for_mutating_request(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))

    outcome = classifier.classify_workspace_action_outcome(
        current_user_input="create notes.txt",
        recent_routing_hint="",
        assistant_reply="I checked notes.txt.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["read_file"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "policy_blocked_tools": [],
            "recent_tools": [{"name": "read_file", "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "not_completed"


def test_workspace_action_outcome_overrides_structured_completion_without_mutation(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": True}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))
    classifier.call_with_retry = lambda *_args, **_kwargs: SimpleNamespace(content='{"outcome":"completed_with_evidence"}')

    outcome = classifier.classify_workspace_action_outcome(
        current_user_input="create notes.txt",
        recent_routing_hint="",
        assistant_reply="I checked notes.txt.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["read_file"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "policy_blocked_tools": [],
            "recent_tools": [{"name": "read_file", "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "not_completed"


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


def test_search_tool_effects_record_attempt_metadata_and_empty_evidence(tmp_path: Path) -> None:
    _runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="latest status",
        time_sensitive=True,
        workspace_action=False,
    )
    state.search_tools_enabled = True
    call = ToolCall(
        stream_id="call_search",
        index=0,
        id="call_search",
        name="web_search",
        arguments={"query": "latest status"},
    )

    orchestrator.record_tool_effects(
        state,
        call,
        {
            "ok": True,
            "data": {
                "results": [],
                "attempts": [{"provider": "searxng", "status": "error", "failure_class": "network"}],
                "evidence_quality": "none",
            },
            "error": None,
            "meta": {},
        },
    )

    assert state.completion.search_failure_count == 1
    assert state.completion.search_has_success is False
    assert state.completion.search_attempts[0]["provider"] == "searxng"
    assert state.completion.search_failure_classes == ["network"]


def test_fetch_tool_effects_do_not_count_unusable_text_as_fetch_evidence(tmp_path: Path) -> None:
    _runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="latest status",
        time_sensitive=True,
        workspace_action=False,
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
        workspace_action=False,
    )
    orchestrator.config["retrieval"] = {"store_path": str(db_path), "pre_context_top_k": 2}
    events: list[dict[str, object]] = []

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
        workspace_action=False,
    )
    orchestrator.config["retrieval"] = {"store_path": str(db_path), "pre_context_top_k": 2}

    orchestrator.inject_policy_retrieval_context(state)

    assert state.ctx.retrieval_hits == []


def test_auto_memory_capture_stores_safe_preference(tmp_path: Path) -> None:
    db_path = tmp_path / "retrieval.sqlite"
    runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="I prefer concise answers for this project.",
        time_sensitive=False,
        workspace_action=False,
    )
    orchestrator.config["retrieval"] = {"store_path": str(db_path)}

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
        workspace_action=False,
    )
    orchestrator.config["retrieval"] = {"enabled": False, "store_path": str(db_path)}

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
        workspace_action=False,
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
        user_input="run checks",
        time_sensitive=False,
        workspace_action=False,
    )
    orchestrator.config["retrieval"] = {"store_path": str(db_path)}
    call = ToolCall(stream_id="call_1", index=0, id="call_1", name="run_checks", arguments={"command": "pytest"})

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
        user_input="run checks",
        time_sensitive=False,
        workspace_action=False,
    )
    orchestrator.config["retrieval"] = {"store_path": str(db_path)}
    call = ToolCall(stream_id="call_1", index=0, id="call_1", name="run_checks", arguments={"command": "pytest"})

    orchestrator.record_tool_effects(
        state,
        call,
        {"ok": False, "data": None, "error": {"message": "failed"}, "meta": {}},
    )

    assert SQLiteRetrievalStore(db_path).search("failed", top_k=1, sources=["tool_outcome"]) == []


def test_workspace_action_coercion_preserves_clarification_reply(mocker, tmp_path: Path) -> None:
    _runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="edit the config",
        time_sensitive=False,
        workspace_action=True,
    )
    mocker.patch.object(orchestrator, "workspace_action_outcome", return_value="needs_clarification")
    result = AgentTurnResult(
        status="done",
        content="Which config file should I edit?",
        reasoning="",
        skill_exchanges=[],
    )

    coerced = orchestrator.coerce_workspace_action_failure(state, result, stop_event=None, pass_id="pass_1")

    assert coerced is result


def test_workspace_action_coercion_preserves_declined_or_blocked_reply(mocker, tmp_path: Path) -> None:
    _runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="delete all workspace files",
        time_sensitive=False,
        workspace_action=True,
    )
    mocker.patch.object(orchestrator, "workspace_action_outcome", return_value="declined_or_blocked")
    result = AgentTurnResult(
        status="done",
        content="No workspace tool actually ran because the requested operation was blocked.",
        reasoning="",
        skill_exchanges=[],
    )

    coerced = orchestrator.coerce_workspace_action_failure(state, result, stop_event=None, pass_id="pass_1")

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
        workspace_action=True,
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

    action, result = orchestrator.execute_tool_calls(
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


def test_plan_mode_allows_workspace_tree_read_only_tool(tmp_path: Path) -> None:
    runtime, _classifier, orchestrator = _orchestrator_runtime(tmp_path)
    runtime.tool_registration = lambda name: type("Reg", (), {"capability": "workspace_tree"})() if name == "workspace_tree" else None
    runtime.tool_is_mutating = lambda _name: True

    assert orchestrator._tool_allowed_in_plan_mode("workspace_tree") is True


def test_finalize_response_skips_workspace_action_enforcement_in_plan_mode(tmp_path: Path) -> None:
    _runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="delete temp files",
        time_sensitive=False,
        workspace_action=True,
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

    action, result = orchestrator.finalize_response(
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
    mocker.patch.object(llm_client.provider, "_status_allows_immediate_send", return_value=ModelStatus(state="online"))
    stream = mocker.patch.object(llm_client.provider, "stream_completion", return_value=expected)

    result = llm_client.call_with_retry({"messages": []}, stop_event=None, on_event=None, pass_id="pass_1")

    assert result == expected
    stream.assert_called_once()


def test_llm_client_call_with_retry_retries_once_on_retryable_failure(mocker) -> None:
    llm_client = LLMClient({"agent": {}})
    llm_client.provider.per_turn_retries = 1
    llm_client.provider.retry_backoff_s = 0.0
    events: list[dict[str, object]] = []
    mocker.patch.object(llm_client.provider, "_status_allows_immediate_send", return_value=ModelStatus(state="online"))
    mocker.patch.object(llm_client.provider, "_should_retry_exception", return_value=True)
    mocker.patch.object(llm_client.provider, "refresh_model_status", return_value=ModelStatus(state="online"))
    mocker.patch.object(llm_client.provider, "check_ready", return_value=True)
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


def test_llm_client_call_with_retry_injects_status_model_when_payload_omits_model(mocker) -> None:
    llm_client = LLMClient({"agent": {}})
    expected = StreamPassResult(finish_reason="stop", content="ok")
    mocker.patch.object(
        llm_client.provider,
        "_status_allows_immediate_send",
        return_value=ModelStatus(state="online", model_name="mlx-community/Qwen2.5-VL-7B-Instruct-4bit"),
    )
    stream = mocker.patch.object(llm_client.provider, "stream_completion", return_value=expected)

    result = llm_client.call_with_retry({"messages": []}, stop_event=None, on_event=None, pass_id="pass_1")

    assert result == expected
    sent_payload = stream.call_args.args[0]
    assert sent_payload["model"] == "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"


def test_llm_client_call_with_retry_keeps_explicit_model_override(mocker) -> None:
    llm_client = LLMClient({"agent": {}})
    expected = StreamPassResult(finish_reason="stop", content="ok")
    mocker.patch.object(
        llm_client.provider,
        "_status_allows_immediate_send",
        return_value=ModelStatus(state="online", model_name="mlx-community/Qwen2.5-VL-7B-Instruct-4bit"),
    )
    stream = mocker.patch.object(llm_client.provider, "stream_completion", return_value=expected)

    result = llm_client.call_with_retry({"messages": [], "model": "custom-model"}, stop_event=None, on_event=None, pass_id="pass_1")

    assert result == expected
    sent_payload = stream.call_args.args[0]
    assert sent_payload["model"] == "custom-model"


def test_llm_client_build_payload_uses_responses_shape_when_forced() -> None:
    llm_client = LLMClient({"agent": {"endpoint_mode": "responses"}})

    payload = llm_client.build_payload(
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

    mocker.patch.object(llm_client.provider, "_status_allows_immediate_send", return_value=ModelStatus(state="online"))
    llm_client.provider._stream_chat_completions = fake_stream

    result = llm_client.call_with_retry(payload, stop_event=None, on_event=None, pass_id="pass_1")

    assert result.finish_reason == "stop"
    assert result.content == "ok"
    assert calls[0].endswith("/v1/responses")
    assert calls[1].endswith("/v1/chat/completions")


def test_llm_client_detects_backend_profile_from_models_payload() -> None:
    llm_client = LLMClient({"agent": {"backend_profile": "auto"}})

    llm_client.provider._refresh_backend_profile({"data": [{"id": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"}]})
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
        llm_client.provider,
        "_status_allows_immediate_send",
        return_value=ModelStatus(state="online", model_name="mlx-community/Qwen2.5-VL-7B-Instruct-4bit"),
    )
    stream = mocker.patch.object(llm_client.provider, "stream_completion", return_value=expected)

    result = llm_client.call_with_retry(payload, stop_event=None, on_event=None, pass_id="pass_1")

    assert result == expected
    sent = stream.call_args.args[0]
    assert "chat_template_kwargs" not in sent
    assert "stream_options" not in sent
    assert sent["messages"][0]["content"][1]["image_url"] == "data:image/png;base64,ZmFrZQ=="


def test_llm_client_fails_fast_on_local_backend_model_mismatch(mocker) -> None:
    llm_client = LLMClient({"agent": {"backend_profile": "llamacpp"}})
    mocker.patch.object(
        llm_client.provider,
        "_status_allows_immediate_send",
        return_value=ModelStatus(state="online", model_name="qwen-3"),
    )
    mocker.patch.object(llm_client.provider, "stream_completion")

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
