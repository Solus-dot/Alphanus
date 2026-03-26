from __future__ import annotations

import json
from pathlib import Path

from agent.classifier import TurnClassifier
from agent.llm_client import LLMClient
from agent.telemetry import TelemetryEmitter, configure_logging
from core.memory import VectorMemory
from core.skills import SkillRuntime
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


def test_workspace_action_outcome_fallback_rejects_manual_terminal_advice_even_with_blocked_evidence(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    llm_client = LLMClient(cfg)
    classifier = TurnClassifier(cfg, runtime, llm_client)

    outcome = classifier.classify_workspace_action_outcome(
        current_user_input="yes",
        recent_routing_hint="",
        assistant_reply="shell_command is not allowed here. Please run rm -rf manually in your terminal.",
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
