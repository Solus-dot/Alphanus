from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from agent.classifier import TurnClassifier
from agent.context import ContextWindowManager
from agent.orchestrator import TurnOrchestrator
from agent.policies import PromptPolicyRenderer
from agent.provider import LLMClient
from core.config_model import default_config
from core.types import ToolCall, ToolExecutionRecord, TurnClassification
from skills.runtime import SkillRuntime
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


def _patch_project_tool_runtime(mocker, runtime: SkillRuntime, *, read_ok: bool = True) -> None:
    def registration(name: str):
        capabilities = {
            "read_file": "project_read",
            "project_tree": "project_tree",
            "find_files": "project_read",
            "edit_file": "project_edit",
        }
        capability = capabilities.get(name, "")
        return SimpleNamespace(capability=capability, actions=["edit"] if name == "edit_file" else ["read"])

    def execute(name: str, args: dict[str, object], **_kwargs):
        if name == "read_file":
            return {
                "ok": read_ok,
                "data": {"filepath": str(args.get("filepath", "")), "content": "alpha\n"} if read_ok else None,
                "error": None if read_ok else {"code": "E_NOT_FOUND", "message": "missing"},
                "meta": {},
            }
        if name == "project_tree":
            return {"ok": True, "data": {"tree": "root/\n└── file.txt"}, "error": None, "meta": {}}
        if name == "edit_file":
            return {"ok": True, "data": {"filepath": str(args.get("filepath", "")), "edited": True}, "error": None, "meta": {}}
        return {"ok": False, "data": None, "error": {"code": "E_UNSUPPORTED", "message": name}, "meta": {}}

    mocker.patch.object(runtime, "tool_registration", side_effect=registration)
    mocker.patch.object(runtime, "tool_is_mutating", side_effect=lambda name: name == "edit_file")
    mocker.patch.object(runtime, "execute_tool_call", side_effect=execute)


def _stream_with_tool(call: ToolCall) -> SimpleNamespace:
    return SimpleNamespace(tool_calls=[call])


def test_orchestrator_policy_snapshot_captures_forced_flags_and_shell_exposure(mocker, tmp_path: Path) -> None:
    runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="latest updates",
        time_sensitive=True,
        project_action=True,
    )
    state.search_tools_enabled = True
    state.forced_search_retry = True
    state.forced_action_retry = True
    mocker.patch.object(
        runtime,
        "allowed_tool_names",
        return_value=["shell_command", "web_search", "request_user_input"],
    )

    snapshot = orchestrator.policy_engine.build_policy_snapshot(state)

    assert snapshot.search_mode is True
    assert snapshot.time_sensitive_query is True
    assert snapshot.forced_search_retry is True
    assert snapshot.requires_project_action is True
    assert snapshot.forced_action_retry is True
    assert snapshot.shell_tool_exposed is True
    assert snapshot.collaboration_mode == "execute"


def test_orchestrator_records_project_evidence_and_policy_blocks(tmp_path: Path) -> None:
    _runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="create a file",
        time_sensitive=False,
        project_action=True,
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
            "meta": {"project_changed": True},
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

    evidence = cast(Any, orchestrator.evidence_guard.project_action_evidence(state))

    assert state.completion.tool_counts["shell_command"] == 2
    assert evidence["has_successful_tool"] is True
    assert "shell_command" in evidence["successful_tools"]
    assert evidence["has_successful_mutation"] is True
    assert "shell_command" in evidence["successful_mutating_tools"]
    assert "shell_command" in evidence["policy_blocked_tools"]


def test_evidence_aggregates_the_whole_turn_but_bounds_recent_details(tmp_path: Path) -> None:
    _runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="create a file",
        time_sensitive=False,
        project_action=True,
    )
    state.evidence.append(
        ToolExecutionRecord(
            name="shell_command",
            args={"command": "touch notes.txt"},
            result={"ok": True, "data": {}, "error": None, "meta": {"project_changed": True}},
        )
    )
    state.evidence.extend(
        ToolExecutionRecord(name="skill_view", args={}, result={"ok": True, "data": {}, "error": None, "meta": {}}) for _ in range(12)
    )

    evidence = cast(Any, orchestrator.evidence_guard.project_action_evidence(state))

    assert evidence["successful_mutating_tools"] == ["shell_command"]
    assert len(evidence["recent_tool_details"]) == 12


def test_tool_loop_repeated_successful_read_is_blocked_then_stopped(mocker, tmp_path: Path) -> None:
    runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="read README",
        time_sensitive=False,
        project_action=False,
    )
    _reload_retrieval(orchestrator, enabled=False)
    _patch_project_tool_runtime(mocker, runtime)
    call = ToolCall(stream_id="s1", index=0, id="call_1", name="read_file", arguments={"filepath": "README.md"})

    status, result = orchestrator.tool_loop.execute_tool_calls(
        system_content="system",
        state=state,
        pass_id="pass_1",
        stream_result=_stream_with_tool(call),
    )
    assert status == "continue"
    assert result is None

    repeat = ToolCall(stream_id="s2", index=0, id="call_2", name="read_file", arguments={"filepath": "README.md"})
    status, result = orchestrator.tool_loop.execute_tool_calls(
        system_content="system",
        state=state,
        pass_id="pass_2",
        stream_result=_stream_with_tool(repeat),
    )
    assert status == "continue"
    assert result is None
    assert state.evidence[-1].policy_blocked is True
    assert cast(Any, state.evidence[-1].result)["error"]["code"] == "E_REPEATED_TOOL_CALL"

    stuck = ToolCall(stream_id="s3", index=0, id="call_3", name="read_file", arguments={"filepath": "README.md"})
    status, result = orchestrator.tool_loop.execute_tool_calls(
        system_content="system",
        state=state,
        pass_id="pass_3",
        stream_result=_stream_with_tool(stuck),
    )
    assert status == "result"
    assert result is not None
    assert result.status == "error"
    assert result.error == "tool_loop_stuck"
    assert state.skill_exchanges[-1].get("role") == "tool"
    assert json.loads(str(state.skill_exchanges[-1].get("content")))["error"]["code"] == "E_TOOL_LOOP_STUCK"


def test_tool_loop_max_depth_adds_synthetic_tool_result(mocker, tmp_path: Path) -> None:
    runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="read README",
        time_sensitive=False,
        project_action=False,
    )
    _reload_retrieval(orchestrator, enabled=False)
    _patch_project_tool_runtime(mocker, runtime)
    state.action_depth = orchestrator.max_action_depth
    call = ToolCall(stream_id="s1", index=0, id="call_1", name="read_file", arguments={"filepath": "README.md"})

    status, result = orchestrator.tool_loop.execute_tool_calls(
        system_content="system",
        state=state,
        pass_id="pass_1",
        stream_result=_stream_with_tool(call),
    )

    assert status == "result"
    assert result is not None
    assert result.status == "error"
    assert state.skill_exchanges[-2].get("role") == "assistant"
    assert state.skill_exchanges[-1].get("role") == "tool"
    assert state.skill_exchanges[-1].get("tool_call_id") == "call_1"
    assert json.loads(str(state.skill_exchanges[-1].get("content")))["error"]["code"] == "E_TOOL_LOOP_BUDGET"


def test_tool_loop_stalled_project_mutation_blocks_extra_inspection(mocker, tmp_path: Path) -> None:
    runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="capitalize headings in README.md",
        time_sensitive=False,
        project_action=True,
    )
    _reload_retrieval(orchestrator, enabled=False)
    _patch_project_tool_runtime(mocker, runtime)

    for idx, args in enumerate(
        [
            {"filepath": "README.md"},
            {"filepath": "README.md", "start_line": 1, "end_line": 10},
            {"filepath": "README.md", "start_line": 11, "end_line": 20},
        ],
        start=1,
    ):
        status, result = orchestrator.tool_loop.execute_tool_calls(
            system_content="system",
            state=state,
            pass_id=f"pass_{idx}",
            stream_result=_stream_with_tool(
                ToolCall(stream_id=f"s{idx}", index=0, id=f"call_{idx}", name="read_file", arguments=cast(Any, args))
            ),
        )
        assert status == "continue"
        assert result is None

    status, result = orchestrator.tool_loop.execute_tool_calls(
        system_content="system",
        state=state,
        pass_id="pass_4",
        stream_result=_stream_with_tool(
            ToolCall(stream_id="s4", index=0, id="call_4", name="project_tree", arguments={"path": ".", "max_depth": 2})
        ),
    )

    assert status == "continue"
    assert result is None
    assert state.evidence[-1].policy_blocked is True
    assert cast(Any, state.evidence[-1].result)["error"]["code"] == "E_PROJECT_ACTION_STALLED"

    status, result = orchestrator.tool_loop.execute_tool_calls(
        system_content="system",
        state=state,
        pass_id="pass_5",
        stream_result=_stream_with_tool(
            ToolCall(stream_id="s5", index=0, id="call_5", name="project_tree", arguments={"path": "src", "max_depth": 2})
        ),
    )
    assert status == "result"
    assert result is not None
    assert result.error == "project_action_stuck"


def test_tool_loop_stalled_inspection_does_not_skip_later_edit_in_same_pass(mocker, tmp_path: Path) -> None:
    runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="capitalize headings in README.md",
        time_sensitive=False,
        project_action=True,
    )
    _reload_retrieval(orchestrator, enabled=False)
    _patch_project_tool_runtime(mocker, runtime)

    for idx, args in enumerate(
        [
            {"filepath": "README.md"},
            {"filepath": "README.md", "start_line": 1, "end_line": 10},
            {"filepath": "README.md", "start_line": 11, "end_line": 20},
        ],
        start=1,
    ):
        status, result = orchestrator.tool_loop.execute_tool_calls(
            system_content="system",
            state=state,
            pass_id=f"pass_{idx}",
            stream_result=_stream_with_tool(
                ToolCall(stream_id=f"s{idx}", index=0, id=f"call_{idx}", name="read_file", arguments=cast(Any, args))
            ),
        )
        assert status == "continue"
        assert result is None

    stream_result = SimpleNamespace(
        tool_calls=[
            ToolCall(stream_id="s4", index=0, id="call_4", name="project_tree", arguments={"path": ".", "max_depth": 2}),
            ToolCall(stream_id="s5", index=1, id="call_5", name="edit_file", arguments={"filepath": "README.md", "content": "# Title\n"}),
        ]
    )
    status, result = orchestrator.tool_loop.execute_tool_calls(
        system_content="system",
        state=state,
        pass_id="pass_4",
        stream_result=stream_result,
    )

    assert status == "continue"
    assert result is None
    assert [record.name for record in state.evidence[-2:]] == ["project_tree", "edit_file"]
    assert cast(Any, state.evidence[-2].result)["error"]["code"] == "E_PROJECT_ACTION_STALLED"
    assert cast(Any, state.evidence[-1].result)["ok"] is True
    assert orchestrator.evidence_guard.project_mutation_count(state) == 1


def test_project_action_outcome_accepts_successful_non_mutating_open_action(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))

    outcome = classifier.classify_project_action_outcome(
        current_user_input="open it using the open command please",
        recent_routing_hint="",
        assistant_reply="The website has been opened in your default browser.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["shell_command"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "successful_action_labels": ["open", "run"],
            "policy_blocked_tools": [],
            "recent_tool_details": [
                {"name": "shell_command", "actions": ["open", "run"], "ok": True, "mutating": False, "policy_blocked": False}
            ],
        },
        pass_id="pass_1",
    )

    assert outcome == "completed_with_evidence"


def test_project_action_outcome_accepts_shell_version_query(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))

    outcome = classifier.classify_project_action_outcome(
        current_user_input="what go version am i running?",
        recent_routing_hint="",
        assistant_reply="You are running Go version `go1.26.4` on `darwin/arm64`.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["shell_command"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "successful_action_labels": ["run"],
            "policy_blocked_tools": [],
            "recent_tool_details": [{"name": "shell_command", "actions": ["run"], "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "completed_with_evidence"


def test_project_action_outcome_accepts_read_file_version_query(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))

    outcome = classifier.classify_project_action_outcome(
        current_user_input="read package.json and tell me the version",
        recent_routing_hint="",
        assistant_reply="I read package.json. The version is `1.2.3`.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["read_file"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "successful_action_labels": ["read"],
            "policy_blocked_tools": [],
            "recent_tool_details": [{"name": "read_file", "actions": ["read"], "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "completed_with_evidence"


def test_project_action_outcome_accepts_open_followup_after_mutating_hint(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))

    outcome = classifier.classify_project_action_outcome(
        current_user_input="open it",
        recent_routing_hint="previous user request: create notes.txt",
        assistant_reply="The website has been opened in your default browser.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["shell_command"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "successful_action_labels": ["open", "run"],
            "policy_blocked_tools": [],
            "recent_tool_details": [
                {"name": "shell_command", "actions": ["open", "run"], "ok": True, "mutating": False, "policy_blocked": False}
            ],
        },
        pass_id="pass_1",
    )

    assert outcome == "completed_with_evidence"


def test_project_action_outcome_accepts_show_request_with_list_evidence(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))

    outcome = classifier.classify_project_action_outcome(
        current_user_input="show me the files in the project",
        recent_routing_hint="",
        assistant_reply="I listed the files in the project.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["list_files"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "successful_action_labels": ["list", "read"],
            "policy_blocked_tools": [],
            "recent_tool_details": [
                {"name": "list_files", "actions": ["list", "read"], "ok": True, "mutating": False, "policy_blocked": False}
            ],
        },
        pass_id="pass_1",
    )

    assert outcome == "completed_with_evidence"


def test_project_action_outcome_accepts_display_tree_with_project_tree_evidence(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))

    outcome = classifier.classify_project_action_outcome(
        current_user_input="display the directory tree",
        recent_routing_hint="",
        assistant_reply="I displayed the directory tree.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["project_tree"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "successful_action_labels": ["read"],
            "policy_blocked_tools": [],
            "recent_tool_details": [{"name": "project_tree", "actions": ["read"], "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "completed_with_evidence"


def test_project_action_outcome_rejects_ack_followup_to_mutating_request_without_mutation(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))

    outcome = classifier.classify_project_action_outcome(
        current_user_input="yes, do it",
        recent_routing_hint="previous user request: create notes.txt",
        assistant_reply="I checked notes.txt.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["read_file"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "policy_blocked_tools": [],
            "recent_tool_details": [{"name": "read_file", "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "not_completed"


def test_project_action_outcome_overrides_structured_ack_followup_without_mutation(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": True}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))
    cast(Any, classifier).call_with_retry = lambda *_args, **_kwargs: SimpleNamespace(content='{"outcome":"completed_with_evidence"}')

    outcome = classifier.classify_project_action_outcome(
        current_user_input="do it",
        recent_routing_hint="previous user request: delete old.txt",
        assistant_reply="I checked old.txt.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["read_file"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "policy_blocked_tools": [],
            "recent_tool_details": [{"name": "read_file", "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "not_completed"


def test_project_action_outcome_rejects_open_claim_with_only_read_evidence(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))

    outcome = classifier.classify_project_action_outcome(
        current_user_input="open the app",
        recent_routing_hint="",
        assistant_reply="I opened the app.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["read_file"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "policy_blocked_tools": [],
            "recent_tool_details": [{"name": "read_file", "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "not_completed"


def test_project_action_outcome_overrides_structured_open_claim_with_only_read_evidence(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": True}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))
    cast(Any, classifier).call_with_retry = lambda *_args, **_kwargs: SimpleNamespace(content='{"outcome":"completed_with_evidence"}')

    outcome = classifier.classify_project_action_outcome(
        current_user_input="open the app",
        recent_routing_hint="",
        assistant_reply="I opened the app.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["read_file"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "policy_blocked_tools": [],
            "recent_tool_details": [{"name": "read_file", "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "not_completed"


def test_project_action_outcome_still_rejects_file_creation_claim_without_mutation(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))

    outcome = classifier.classify_project_action_outcome(
        current_user_input="create notes.txt",
        recent_routing_hint="",
        assistant_reply="I created notes.txt.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["read_file"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "policy_blocked_tools": [],
            "recent_tool_details": [{"name": "read_file", "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "not_completed"


def test_project_action_outcome_rejects_non_mutating_completion_for_mutating_request(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": False}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))

    outcome = classifier.classify_project_action_outcome(
        current_user_input="create notes.txt",
        recent_routing_hint="",
        assistant_reply="I checked notes.txt.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["read_file"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "policy_blocked_tools": [],
            "recent_tool_details": [{"name": "read_file", "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "not_completed"


def test_project_action_outcome_overrides_structured_completion_without_mutation(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": True}}
    classifier = TurnClassifier(cfg, runtime, LLMClient(cfg))
    cast(Any, classifier).call_with_retry = lambda *_args, **_kwargs: SimpleNamespace(content='{"outcome":"completed_with_evidence"}')

    outcome = classifier.classify_project_action_outcome(
        current_user_input="create notes.txt",
        recent_routing_hint="",
        assistant_reply="I checked notes.txt.",
        evidence={
            "has_successful_tool": True,
            "successful_tools": ["read_file"],
            "has_successful_mutation": False,
            "successful_mutating_tools": [],
            "policy_blocked_tools": [],
            "recent_tool_details": [{"name": "read_file", "ok": True, "mutating": False, "policy_blocked": False}],
        },
        pass_id="pass_1",
    )

    assert outcome == "not_completed"


def test_orchestrator_search_budget_reason_is_explicit_for_time_sensitive_turns(tmp_path: Path) -> None:
    _runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="latest status",
        time_sensitive=True,
        project_action=False,
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

    reason = orchestrator.policy_engine.tool_budget_reason(state, web_search_call)

    assert reason is not None
    assert "search-attempt budget is exhausted" in reason


def test_search_tool_effects_record_attempt_metadata_and_empty_evidence(tmp_path: Path) -> None:
    _runtime, orchestrator, state = _turn_state(
        tmp_path,
        user_input="latest status",
        time_sensitive=True,
        project_action=False,
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
