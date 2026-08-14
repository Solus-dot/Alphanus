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
from agent.turn_policy_engine import build_turn_state, tool_budget_reason
from core.config_model import default_config
from core.types import ToolCall, TurnClassification
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
    state = build_turn_state(runtime, orchestrator.default_tool_budgets, ctx, [], [], classification)
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
            "create_file": "project_write",
        }
        capability = capabilities.get(name, "")
        return SimpleNamespace(capability=capability, actions=["edit"] if name == "edit_file" else ["read"])

    def execute(name: str, args: dict[str, object], **_kwargs):
        if name == "read_file":
            return {
                "ok": read_ok,
                "data": {
                    "filepath": str(args.get("filepath", "")),
                    "content": "alpha\n",
                    "resolved_start_line": args.get("start_line") or 1,
                    "resolved_end_line": args.get("end_line") or 20,
                    "total_line_count": 20,
                }
                if read_ok
                else None,
                "error": None if read_ok else {"code": "E_NOT_FOUND", "message": "missing"},
                "meta": {},
            }
        if name == "project_tree":
            return {"ok": True, "data": {"tree": "root/\n└── file.txt"}, "error": None, "meta": {}}
        if name in {"create_file", "edit_file"}:
            return {
                "ok": True,
                "data": {"filepath": str(args.get("filepath", "")), "write_verified": True},
                "error": None,
                "meta": {},
            }
        return {"ok": False, "data": None, "error": {"code": "E_UNSUPPORTED", "message": name}, "meta": {}}

    mocker.patch.object(runtime, "tool_registration", side_effect=registration)
    mocker.patch.object(runtime, "tool_is_mutating", side_effect=lambda name: name in {"create_file", "edit_file"})
    mocker.patch.object(runtime, "execute_tool_call", side_effect=execute)


def _stream_with_tool(call: ToolCall) -> SimpleNamespace:
    return SimpleNamespace(tool_calls=[call])


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

    assert state.completion.tool_counts["shell_command"] == 2
    assert [record.result["ok"] for record in state.evidence] == [True, False]
    assert state.evidence[-1].policy_blocked is True


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
    assert cast(Any, state.evidence[-1].result)["error"]["code"] == "E_READ_RANGE_COVERED"

    stuck = ToolCall(stream_id="s3", index=0, id="call_3", name="read_file", arguments={"filepath": "README.md"})
    status, result = orchestrator.tool_loop.execute_tool_calls(
        system_content="system",
        state=state,
        pass_id="pass_3",
        stream_result=_stream_with_tool(stuck),
    )
    assert status == "continue"
    assert result is None
    assert state.skill_exchanges[-1].get("role") == "tool"
    assert json.loads(str(state.skill_exchanges[-1].get("content")))["error"]["code"] == "E_READ_RANGE_COVERED"


def test_overlapping_read_ranges_are_merged_and_covered_reads_blocked(mocker, tmp_path: Path) -> None:
    runtime, orchestrator, state = _turn_state(tmp_path, user_input="inspect README", time_sensitive=False, project_action=True)
    _patch_project_tool_runtime(mocker, runtime)

    for index, (start, end) in enumerate(((1, 10), (8, 20)), start=1):
        status, result = orchestrator.tool_loop.execute_tool_calls(
            system_content="system",
            state=state,
            pass_id=f"pass_{index}",
            stream_result=_stream_with_tool(
                ToolCall(
                    stream_id=f"s{index}",
                    index=0,
                    id=f"call_{index}",
                    name="read_file",
                    arguments={"filepath": "README.md", "start_line": start, "end_line": end},
                )
            ),
        )
        assert status == "continue"
        assert result is None

    assert state.read_line_ranges["README.md"] == [(1, 20)]
    status, result = orchestrator.tool_loop.execute_tool_calls(
        system_content="system",
        state=state,
        pass_id="pass_3",
        stream_result=_stream_with_tool(
            ToolCall(
                stream_id="s3",
                index=0,
                id="call_3",
                name="read_file",
                arguments={"filepath": "README.md", "start_line": 5, "end_line": 15},
            )
        ),
    )
    assert status == "continue"
    assert result is None
    assert cast(Any, state.evidence[-1].result)["error"]["code"] == "E_READ_RANGE_COVERED"


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

    reason = tool_budget_reason(state, web_search_call)

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
