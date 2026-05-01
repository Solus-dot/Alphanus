from __future__ import annotations

import threading
from pathlib import Path
from typing import cast

from agent.core import Agent
from core.memory import LexicalMemory
from core.message_types import ChatMessage
from core.skills import SkillContext, SkillRuntime
from core.workspace import WorkspaceManager


def _runtime(tmp_path: Path, config: dict) -> SkillRuntime:
    repo_root = Path(__file__).resolve().parents[1]
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    memory = LexicalMemory(storage_path=str(tmp_path / "mem.pkl"))
    return SkillRuntime(
        skills_dir=str(repo_root / "skills"),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=memory,
        config=config,
    )


def _ctx(workspace_root: str) -> SkillContext:
    return SkillContext(
        user_input="run command",
        branch_labels=[],
        attachments=[],
        workspace_root=workspace_root,
        memory_hits=[],
    )


def test_shell_command_requires_confirmation_callback(tmp_path: Path):
    runtime = _runtime(tmp_path, {"capabilities": {"shell_require_confirmation": True}})
    shell_skill = runtime.get_skill("shell-ops")
    assert shell_skill is not None

    out = runtime.execute_tool_call(
        "shell_command",
        {"command": "echo hi"},
        selected=[shell_skill],
        ctx=_ctx(str(runtime.workspace.workspace_root)),
        confirm_shell=None,
    )
    assert out["ok"] is False
    assert out["error"]["code"] == "E_POLICY"
    assert out["error"]["message"] == "Shell confirmation callback is required"


def test_shell_command_rejected_by_user(tmp_path: Path):
    runtime = _runtime(tmp_path, {"capabilities": {"shell_require_confirmation": True}})
    shell_skill = runtime.get_skill("shell-ops")
    assert shell_skill is not None

    out = runtime.execute_tool_call(
        "shell_command",
        {"command": "echo hi"},
        selected=[shell_skill],
        ctx=_ctx(str(runtime.workspace.workspace_root)),
        confirm_shell=lambda _: False,
    )
    assert out["ok"] is False
    assert out["error"]["code"] == "E_POLICY"
    assert out["error"]["message"] == "Shell command rejected by user"


def test_shell_command_skips_confirmation_when_dangerous_mode_enabled(tmp_path: Path):
    runtime = _runtime(
        tmp_path,
        {
            "capabilities": {
                "shell_require_confirmation": True,
                "dangerously_skip_permissions": True,
            }
        },
    )
    shell_skill = runtime.get_skill("shell-ops")
    assert shell_skill is not None

    def _must_not_be_called(_: str) -> bool:
        raise AssertionError("confirm_shell should not be called when dangerous mode is enabled")

    out = runtime.execute_tool_call(
        "shell_command",
        {"command": "echo hi"},
        selected=[shell_skill],
        ctx=_ctx(str(runtime.workspace.workspace_root)),
        confirm_shell=_must_not_be_called,
    )
    assert out["ok"] is True
    assert out["data"]["returncode"] == 0
    assert out["data"]["stdout"].strip() == "hi"


def test_shell_command_recovers_from_raw_argument_payload(tmp_path: Path):
    runtime = _runtime(
        tmp_path,
        {
            "capabilities": {
                "shell_require_confirmation": False,
                "dangerously_skip_permissions": True,
            }
        },
    )
    shell_skill = runtime.get_skill("shell-ops")
    assert shell_skill is not None

    out = runtime.execute_tool_call(
        "shell_command",
        {"_raw": '{"command":"echo hi"}'},
        selected=[shell_skill],
        ctx=_ctx(str(runtime.workspace.workspace_root)),
    )
    assert out["ok"] is True
    assert out["data"]["returncode"] == 0
    assert out["data"]["stdout"].strip() == "hi"


def test_shell_command_executes_with_selected_shell_skill(tmp_path: Path):
    runtime = _runtime(
        tmp_path,
        {
            "capabilities": {
                "shell_require_confirmation": True,
                "dangerously_skip_permissions": False,
            }
        },
    )

    shell_skill = runtime.get_skill("shell-ops")
    assert shell_skill is not None

    out = runtime.execute_tool_call(
        "shell_command",
        {"command": "echo hi"},
        selected=[shell_skill],
        ctx=_ctx(str(runtime.workspace.workspace_root)),
        confirm_shell=lambda _: True,
    )

    assert out["ok"] is True
    assert out["data"]["returncode"] == 0
    assert out["data"]["stdout"].strip() == "hi"


def test_shell_command_nonzero_exit_bubbles_up_as_tool_failure(tmp_path: Path):
    runtime = _runtime(
        tmp_path,
        {
            "capabilities": {
                "shell_require_confirmation": True,
                "dangerously_skip_permissions": False,
            }
        },
    )

    shell_skill = runtime.get_skill("shell-ops")
    assert shell_skill is not None

    out = runtime.execute_tool_call(
        "shell_command",
        {"command": 'python3 -c "raise SystemExit(3)"'},
        selected=[shell_skill],
        ctx=_ctx(str(runtime.workspace.workspace_root)),
        confirm_shell=lambda _: True,
    )

    assert out["ok"] is False
    assert out["error"]["code"] == "E_SHELL"
    assert "code 3" in out["error"]["message"]
    assert out["data"]["returncode"] == 3


def test_runtime_select_skills_returns_loaded_shell_and_memory_skills(tmp_path: Path):
    runtime = _runtime(tmp_path, {})
    ctx = SkillContext(
        user_input="check my go version",
        branch_labels=[],
        attachments=[],
        workspace_root=str(runtime.workspace.workspace_root),
        memory_hits=[],
        loaded_skill_ids=["shell-ops", "memory-rag"],
    )

    selected_ids = {skill.id for skill in runtime.select_skills(ctx)}
    assert "shell-ops" in selected_ids
    assert "memory-rag" in selected_ids


def test_shell_confirmation_reuses_recent_assistant_action_context(mocker, tmp_path: Path):
    runtime = _runtime(tmp_path, {})
    agent = Agent({"agent": {}}, runtime)
    mocker.patch("agent.classifier.TurnClassifier._should_model_classify", return_value=True)

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        assert pass_id == "turn_classify"
        return type("R", (), {"finish_reason": "stop", "content": '{"followup_kind":"confirmation"}'})()

    mocker.patch.object(agent, "_call_with_retry", side_effect=fake_call_with_retry)

    history_messages = cast(
        list[ChatMessage],
        [
        {"role": "user", "content": "how do i check my go version"},
        {"role": "assistant", "content": "I can run `go version` in the workspace if you want."},
        ],
    )

    ctx = agent._build_skill_context("Yeah check my version", [], [], history_messages)
    ctx.loaded_skill_ids = ["shell-ops"]

    assert "assistant just said:" in ctx.recent_routing_hint
    assert "go version" in ctx.recent_routing_hint

    selected = agent._select_skills(ctx, threading.Event())

    assert selected
    assert any(skill.id == "shell-ops" for skill in selected)


def test_shell_command_tool_description_and_skill_block_delegate_confirmation_to_tool(tmp_path: Path):
    runtime = _runtime(tmp_path, {})
    shell_skill = runtime.get_skill("shell-ops")
    assert shell_skill is not None

    ctx = _ctx(str(runtime.workspace.workspace_root))
    tools = runtime.tools_for_turn([shell_skill], ctx=ctx)
    shell_tool = next(tool["function"] for tool in tools if tool["function"]["name"] == "shell_command")
    assert "tool itself asks for confirmation" in shell_tool["description"]
    assert "do not ask separately" in shell_tool["description"]

    block = runtime.compose_skill_block([shell_skill], ctx, context_limit=2048)
    assert "tool itself asks the user for confirmation" in block
    assert "Do not ask for duplicate confirmation" in block
