from __future__ import annotations

from pathlib import Path
from typing import cast

from agent.core import Agent
from core.memory import LexicalMemory
from core.message_types import ChatMessage
from core.project import ProjectRuntime
from skills.runtime import SkillContext, SkillRuntime


def _runtime(tmp_path: Path, config: dict) -> SkillRuntime:
    repo_root = Path(__file__).resolve().parents[1]
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    memory = LexicalMemory(storage_path=str(tmp_path / "mem.pkl"))
    return SkillRuntime(
        skills_dir=str(repo_root / "bundled-skills"),
        project=ProjectRuntime(str(ws)),
        memory=memory,
        config=config,
    )


def _ctx(project_root: str) -> SkillContext:
    return SkillContext(
        user_input="run command",
        branch_labels=[],
        attachments=[],
        project_root=project_root,
        memory_hits=[],
    )


def test_shell_command_requires_approval_callback_for_high_risk_command(tmp_path: Path):
    runtime = _runtime(tmp_path, {"permissions": {"mode": "project-write", "approvals": "on-boundary", "network": False}})
    shell_skill = runtime.get_skill("shell-ops")
    assert shell_skill is not None

    out = runtime.execute_tool_call(
        "shell_command",
        {"command": "rm -rf build"},
        selected=[shell_skill],
        ctx=_ctx(str(runtime.project.project_root)),
        request_approval=None,
    )
    assert out["ok"] is False
    assert out["error"]["code"] == "E_POLICY"
    assert out["error"]["message"] == "Approval callback is required"


def test_shell_command_rejected_by_user(tmp_path: Path):
    runtime = _runtime(tmp_path, {"permissions": {"mode": "project-write", "approvals": "on-boundary", "network": False}})
    shell_skill = runtime.get_skill("shell-ops")
    assert shell_skill is not None

    out = runtime.execute_tool_call(
        "shell_command",
        {"command": "rm -rf build"},
        selected=[shell_skill],
        ctx=_ctx(str(runtime.project.project_root)),
        request_approval=lambda _: False,
    )
    assert out["ok"] is False
    assert out["error"]["code"] == "E_POLICY"
    assert out["error"]["message"] == "Shell command rejected by user"


def test_shell_command_skips_confirmation_when_dangerous_mode_enabled(tmp_path: Path):
    runtime = _runtime(
        tmp_path,
        {"permissions": {"mode": "danger-full-access", "approvals": "on-boundary", "network": False}},
    )
    shell_skill = runtime.get_skill("shell-ops")
    assert shell_skill is not None

    def _must_not_be_called(_: dict) -> bool:
        raise AssertionError("request_approval should not be called when dangerous mode is enabled")

    out = runtime.execute_tool_call(
        "shell_command",
        {"command": "echo hi"},
        selected=[shell_skill],
        ctx=_ctx(str(runtime.project.project_root)),
        request_approval=_must_not_be_called,
    )
    assert out["ok"] is True
    assert out["data"]["returncode"] == 0
    assert out["data"]["stdout"].strip() == "hi"


def test_shell_command_executes_with_selected_shell_skill(tmp_path: Path):
    runtime = _runtime(
        tmp_path,
        {"permissions": {"mode": "project-write", "approvals": "on-boundary", "network": False}},
    )

    shell_skill = runtime.get_skill("shell-ops")
    assert shell_skill is not None

    out = runtime.execute_tool_call(
        "shell_command",
        {"command": "echo hi"},
        selected=[shell_skill],
        ctx=_ctx(str(runtime.project.project_root)),
        request_approval=lambda _: True,
    )

    assert out["ok"] is True
    assert out["data"]["returncode"] == 0
    assert out["data"]["stdout"].strip() == "hi"


def test_shell_command_uses_longer_default_timeout(mocker, tmp_path: Path):
    runtime = _runtime(
        tmp_path,
        {"permissions": {"mode": "danger-full-access", "approvals": "on-boundary", "network": False}},
    )
    shell_skill = runtime.get_skill("shell-ops")
    assert shell_skill is not None
    run = mocker.patch.object(
        runtime.project,
        "run_shell_command",
        return_value={"ok": True, "data": {"returncode": 0}, "error": None, "meta": {}},
    )

    out = runtime.execute_tool_call(
        "shell_command",
        {"command": "llama-update"},
        selected=[shell_skill],
        ctx=_ctx(str(runtime.project.project_root)),
    )

    assert out["ok"] is True
    run.assert_called_once_with("llama-update", timeout_s=600, cwd=None, allowed_cwd_roots=None, approved=False)


def test_shell_command_allows_explicit_timeout_and_caps_it(mocker, tmp_path: Path):
    runtime = _runtime(
        tmp_path,
        {"permissions": {"mode": "danger-full-access", "approvals": "on-boundary", "network": False}},
    )
    shell_skill = runtime.get_skill("shell-ops")
    assert shell_skill is not None
    run = mocker.patch.object(
        runtime.project,
        "run_shell_command",
        return_value={"ok": True, "data": {"returncode": 0}, "error": None, "meta": {}},
    )

    runtime.execute_tool_call(
        "shell_command",
        {"command": "make", "timeout_s": 1200},
        selected=[shell_skill],
        ctx=_ctx(str(runtime.project.project_root)),
    )
    runtime.execute_tool_call(
        "shell_command",
        {"command": "make world", "timeout_s": 99999},
        selected=[shell_skill],
        ctx=_ctx(str(runtime.project.project_root)),
    )

    assert run.call_args_list[0].args == ("make",)
    assert run.call_args_list[0].kwargs == {"timeout_s": 1200, "cwd": None, "allowed_cwd_roots": None, "approved": False}
    assert run.call_args_list[1].args == ("make world",)
    assert run.call_args_list[1].kwargs == {"timeout_s": 7200, "cwd": None, "allowed_cwd_roots": None, "approved": False}


def test_shell_command_external_cwd_requests_approval_and_forwards_cwd(mocker, tmp_path: Path):
    runtime = _runtime(
        tmp_path,
        {"permissions": {"mode": "project-write", "approvals": "on-boundary", "network": False}},
    )
    shell_skill = runtime.get_skill("shell-ops")
    assert shell_skill is not None
    outside = runtime.project.project_root.parent / "outside"
    outside.mkdir()
    run = mocker.patch.object(
        runtime.project,
        "run_shell_command",
        return_value={"ok": True, "data": {"returncode": 0}, "error": None, "meta": {}},
    )
    approvals: list[dict] = []

    out = runtime.execute_tool_call(
        "shell_command",
        {"command": "pwd", "cwd": str(outside)},
        selected=[shell_skill],
        ctx=_ctx(str(runtime.project.project_root)),
        request_approval=lambda request: approvals.append(request) or True,
    )

    assert out["ok"] is True
    assert approvals and approvals[0]["cwd"] == str(outside)
    run.assert_called_once_with(
        "pwd",
        timeout_s=600,
        cwd=str(outside),
        allowed_cwd_roots=[str(outside)],
        approved=True,
    )


def test_shell_command_external_argument_requests_path_approval(mocker, tmp_path: Path):
    runtime = _runtime(
        tmp_path,
        {"permissions": {"mode": "project-write", "approvals": "on-boundary", "network": False}},
    )
    shell_skill = runtime.get_skill("shell-ops")
    assert shell_skill is not None
    outside_file = runtime.project.project_root.parent / "outside.py"
    outside_file.write_text("print('ok')\n", encoding="utf-8")
    run = mocker.patch.object(
        runtime.project,
        "run_shell_command",
        return_value={"ok": True, "data": {"returncode": 0}, "error": None, "meta": {}},
    )
    approvals: list[dict] = []
    command = f"python3 {outside_file}"

    out = runtime.execute_tool_call(
        "shell_command",
        {"command": command},
        selected=[shell_skill],
        ctx=_ctx(str(runtime.project.project_root)),
        request_approval=lambda request: approvals.append(request) or True,
    )

    assert out["ok"] is True
    assert approvals[0]["paths"] == [str(outside_file.resolve())]
    run.assert_called_once_with(command, timeout_s=600, cwd=None, allowed_cwd_roots=None, approved=True)


def test_shell_command_nonzero_exit_bubbles_up_as_tool_failure(tmp_path: Path):
    runtime = _runtime(
        tmp_path,
        {"permissions": {"mode": "project-write", "approvals": "on-boundary", "network": False}},
    )

    shell_skill = runtime.get_skill("shell-ops")
    assert shell_skill is not None

    out = runtime.execute_tool_call(
        "shell_command",
        {"command": 'python3 -c "raise SystemExit(3)"'},
        selected=[shell_skill],
        ctx=_ctx(str(runtime.project.project_root)),
        request_approval=lambda _: True,
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
        project_root=str(runtime.project.project_root),
        memory_hits=[],
        loaded_skill_ids=["shell-ops", "memory-rag"],
    )

    selected_ids = {skill.id for skill in runtime.select_skills(ctx)}
    assert "shell-ops" in selected_ids
    assert "memory-rag" in selected_ids


def test_action_approvalation_reuses_recent_assistant_action_context(mocker, tmp_path: Path):
    runtime = _runtime(tmp_path, {})
    agent = Agent({"agent": {}}, runtime)
    mocker.patch("agent.classifier.TurnClassifier._should_model_classify", return_value=True)

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        assert pass_id == "turn_classify"
        return type("R", (), {"finish_reason": "stop", "content": '{"followup_kind":"confirmation"}'})()

    mocker.patch.object(agent.llm_client, "call_with_retry", side_effect=fake_call_with_retry)

    history_messages = cast(
        list[ChatMessage],
        [
        {"role": "user", "content": "how do i check my go version"},
        {"role": "assistant", "content": "I can run `go version` in the project if you want."},
        ],
    )

    ctx = agent.classifier.build_skill_context("Yeah check my version", [], [], history_messages)
    ctx.loaded_skill_ids = ["shell-ops"]

    assert "assistant just said:" in ctx.recent_routing_hint
    assert "go version" in ctx.recent_routing_hint

    selected = agent.skill_runtime.select_skills(ctx)

    assert selected
    assert any(skill.id == "shell-ops" for skill in selected)


def test_shell_command_tool_description_and_skill_block_delegate_confirmation_to_tool(tmp_path: Path):
    runtime = _runtime(tmp_path, {})
    shell_skill = runtime.get_skill("shell-ops")
    assert shell_skill is not None

    ctx = _ctx(str(runtime.project.project_root))
    tools = runtime.tools_for_turn([shell_skill], ctx=ctx)
    shell_tool = next(tool["function"] for tool in tools if tool["function"]["name"] == "shell_command")
    assert "tool itself asks for confirmation" in shell_tool["description"]
    assert "do not ask separately" in shell_tool["description"]
    assert "Shell syntax" in shell_tool["description"]
    assert "&&" in shell_tool["description"]

    block = runtime.compose_skill_block([shell_skill], context_limit=2048)
    assert "tool itself asks the user for confirmation" in block
    assert "Do not ask for duplicate confirmation" in block
    assert "Shell syntax is available" in block
    assert "malicious" in block
