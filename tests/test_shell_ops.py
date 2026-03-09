from __future__ import annotations

from pathlib import Path

from core.memory import VectorMemory
from core.skills import SkillContext, SkillRuntime
from core.workspace import WorkspaceManager


def _runtime(tmp_path: Path, config: dict) -> SkillRuntime:
    repo_root = Path(__file__).resolve().parents[1]
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    memory = VectorMemory(storage_path=str(tmp_path / "mem.pkl"))
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
