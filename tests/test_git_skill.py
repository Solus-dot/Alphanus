from __future__ import annotations

import subprocess
from pathlib import Path

from core.memory import LexicalMemory
from core.skills import SkillContext, SkillRuntime
from core.workspace import WorkspaceManager


def _runtime(tmp_path: Path) -> SkillRuntime:
    repo_root = Path(__file__).resolve().parents[1]
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    return SkillRuntime(
        skills_dir=str(repo_root / "skills"),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )


def _ctx(workspace_root: str) -> SkillContext:
    return SkillContext(
        user_input="use git",
        branch_labels=[],
        attachments=[],
        workspace_root=workspace_root,
        memory_hits=[],
        loaded_skill_ids=["git"],
    )


def _git(runtime: SkillRuntime, tool_name: str, args: dict[str, object]):
    skill = runtime.get_skill("git")
    assert skill is not None
    return runtime.execute_tool_call(tool_name, args, selected=[skill], ctx=_ctx(str(runtime.workspace.workspace_root)))


def _run_git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


def _init_repo(runtime: SkillRuntime, path: str = "repo") -> Path:
    out = _git(runtime, "git_init", {"path": path, "create_if_missing": True})
    assert out["ok"] is True
    repo = Path(out["data"]["path"])
    _run_git(repo, "config", "user.name", "Alphanus Test")
    _run_git(repo, "config", "user.email", "test@example.invalid")
    return repo


def test_git_skill_loads_with_core_tool_surface(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("git")

    assert skill is not None
    assert skill.enabled is True
    assert set(skill.allowed_tools) >= {
        "git_status",
        "git_log",
        "git_diff",
        "git_show",
        "git_branch_list",
        "git_branch_create",
        "git_branch_switch",
        "git_add",
        "git_commit",
        "git_fetch",
        "git_pull",
        "git_push",
        "git_init",
    }
    assert "git_init" in runtime.allowed_tool_names([skill], ctx=_ctx(str(runtime.workspace.workspace_root)))


def test_git_init_blocks_workspace_root(tmp_path: Path):
    runtime = _runtime(tmp_path)

    out = _git(runtime, "git_init", {"path": ".", "create_if_missing": False})

    assert out["ok"] is False
    assert out["error"]["code"] == "E_POLICY"
    assert out["data"]["block_reason"] == "workspace_root"


def test_git_init_allows_nested_subfolder(tmp_path: Path):
    runtime = _runtime(tmp_path)

    out = _git(runtime, "git_init", {"path": "project", "create_if_missing": True})

    assert out["ok"] is True
    assert out["data"]["created"] is True
    assert Path(out["data"]["path"]).name == "project"
    assert (Path(out["data"]["path"]) / ".git").is_dir()


def test_git_init_blocks_path_escape(tmp_path: Path):
    runtime = _runtime(tmp_path)

    out = _git(runtime, "git_init", {"path": "../outside", "create_if_missing": True})

    assert out["ok"] is False
    assert out["error"]["code"] == "E_POLICY"
    assert "escapes workspace root" in out["error"]["message"]


def test_git_init_blocks_nested_repo_initialization(tmp_path: Path):
    runtime = _runtime(tmp_path)
    repo = _init_repo(runtime)
    nested = repo / "nested"
    nested.mkdir()

    out = _git(runtime, "git_init", {"path": "repo/nested", "create_if_missing": False})

    assert out["ok"] is False
    assert out["error"]["code"] == "E_POLICY"
    assert out["data"]["block_reason"] == "nested_repo"
    assert out["data"]["existing_repo_root"] == str(repo)


def test_git_init_requires_create_flag_for_missing_directory(tmp_path: Path):
    runtime = _runtime(tmp_path)

    out = _git(runtime, "git_init", {"path": "missing", "create_if_missing": False})

    assert out["ok"] is False
    assert out["error"]["code"] == "E_NOT_FOUND"
    assert out["data"]["block_reason"] == "missing"


def test_git_read_operations_return_structured_output(tmp_path: Path):
    runtime = _runtime(tmp_path)
    repo = _init_repo(runtime)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(runtime, "git_add", {"path": "repo", "paths": ["repo/README.md"]})
    committed = _git(runtime, "git_commit", {"path": "repo", "message": "initial commit"})
    assert committed["ok"] is True

    status = _git(runtime, "git_status", {"path": "repo"})
    log = _git(runtime, "git_log", {"path": "repo", "max_count": 5})
    show = _git(runtime, "git_show", {"path": "repo", "rev": "HEAD"})

    assert status["ok"] is True
    assert status["data"]["entries"] == []
    assert log["ok"] is True
    assert log["data"]["count"] == 1
    assert log["data"]["commits"][0]["subject"] == "initial commit"
    assert show["ok"] is True
    assert "initial commit" in show["data"]["content"]


def test_git_pathspecs_accept_repo_relative_paths_for_nested_repos(tmp_path: Path):
    runtime = _runtime(tmp_path)
    repo = _init_repo(runtime)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")

    added = _git(runtime, "git_add", {"path": "repo", "paths": ["README.md"]})
    diff = _git(runtime, "git_diff", {"path": "repo", "staged": True, "paths": ["README.md"]})

    assert added["ok"] is True
    assert added["data"]["staged_paths"] == ["README.md"]
    assert diff["ok"] is True
    assert "README.md" in diff["data"]["diff"]


def test_git_commit_requires_message_and_staged_changes(tmp_path: Path):
    runtime = _runtime(tmp_path)
    repo = _init_repo(runtime)
    (repo / "notes.txt").write_text("unstaged\n", encoding="utf-8")

    empty_message = _git(runtime, "git_commit", {"path": "repo", "message": "   "})
    no_staged = _git(runtime, "git_commit", {"path": "repo", "message": "commit"})

    assert empty_message["ok"] is False
    assert empty_message["error"]["code"] == "E_VALIDATION"
    assert no_staged["ok"] is False
    assert no_staged["error"]["code"] == "E_NOOP"


def test_git_branch_switch_rejects_dirty_worktree(tmp_path: Path):
    runtime = _runtime(tmp_path)
    repo = _init_repo(runtime)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(runtime, "git_add", {"path": "repo", "paths": ["repo/README.md"]})
    _git(runtime, "git_commit", {"path": "repo", "message": "initial commit"})
    branch = _git(runtime, "git_branch_create", {"path": "repo", "name": "feature"})
    assert branch["ok"] is True
    (repo / "README.md").write_text("dirty\n", encoding="utf-8")

    switched = _git(runtime, "git_branch_switch", {"path": "repo", "name": "feature"})

    assert switched["ok"] is False
    assert switched["error"]["code"] == "E_DIRTY_WORKTREE"


def test_git_push_requires_confirmation_and_rejects_force(tmp_path: Path):
    runtime = _runtime(tmp_path)
    _init_repo(runtime)

    needs_confirmation = _git(runtime, "git_push", {"path": "repo", "confirm_push": False})
    force = _git(runtime, "git_push", {"path": "repo", "confirm_push": True, "force": True})

    assert needs_confirmation["ok"] is False
    assert needs_confirmation["error"]["code"] == "E_CONFIRMATION_REQUIRED"
    assert force["ok"] is False
    assert force["error"]["code"] == "E_POLICY"


def test_git_pull_normalizes_rebase_conflict_errors(tmp_path: Path):
    runtime = _runtime(tmp_path)
    repo = _init_repo(runtime)
    (repo / "README.md").write_text("local\n", encoding="utf-8")
    _git(runtime, "git_add", {"path": "repo", "paths": ["repo/README.md"]})
    _git(runtime, "git_commit", {"path": "repo", "message": "local commit"})

    out = _git(runtime, "git_pull", {"path": "repo", "remote": "origin", "branch": "main"})

    assert out["ok"] is False
    assert out["error"]["code"] in {"E_GIT", "E_CONFLICT"}
    assert out["data"]["argv"][:3] == ["git", "pull", "--rebase"]
