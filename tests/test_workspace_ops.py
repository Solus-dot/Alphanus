from __future__ import annotations

from pathlib import Path

from core.memory import VectorMemory
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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )


def _ctx(workspace_root: str) -> SkillContext:
    return SkillContext(
        user_input="edit a file",
        branch_labels=[],
        attachments=[],
        workspace_root=workspace_root,
        memory_hits=[],
    )


def test_workspace_ops_returns_rich_file_metadata(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("workspace-ops")
    assert skill is not None

    ctx = _ctx(str(runtime.workspace.workspace_root))
    created = runtime.execute_tool_call(
        "create_file",
        {"filepath": "notes.txt", "content": "alpha\nbeta\n"},
        selected=[skill],
        ctx=ctx,
    )
    assert created["ok"] is True
    assert created["data"]["created"] is True
    assert created["data"]["bytes_written"] > 0
    assert created["data"]["line_count"] == 3

    edited = runtime.execute_tool_call(
        "edit_file",
        {"filepath": "notes.txt", "content": "alpha\ngamma\n"},
        selected=[skill],
        ctx=ctx,
    )
    assert edited["ok"] is True
    assert edited["data"]["edited"] is True
    assert edited["data"]["changed"] is True
    assert edited["data"]["changed_lines"] == 1
    assert edited["data"]["line_count_before"] == 3
    assert edited["data"]["line_count_after"] == 3

    read = runtime.execute_tool_call(
        "read_file",
        {"filepath": "notes.txt"},
        selected=[skill],
        ctx=ctx,
    )
    assert read["ok"] is True
    assert read["data"]["content"] == "alpha\ngamma\n"
    assert read["data"]["size_bytes"] > 0
    assert read["data"]["line_count"] == 3


def test_workspace_ops_list_delete_and_tree(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("workspace-ops")
    assert skill is not None

    ctx = _ctx(str(runtime.workspace.workspace_root))
    runtime.execute_tool_call(
        "create_file",
        {"filepath": "src/app.py", "content": "print('hi')\n"},
        selected=[skill],
        ctx=ctx,
    )

    listed = runtime.execute_tool_call(
        "list_files",
        {"path": "src"},
        selected=[skill],
        ctx=ctx,
    )
    assert listed["ok"] is True
    assert listed["data"]["count"] == 1
    assert listed["data"]["files"] == ["app.py"]

    tree = runtime.execute_tool_call(
        "workspace_tree",
        {"max_depth": 3},
        selected=[skill],
        ctx=ctx,
    )
    assert tree["ok"] is True
    assert "src/" in tree["data"]["tree"]
    assert "app.py" in tree["data"]["tree"]

    deleted = runtime.execute_tool_call(
        "delete_file",
        {"filepath": "src/app.py"},
        selected=[skill],
        ctx=ctx,
    )
    assert deleted["ok"] is True
    assert deleted["data"]["deleted"] is True
    assert deleted["data"]["size_bytes"] > 0
    assert deleted["data"]["kind"] == "file"


def test_workspace_ops_delete_file_supports_binary_files(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("workspace-ops")
    assert skill is not None

    binary_path = runtime.workspace.workspace_root / ".DS_Store"
    binary_path.write_bytes(b"\x00\x87binary")

    ctx = _ctx(str(runtime.workspace.workspace_root))
    deleted = runtime.execute_tool_call(
        "delete_file",
        {"filepath": ".DS_Store"},
        selected=[skill],
        ctx=ctx,
    )
    assert deleted["ok"] is True
    assert deleted["data"]["deleted"] is True
    assert deleted["data"]["size_bytes"] == 8
    assert deleted["data"]["kind"] == "file"


def test_workspace_ops_delete_path_supports_directories(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("workspace-ops")
    assert skill is not None

    ctx = _ctx(str(runtime.workspace.workspace_root))
    runtime.execute_tool_call(
        "create_file",
        {"filepath": "site/assets/a.txt", "content": "alpha"},
        selected=[skill],
        ctx=ctx,
    )

    non_recursive = runtime.execute_tool_call(
        "delete_path",
        {"path": "site", "recursive": False},
        selected=[skill],
        ctx=ctx,
    )
    assert non_recursive["ok"] is False

    recursive = runtime.execute_tool_call(
        "delete_path",
        {"path": "site", "recursive": True},
        selected=[skill],
        ctx=ctx,
    )
    assert recursive["ok"] is True
    assert recursive["data"]["deleted"] is True
    assert recursive["data"]["kind"] == "directory"
    assert recursive["data"]["recursive"] is True
    assert recursive["data"]["file_count"] == 1


def test_workspace_ops_delete_path_handles_empty_directory(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("workspace-ops")
    assert skill is not None

    empty_dir = runtime.workspace.workspace_root / "empty-dir"
    empty_dir.mkdir(parents=True)

    ctx = _ctx(str(runtime.workspace.workspace_root))
    deleted = runtime.execute_tool_call(
        "delete_path",
        {"path": "empty-dir"},
        selected=[skill],
        ctx=ctx,
    )
    assert deleted["ok"] is True
    assert deleted["data"]["kind"] == "directory"
    assert deleted["data"]["file_count"] == 0
