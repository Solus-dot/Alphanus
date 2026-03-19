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
    assert "--- notes.txt (before)" in edited["data"]["diff"]
    assert "+++ notes.txt (after)" in edited["data"]["diff"]
    assert "-beta" in edited["data"]["diff"]
    assert "+gamma" in edited["data"]["diff"]

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


def test_workspace_ops_edit_file_supports_localized_replacement(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("workspace-ops")
    assert skill is not None

    ctx = _ctx(str(runtime.workspace.workspace_root))
    runtime.execute_tool_call(
        "create_file",
        {"filepath": "notes.txt", "content": "alpha\nbeta\n"},
        selected=[skill],
        ctx=ctx,
    )

    edited = runtime.execute_tool_call(
        "edit_file",
        {"filepath": "notes.txt", "old_string": "beta", "new_string": "gamma"},
        selected=[skill],
        ctx=ctx,
    )

    assert edited["ok"] is True
    assert edited["data"]["edit_mode"] == "replace_one"
    assert edited["data"]["replacements_applied"] == 1
    assert runtime.workspace.read_file("notes.txt") == "alpha\ngamma\n"


def test_workspace_ops_edit_file_rejects_ambiguous_localized_replacement(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("workspace-ops")
    assert skill is not None

    ctx = _ctx(str(runtime.workspace.workspace_root))
    runtime.execute_tool_call(
        "create_file",
        {"filepath": "notes.txt", "content": "beta\nalpha\nbeta\n"},
        selected=[skill],
        ctx=ctx,
    )

    edited = runtime.execute_tool_call(
        "edit_file",
        {"filepath": "notes.txt", "old_string": "beta", "new_string": "gamma"},
        selected=[skill],
        ctx=ctx,
    )

    assert edited["ok"] is False
    assert edited["error"]["message"] == (
        "edit_file old_string matched multiple locations; provide a more specific old_string or set replace_all=true"
    )


def test_workspace_ops_edit_file_supports_replace_all(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("workspace-ops")
    assert skill is not None

    ctx = _ctx(str(runtime.workspace.workspace_root))
    runtime.execute_tool_call(
        "create_file",
        {"filepath": "notes.txt", "content": "beta\nalpha\nbeta\n"},
        selected=[skill],
        ctx=ctx,
    )

    edited = runtime.execute_tool_call(
        "edit_file",
        {"filepath": "notes.txt", "old_string": "beta", "new_string": "gamma", "replace_all": True},
        selected=[skill],
        ctx=ctx,
    )

    assert edited["ok"] is True
    assert edited["data"]["edit_mode"] == "replace_all"
    assert edited["data"]["replacements_applied"] == 2
    assert runtime.workspace.read_file("notes.txt") == "gamma\nalpha\ngamma\n"


def test_workspace_ops_create_directory_and_create_files(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("workspace-ops")
    assert skill is not None

    ctx = _ctx(str(runtime.workspace.workspace_root))
    created_dir = runtime.execute_tool_call(
        "create_directory",
        {"path": "site/assets"},
        selected=[skill],
        ctx=ctx,
    )
    assert created_dir["ok"] is True
    assert created_dir["data"]["kind"] == "directory"

    created_files = runtime.execute_tool_call(
        "create_files",
        {
            "files": [
                {"filepath": "site/index.html", "content": "<!doctype html>"},
                {"filepath": "site/script.js", "content": "console.log('hi')\n"},
            ]
        },
        selected=[skill],
        ctx=ctx,
    )
    assert created_files["ok"] is True
    assert created_files["data"]["count"] == 2
    assert (runtime.workspace.workspace_root / "site" / "index.html").exists()
    assert (runtime.workspace.workspace_root / "site" / "script.js").exists()


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


def test_workspace_ops_accepts_workspace_root_prefixed_relative_paths(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("workspace-ops")
    assert skill is not None

    nested = runtime.workspace.workspace_root / "nested"
    nested.mkdir(parents=True)
    (nested / "file.txt").write_text("alpha", encoding="utf-8")

    ctx = _ctx(str(runtime.workspace.workspace_root))
    prefixed = f"{runtime.workspace.workspace_root.name}/nested"
    deleted = runtime.execute_tool_call(
        "delete_path",
        {"path": prefixed, "recursive": True},
        selected=[skill],
        ctx=ctx,
    )

    assert deleted["ok"] is True
    assert deleted["data"]["kind"] == "directory"
    assert not nested.exists()


def test_workspace_ops_accepts_workspace_root_prefixed_absolute_like_paths(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("workspace-ops")
    assert skill is not None

    nested = runtime.workspace.workspace_root / "nested"
    nested.mkdir(parents=True)
    (nested / "file.txt").write_text("alpha", encoding="utf-8")

    ctx = _ctx(str(runtime.workspace.workspace_root))
    prefixed = f"/{runtime.workspace.workspace_root.name}/nested"
    deleted = runtime.execute_tool_call(
        "delete_path",
        {"path": prefixed, "recursive": True},
        selected=[skill],
        ctx=ctx,
    )

    assert deleted["ok"] is True
    assert deleted["data"]["kind"] == "directory"
    assert not nested.exists()


def test_workspace_ops_read_files_search_code_and_run_checks(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("workspace-ops")
    assert skill is not None

    ctx = _ctx(str(runtime.workspace.workspace_root))
    runtime.execute_tool_call(
        "create_files",
        {
            "files": [
                {
                    "filepath": "src/app.py",
                    "content": "def greet(name):\n    return f'hello {name}'\n",
                },
                {
                    "filepath": "src/util.py",
                    "content": "VALUE = '" + ("x" * 80) + "'\n",
                },
            ]
        },
        selected=[skill],
        ctx=ctx,
    )

    search = runtime.execute_tool_call(
        "search_code",
        {"query": "greet(", "path": "src", "glob": "*.py"},
        selected=[skill],
        ctx=ctx,
    )
    assert search["ok"] is True
    assert search["data"]["count"] == 1
    assert search["data"]["results"][0]["filepath"].endswith("src/app.py")
    assert search["data"]["results"][0]["line_number"] == 1

    read_many = runtime.execute_tool_call(
        "read_files",
        {"paths": ["src/app.py", "src/util.py"], "max_chars_per_file": 20},
        selected=[skill],
        ctx=ctx,
    )
    assert read_many["ok"] is True
    assert read_many["data"]["count"] == 2
    assert read_many["data"]["files"][0]["truncated"] is True
    assert read_many["data"]["files"][0]["returned_chars"] == 20

    checks = runtime.execute_tool_call(
        "run_checks",
        {"command": "pytest", "args": ["--version"]},
        selected=[skill],
        ctx=ctx,
    )
    assert checks["ok"] is True
    assert checks["data"]["passed"] is True
    assert "pytest" in checks["data"]["stdout"].lower()


def test_workspace_ops_search_code_skips_blocked_home_descendants(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("workspace-ops")
    assert skill is not None

    home = runtime.workspace.home_root
    (home / ".ssh").mkdir(parents=True, exist_ok=True)
    (home / ".ssh" / "id_rsa").write_text("topsecret\n", encoding="utf-8")
    (home / ".env").write_text("API_KEY=topsecret\n", encoding="utf-8")
    (home / "notes.txt").write_text("topsecret is only here\n", encoding="utf-8")

    ctx = _ctx(str(runtime.workspace.workspace_root))
    search = runtime.execute_tool_call(
        "search_code",
        {"query": "topsecret", "path": str(home)},
        selected=[skill],
        ctx=ctx,
    )
    assert search["ok"] is True
    assert search["data"]["count"] == 1
    assert search["data"]["results"][0]["filepath"].endswith("notes.txt")
    assert ".ssh" not in search["data"]["results"][0]["filepath"]
    assert ".env" not in search["data"]["results"][0]["filepath"]


def test_workspace_ops_run_checks_rejects_non_verification_commands(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("workspace-ops")
    assert skill is not None

    ctx = _ctx(str(runtime.workspace.workspace_root))
    out = runtime.execute_tool_call(
        "run_checks",
        {"command": "python3", "args": ["-c", "print('hi')"]},
        selected=[skill],
        ctx=ctx,
    )
    assert out["ok"] is False
    assert out["error"]["code"] == "E_POLICY"
    assert out["error"]["message"] == "run_checks only supports approved verification runners"
