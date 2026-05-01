from __future__ import annotations

import io
import json
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


def test_workspace_ops_edit_file_supports_section_scoped_replacement(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("workspace-ops")
    assert skill is not None

    ctx = _ctx(str(runtime.workspace.workspace_root))
    runtime.execute_tool_call(
        "create_file",
        {"filepath": "notes.txt", "content": "head\nvalue=1\nmid\nvalue=1\n"},
        selected=[skill],
        ctx=ctx,
    )

    edited = runtime.execute_tool_call(
        "edit_file",
        {
            "filepath": "notes.txt",
            "old_string": "value=1",
            "new_string": "value=2",
            "start_line": 1,
            "end_line": 2,
        },
        selected=[skill],
        ctx=ctx,
    )

    assert edited["ok"] is True
    assert edited["data"]["section_scoped"] is True
    assert edited["data"]["resolved_start_line"] == 1
    assert edited["data"]["resolved_end_line"] == 2
    assert edited["data"]["replacements_applied"] == 1
    assert runtime.workspace.read_file("notes.txt") == "head\nvalue=2\nmid\nvalue=1\n"


def test_workspace_ops_edit_file_supports_section_scoped_regex_replacement(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("workspace-ops")
    assert skill is not None

    ctx = _ctx(str(runtime.workspace.workspace_root))
    runtime.execute_tool_call(
        "create_file",
        {"filepath": "notes.txt", "content": "A=1\nA=2\nA=3\n"},
        selected=[skill],
        ctx=ctx,
    )

    edited = runtime.execute_tool_call(
        "edit_file",
        {
            "filepath": "notes.txt",
            "regex_pattern": r"A=\d",
            "regex_replacement": "A=9",
            "start_line": 2,
            "end_line": 3,
            "regex_replace_all": True,
        },
        selected=[skill],
        ctx=ctx,
    )

    assert edited["ok"] is True
    assert edited["data"]["edit_mode"] == "regex_replace_all"
    assert edited["data"]["section_scoped"] is True
    assert edited["data"]["replacements_applied"] == 2
    assert runtime.workspace.read_file("notes.txt") == "A=1\nA=9\nA=9\n"


def test_workspace_ops_edit_file_rejects_mixed_edit_modes(tmp_path: Path):
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
        {
            "filepath": "notes.txt",
            "content": "alpha\ngamma\n",
            "old_string": "beta",
            "new_string": "gamma",
        },
        selected=[skill],
        ctx=ctx,
    )

    assert edited["ok"] is False
    assert edited["error"]["message"] == (
        "edit_file requires exactly one edit mode: content, old_string/new_string, or regex_pattern/regex_replacement"
    )


def test_workspace_ops_create_directory_and_create_file(tmp_path: Path):
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

    created_file = runtime.execute_tool_call(
        "create_file",
        {"filepath": "site/index.html", "content": "<!doctype html>"},
        selected=[skill],
        ctx=ctx,
    )
    assert created_file["ok"] is True
    assert (runtime.workspace.workspace_root / "site" / "index.html").exists()
    runtime.execute_tool_call(
        "create_file",
        {"filepath": "site/script.js", "content": "console.log('hi')\n"},
        selected=[skill],
        ctx=ctx,
    )
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
        "delete_path",
        {"path": "src/app.py"},
        selected=[skill],
        ctx=ctx,
    )
    assert deleted["ok"] is True
    assert deleted["data"]["deleted"] is True
    assert deleted["data"]["size_bytes"] > 0
    assert deleted["data"]["kind"] == "file"


def test_workspace_ops_move_path_renames_file(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("workspace-ops")
    assert skill is not None

    ctx = _ctx(str(runtime.workspace.workspace_root))
    created = runtime.execute_tool_call(
        "create_file",
        {"filepath": "index.html", "content": "<h1>hello</h1>\n"},
        selected=[skill],
        ctx=ctx,
    )
    assert created["ok"] is True

    moved = runtime.execute_tool_call(
        "move_path",
        {"source_path": "index.html", "destination_path": "site/index.html"},
        selected=[skill],
        ctx=ctx,
    )

    assert moved["ok"] is True
    assert moved["data"]["moved"] is True
    assert moved["data"]["kind"] == "file"
    assert moved["data"]["destination_path"].endswith("/site/index.html")
    assert not (runtime.workspace.workspace_root / "index.html").exists()
    assert (runtime.workspace.workspace_root / "site" / "index.html").exists()


def test_workspace_ops_delete_path_supports_binary_files(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("workspace-ops")
    assert skill is not None

    binary_path = runtime.workspace.workspace_root / ".DS_Store"
    binary_path.write_bytes(b"\x00\x87binary")

    ctx = _ctx(str(runtime.workspace.workspace_root))
    deleted = runtime.execute_tool_call(
        "delete_path",
        {"path": ".DS_Store"},
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
        "create_file",
        {
            "filepath": "src/app.py",
            "content": "def greet(name):\n    return f'hello {name}'\n",
        },
        selected=[skill],
        ctx=ctx,
    )
    runtime.execute_tool_call(
        "create_file",
        {
            "filepath": "src/util.py",
            "content": "VALUE = '" + ("x" * 80) + "'\n",
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


def test_workspace_ops_read_file_supports_section_and_numbered_output(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("workspace-ops")
    assert skill is not None

    ctx = _ctx(str(runtime.workspace.workspace_root))
    runtime.execute_tool_call(
        "create_file",
        {"filepath": "notes.txt", "content": "alpha\nbeta\ngamma\ndelta"},
        selected=[skill],
        ctx=ctx,
    )

    read = runtime.execute_tool_call(
        "read_file",
        {
            "filepath": "notes.txt",
            "before_anchor": "alpha",
            "after_anchor": "delta",
            "include_line_numbers": True,
        },
        selected=[skill],
        ctx=ctx,
    )
    assert read["ok"] is True
    assert read["data"]["content"] == "2: beta\n3: gamma\n"
    assert read["data"]["section_scoped"] is True
    assert read["data"]["resolved_start_line"] == 2
    assert read["data"]["resolved_end_line"] == 3
    assert read["data"]["before_anchor_line"] == 1
    assert read["data"]["after_anchor_line"] == 4
    assert read["data"]["returned_line_count"] == 2


def test_workspace_ops_section_bounds_can_use_reported_line_count(tmp_path: Path):
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

    full_read = runtime.execute_tool_call(
        "read_file",
        {"filepath": "notes.txt"},
        selected=[skill],
        ctx=ctx,
    )
    assert full_read["ok"] is True
    assert full_read["data"]["line_count"] == 3

    section_read = runtime.execute_tool_call(
        "read_file",
        {"filepath": "notes.txt", "start_line": 3, "end_line": 3},
        selected=[skill],
        ctx=ctx,
    )
    assert section_read["ok"] is True
    assert section_read["data"]["resolved_start_line"] == 3
    assert section_read["data"]["resolved_end_line"] == 3
    assert section_read["data"]["content"] == ""

    edited = runtime.execute_tool_call(
        "edit_file",
        {
            "filepath": "notes.txt",
            "old_string": "beta",
            "new_string": "gamma",
            "start_line": 2,
            "end_line": 3,
        },
        selected=[skill],
        ctx=ctx,
    )
    assert edited["ok"] is True
    assert runtime.workspace.read_file("notes.txt") == "alpha\ngamma\n"


def test_workspace_ops_search_code_supports_context_lines(tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("workspace-ops")
    assert skill is not None

    ctx = _ctx(str(runtime.workspace.workspace_root))
    runtime.execute_tool_call(
        "create_file",
        {
            "filepath": "src/app.py",
            "content": "def prelude():\n    return 1\n\ndef greet(name):\n    return f'hello {name}'\n",
        },
        selected=[skill],
        ctx=ctx,
    )

    search = runtime.execute_tool_call(
        "search_code",
        {"query": "greet(", "path": "src", "glob": "*.py", "before_context": 1, "after_context": 1},
        selected=[skill],
        ctx=ctx,
    )
    assert search["ok"] is True
    assert search["data"]["count"] == 1
    assert search["data"]["before_context"] == 1
    assert search["data"]["after_context"] == 1
    match = search["data"]["results"][0]
    assert match["line"] == "def greet(name):"
    assert match["before_context"] == [{"line_number": 3, "line": ""}]
    assert match["after_context"] == [{"line_number": 5, "line": "    return f'hello {name}'"}]


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


def test_workspace_search_code_uses_direct_rg_without_prelisting_workspace_files(tmp_path: Path, monkeypatch):
    manager = WorkspaceManager(str(tmp_path / "ws"), home_root=str(tmp_path / "home"))
    (manager.workspace_root / "src").mkdir(parents=True)
    (manager.workspace_root / "src" / "app.py").write_text("def greet(name):\n    return f'hello {name}'\n", encoding="utf-8")
    monkeypatch.setattr(
        manager, "_iter_searchable_files", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not pre-list files"))
    )
    monkeypatch.setattr("core.workspace.shutil.which", lambda name: "/usr/bin/rg" if name == "rg" else None)

    match_event = {
        "type": "match",
        "data": {
            "path": {"text": "src/app.py"},
            "lines": {"text": "def greet(name):\n"},
            "line_number": 1,
            "submatches": [{"match": {"text": "greet("}, "start": 4, "end": 10}],
        },
    }

    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.stdout = io.StringIO(json.dumps(match_event) + "\n")
            self.stderr = io.StringIO("")
            self.returncode = 0

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    result = manager.search_code("greet(", path="src", glob="*.py")

    assert result["backend"] == "rg"
    assert result["count"] == 1
    assert result["results"][0]["filepath"].endswith("src/app.py")


def test_workspace_search_code_tolerates_rg_io_exit_with_partial_results(tmp_path: Path, monkeypatch):
    manager = WorkspaceManager(str(tmp_path / "ws"), home_root=str(tmp_path / "home"))
    (manager.workspace_root / "src").mkdir(parents=True)
    (manager.workspace_root / "src" / "app.py").write_text("def greet(name):\n    return f'hello {name}'\n", encoding="utf-8")
    monkeypatch.setattr("core.workspace.shutil.which", lambda name: "/usr/bin/rg" if name == "rg" else None)

    match_event = {
        "type": "match",
        "data": {
            "path": {"text": "src/app.py"},
            "lines": {"text": "def greet(name):\n"},
            "line_number": 1,
            "submatches": [{"match": {"text": "greet("}, "start": 4, "end": 10}],
        },
    }

    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.stdout = io.StringIO(json.dumps(match_event) + "\n")
            self.stderr = io.StringIO("permission denied\n")
            self.returncode = 2

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    result = manager.search_code("greet(", path="src", glob="*.py")

    assert result["backend"] == "rg"
    assert result["count"] == 1
    assert result["results"][0]["filepath"].endswith("src/app.py")


def test_workspace_search_code_reuses_context_lines_for_same_file(tmp_path: Path, monkeypatch):
    manager = WorkspaceManager(str(tmp_path / "ws"), home_root=str(tmp_path / "home"))
    (manager.workspace_root / "src").mkdir(parents=True)
    (manager.workspace_root / "src" / "app.py").write_text(
        "def greet_one(name):\n    return name\n\ndef greet_two(name):\n    return name\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("core.workspace.shutil.which", lambda name: "/usr/bin/rg" if name == "rg" else None)

    match_one = {
        "type": "match",
        "data": {
            "path": {"text": "src/app.py"},
            "lines": {"text": "def greet_one(name):\n"},
            "line_number": 1,
            "submatches": [{"match": {"text": "greet_"}, "start": 4, "end": 10}],
        },
    }
    match_two = {
        "type": "match",
        "data": {
            "path": {"text": "src/app.py"},
            "lines": {"text": "def greet_two(name):\n"},
            "line_number": 4,
            "submatches": [{"match": {"text": "greet_"}, "start": 4, "end": 10}],
        },
    }

    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.stdout = io.StringIO(json.dumps(match_one) + "\n" + json.dumps(match_two) + "\n")
            self.stderr = io.StringIO("")
            self.returncode = 0

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    resolve_calls: list[str] = []
    original_resolve_read_path = manager._resolve_read_path

    def _tracking_resolve(path: str, extra_allowed_roots=None):
        resolved = original_resolve_read_path(path, extra_allowed_roots=extra_allowed_roots)
        if str(path).endswith("src/app.py"):
            resolve_calls.append(str(path))
        return resolved

    monkeypatch.setattr(manager, "_resolve_read_path", _tracking_resolve)
    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    result = manager.search_code("greet_", path="src", glob="*.py", before_context=1, after_context=1)

    assert result["backend"] == "rg"
    assert result["count"] == 2
    assert len(resolve_calls) == 1


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
