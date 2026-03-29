from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from core.workspace import WorkspaceManager


def test_write_path_traversal_denied(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    mgr = WorkspaceManager(str(ws), home_root=str(home))

    with pytest.raises(PermissionError):
        mgr.create_file("../escape.txt", "x")


def test_absolute_write_escape_denied(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    mgr = WorkspaceManager(str(ws), home_root=str(home))

    with pytest.raises(PermissionError):
        mgr.create_file("/tmp/outside.txt", "x")


def test_restricted_secret_reads_denied(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    ssh = home / ".ssh"
    ssh.mkdir(parents=True)
    (ssh / "id_rsa").write_text("secret", encoding="utf-8")
    ws.mkdir()

    mgr = WorkspaceManager(str(ws), home_root=str(home))
    with pytest.raises(PermissionError):
        mgr.read_file(str(ssh / "id_rsa"))


def test_symlink_escape_denied(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    outside = home / "outside"
    home.mkdir()
    ws.mkdir()
    outside.mkdir()

    link = ws / "link"
    link.symlink_to(outside, target_is_directory=True)

    mgr = WorkspaceManager(str(ws), home_root=str(home))
    with pytest.raises(PermissionError):
        mgr.create_file("link/evil.txt", "boom")


def test_shell_metachar_policy(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = WorkspaceManager(str(ws), home_root=str(home))
    res = mgr.run_shell_command("echo ok; rm -rf /")
    assert res["ok"] is False
    assert res["error"]["code"] == "E_POLICY"


def test_workspace_reads_allowed_outside_home_root(tmp_path: Path):
    home = tmp_path / "home"
    ws_parent = tmp_path / "external"
    ws = ws_parent / "ws"
    home.mkdir()
    ws.mkdir(parents=True)
    target = ws / "notes.txt"
    target.write_text("alpha", encoding="utf-8")

    mgr = WorkspaceManager(str(ws), home_root=str(home))
    assert mgr.read_file(str(target)) == "alpha"
    assert mgr.list_files(str(ws)) == ["notes.txt"]


def test_shell_command_runs_with_argv_not_shell(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = WorkspaceManager(str(ws), home_root=str(home))
    res = mgr.run_shell_command("python3 -c \"print('ok')\"")
    assert res["ok"] is True
    assert res["data"]["stdout"].strip() == "ok"
    assert res["data"]["argv"][0] == "python3"
    assert res["meta"]["workspace_changed"] is False


def test_shell_command_nonzero_exit_is_reported_as_failure(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = WorkspaceManager(str(ws), home_root=str(home))
    res = mgr.run_shell_command("python3 -c \"raise SystemExit(7)\"")
    assert res["ok"] is False
    assert res["error"]["code"] == "E_SHELL"
    assert "code 7" in res["error"]["message"]
    assert res["data"]["returncode"] == 7


def test_shell_command_allows_semicolon_inside_quoted_python_arg(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = WorkspaceManager(str(ws), home_root=str(home))
    res = mgr.run_shell_command("python3 -c \"import sys; print('ok')\"")
    assert res["ok"] is True
    assert res["data"]["stdout"].strip() == "ok"


def test_shell_wrapper_with_dash_c_is_rejected(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = WorkspaceManager(str(ws), home_root=str(home))
    res = mgr.run_shell_command("bash -lc 'echo ok'")
    assert res["ok"] is False
    assert res["error"]["code"] == "E_POLICY"


def test_shell_command_runs_in_requested_cwd(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    subdir = ws / "nested"
    home.mkdir()
    subdir.mkdir(parents=True)

    mgr = WorkspaceManager(str(ws), home_root=str(home))
    res = mgr.run_shell_command("python3 -c \"import os; print(os.getcwd())\"", cwd=str(subdir))
    assert res["ok"] is True
    assert res["data"]["stdout"].strip() == str(subdir.resolve())
    assert res["data"]["cwd"] == str(subdir.resolve())


def test_shell_command_known_mutator_sets_workspace_changed_true(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = WorkspaceManager(str(ws), home_root=str(home))
    res = mgr.run_shell_command("mkdir demo")

    assert res["ok"] is True
    assert res["meta"]["workspace_changed"] is True
    assert (ws / "demo").is_dir()


def test_shell_command_ambiguous_command_uses_git_snapshot_when_repo_present(tmp_path: Path):
    git_path = shutil.which("git")
    if not git_path:
        pytest.skip("git is not available")

    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    subprocess.run([git_path, "-C", str(ws), "init"], check=True, capture_output=True, text=True)

    mgr = WorkspaceManager(str(ws), home_root=str(home))
    res = mgr.run_shell_command(
        'python3 -c "from pathlib import Path; Path(\'note.txt\').write_text(\'x\', encoding=\'utf-8\')"'
    )

    assert res["ok"] is True
    assert res["meta"]["workspace_changed"] is True
    assert (ws / "note.txt").read_text(encoding="utf-8") == "x"


def test_shell_command_detects_changes_to_existing_untracked_file_in_git_repo(tmp_path: Path):
    git_path = shutil.which("git")
    if not git_path:
        pytest.skip("git is not available")

    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    subprocess.run([git_path, "-C", str(ws), "init"], check=True, capture_output=True, text=True)
    scratch = ws / "scratch.txt"
    scratch.write_text("before", encoding="utf-8")

    mgr = WorkspaceManager(str(ws), home_root=str(home))
    res = mgr.run_shell_command(
        'python3 -c "from pathlib import Path; Path(\'scratch.txt\').write_text(\'after\', encoding=\'utf-8\')"'
    )

    assert res["ok"] is True
    assert res["meta"]["workspace_changed"] is True
    assert scratch.read_text(encoding="utf-8") == "after"


def test_shell_command_detects_git_branch_creation_in_git_repo(tmp_path: Path):
    git_path = shutil.which("git")
    if not git_path:
        pytest.skip("git is not available")

    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    subprocess.run([git_path, "-C", str(ws), "init"], check=True, capture_output=True, text=True)
    (ws / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run([git_path, "-C", str(ws), "add", "tracked.txt"], check=True, capture_output=True, text=True)
    subprocess.run(
        [
            git_path,
            "-C",
            str(ws),
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "init",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    mgr = WorkspaceManager(str(ws), home_root=str(home))
    res = mgr.run_shell_command("git branch feature/test")

    assert res["ok"] is True
    assert res["meta"]["workspace_changed"] is True


def test_shell_command_detects_ignored_output_changes_in_git_repo(tmp_path: Path):
    git_path = shutil.which("git")
    if not git_path:
        pytest.skip("git is not available")

    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    subprocess.run([git_path, "-C", str(ws), "init"], check=True, capture_output=True, text=True)
    (ws / ".gitignore").write_text("dist/\n", encoding="utf-8")

    mgr = WorkspaceManager(str(ws), home_root=str(home))
    res = mgr.run_shell_command(
        'python3 -c "from pathlib import Path; Path(\'dist\').mkdir(exist_ok=True); '
        'Path(\'dist/out.txt\').write_text(\'x\', encoding=\'utf-8\')"'
    )

    assert res["ok"] is True
    assert res["meta"]["workspace_changed"] is True
    assert (ws / "dist" / "out.txt").read_text(encoding="utf-8") == "x"


def test_move_path_renames_workspace_file(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    source = ws / "index.html"
    source.write_text("<h1>hello</h1>\n", encoding="utf-8")

    mgr = WorkspaceManager(str(ws), home_root=str(home))
    moved = mgr.move_path("index.html", "site/index.html")

    assert moved == str((ws / "site" / "index.html").resolve())
    assert not source.exists()
    assert (ws / "site" / "index.html").read_text(encoding="utf-8") == "<h1>hello</h1>\n"


def test_delete_workspace_root_denied_for_dot_path(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = WorkspaceManager(str(ws), home_root=str(home))
    with pytest.raises(PermissionError):
        mgr.delete_path(".", recursive=True)


def test_delete_workspace_root_denied_for_empty_path(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = WorkspaceManager(str(ws), home_root=str(home))
    with pytest.raises(PermissionError):
        mgr.delete_path("", recursive=True)


def test_delete_alphanus_state_denied(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    state_dir = ws / ".alphanus" / "sessions"
    home.mkdir()
    state_dir.mkdir(parents=True)

    mgr = WorkspaceManager(str(ws), home_root=str(home))
    with pytest.raises(PermissionError):
        mgr.delete_path(".alphanus", recursive=True)
