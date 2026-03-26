from __future__ import annotations

from pathlib import Path

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
    with pytest.raises(PermissionError):
        mgr.delete_path(".alphanus/sessions", recursive=True)


def test_delete_symlink_to_protected_target_unlinks_link_only(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    state_dir = ws / ".alphanus"
    home.mkdir()
    state_dir.mkdir(parents=True)
    state_link = ws / "state-link"
    root_link = ws / "root-link"
    state_link.symlink_to(state_dir, target_is_directory=True)
    root_link.symlink_to(ws, target_is_directory=True)

    mgr = WorkspaceManager(str(ws), home_root=str(home))

    assert mgr.delete_path("state-link", recursive=True) == str(state_link)
    assert not state_link.exists()
    assert state_dir.exists()

    assert mgr.delete_path("root-link", recursive=True) == str(root_link)
    assert not root_link.exists()
    assert ws.exists()
