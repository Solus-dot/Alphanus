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
