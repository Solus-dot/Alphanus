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
