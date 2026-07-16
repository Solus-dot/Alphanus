from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from core.sandbox import MAX_SANDBOX_OUTPUT_BYTES, SandboxCommand, SandboxConfig, SandboxRunner, run_bounded_process


def test_bounded_process_caps_output_and_accepts_stdin(tmp_path: Path) -> None:
    result = run_bounded_process(
        [sys.executable, "-c", "import sys; print(sys.stdin.read()); sys.stderr.write('e' * 25000)"],
        cwd=tmp_path,
        timeout_s=5,
        stdin="hello",
    )

    assert result["stdout"].strip() == "hello"
    assert len(result["stderr"].encode()) == MAX_SANDBOX_OUTPUT_BYTES
    assert result["stderr_truncated"] is True


def test_bounded_process_times_out_process_group(tmp_path: Path) -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        run_bounded_process([sys.executable, "-c", "import time; time.sleep(60)"], cwd=tmp_path, timeout_s=1)


def test_preflight_success_message_does_not_describe_available_backend_as_missing(monkeypatch) -> None:
    monkeypatch.setattr("core.sandbox.shutil.which", lambda name: f"/usr/bin/{name}")

    result = SandboxRunner().preflight(SandboxConfig(mode="project-write", backend="macos-seatbelt"))

    assert result == {"ok": True, "backend": "macos-seatbelt", "message": "sandbox-exec is available"}


def test_linux_backend_unshares_network_only_when_disabled(tmp_path: Path, monkeypatch) -> None:
    runner = SandboxRunner()
    commands: list[list[str]] = []

    monkeypatch.setattr("core.sandbox.shutil.which", lambda name: "/usr/bin/bwrap" if name == "bwrap" else None)

    def fake_run_subprocess(argv: list[str], *, cwd: Path, timeout_s: int):
        commands.append(argv)
        return {
            "argv": argv,
            "stdout": "",
            "stderr": "",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "returncode": 0,
            "cwd": str(cwd),
            "duration_ms": 1,
        }

    monkeypatch.setattr("core.sandbox.run_bounded_process", fake_run_subprocess)

    runner.run(
        SandboxCommand(
            command="true",
            cwd=tmp_path,
            project_root=tmp_path,
            timeout_s=5,
            config=SandboxConfig(mode="project-write", network=False, backend="linux-bubblewrap"),
        )
    )
    runner.run(
        SandboxCommand(
            command="true",
            cwd=tmp_path,
            project_root=tmp_path,
            timeout_s=5,
            config=SandboxConfig(mode="project-write", network=True, backend="linux-bubblewrap"),
        )
    )

    assert "--unshare-net" in commands[0]
    assert "--unshare-net" not in commands[1]
    assert "--unshare-all" not in commands[0]
    assert "--unshare-all" not in commands[1]


def test_macos_profile_does_not_allow_global_file_reads(tmp_path: Path) -> None:
    runner = SandboxRunner()
    profile_path = runner._macos_profile(
        SandboxCommand(
            command="true",
            cwd=tmp_path,
            project_root=tmp_path,
            timeout_s=5,
            config=SandboxConfig(mode="project-write", network=False, backend="macos-seatbelt"),
        )
    )
    try:
        profile = profile_path.read_text(encoding="utf-8")
    finally:
        profile_path.unlink()

    assert "(allow file-read*)" not in profile.splitlines()
    assert f'(allow file-read* (subpath "{tmp_path}"))' in profile
    assert f'(deny file-write* (subpath "{tmp_path}/.git"))' in profile
    assert f'(deny file-read* file-write* (subpath "{tmp_path}/.alphanus"))' in profile


def test_macos_profile_allows_single_git_command_to_manage_git_state(tmp_path: Path) -> None:
    runner = SandboxRunner()
    profile_path = runner._macos_profile(
        SandboxCommand(
            command="git branch feature/test",
            cwd=tmp_path,
            project_root=tmp_path,
            timeout_s=5,
            config=SandboxConfig(mode="project-write", network=False, backend="macos-seatbelt"),
        )
    )
    try:
        profile = profile_path.read_text(encoding="utf-8")
    finally:
        profile_path.unlink()

    assert f'(deny file-write* (subpath "{tmp_path}/.git"))' not in profile
    assert f'(deny file-read* file-write* (subpath "{tmp_path}/.alphanus"))' in profile
