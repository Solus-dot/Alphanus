from __future__ import annotations

import fcntl
import importlib.util
import os
import pty
import select
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

import pytest

from core.configuration import DEFAULT_CONFIG, save_global_config


@pytest.mark.integration
def test_ratatui_starts_handshakes_and_restores_terminal(tmp_path: Path) -> None:
    extension = importlib.util.find_spec("_alphanus_tui")
    if extension is None or extension.origin is None:
        pytest.skip("Ratatui extension is not built in this source checkout")

    state_root = tmp_path / "state"
    save_global_config(state_root / "config" / "config.toml", DEFAULT_CONFIG)
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
    env = os.environ.copy()
    env.update({"ALPHANUS_APP_ROOT": str(state_root), "TERM": "xterm-256color"})
    process = subprocess.Popen(
        [sys.executable, "-m", "alphanus.cli", "--project-root", str(tmp_path)],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        close_fds=True,
    )
    os.close(slave)
    output = bytearray()
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and process.poll() is None and not (
                b"Alphanus" in output
                and b"Ctrl+F" in output
                and b"ctx" in output
                and b"Recent conversations" in output
        ):
            readable, _, _ = select.select([master], [], [], 0.1)
            if not readable:
                continue
            data = os.read(master, 65_536)
            output.extend(data)
            if b"\x1b[6n" in data:
                os.write(master, b"\x1b[1;1R")
        assert b"Alphanus" in output
        assert b"Alphanus Alpha" not in output
        assert b"Ctrl+F" in output
        assert b"ctx" in output
        assert b"Recent conversations" in output
        os.write(master, b"\x03")
        shutdown_deadline = time.monotonic() + 10
        while time.monotonic() < shutdown_deadline and (process.poll() is None or b"\x1b[?1049l" not in output):
            readable, _, _ = select.select([master], [], [], 0.1)
            if not readable:
                continue
            try:
                data = os.read(master, 65_536)
            except OSError:
                break
            if not data:
                break
            output.extend(data)
        process.wait(timeout=2)
        assert process.returncode == 0
        assert b"\x1b[?1049h" in output
        assert b"\x1b[?1049l" in output
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        os.close(master)
