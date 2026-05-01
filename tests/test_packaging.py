from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def _looks_like_network_error(exc: subprocess.CalledProcessError) -> bool:
    text = f"{exc.stdout or ''}\n{exc.stderr or ''}".lower()
    markers = (
        "dns",
        "host",
        "network",
        "could not resolve",
        "failed to download",
        "timed out",
        "connection",
        "temporary failure",
        "name or service not known",
    )
    return any(marker in text for marker in markers)


def test_built_wheel_contains_bundled_skills(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str((tmp_path / ".uv-cache").resolve())

    try:
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        if not _looks_like_network_error(exc):
            raise
        # Offline fallback so packaging validation remains runnable in restricted CI/sandboxes.
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    "--no-build-isolation",
                    ".",
                    "-w",
                    str(tmp_path),
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as pip_exc:
            pytest.skip(f"wheel build unavailable in this environment (uv offline + pip fallback failed): {pip_exc.stderr.strip()}")

    wheel_paths = list(tmp_path.glob("alphanus-*.whl"))
    assert len(wheel_paths) == 1

    with zipfile.ZipFile(wheel_paths[0]) as wheel:
        names = set(wheel.namelist())

    assert "alphanus_bundled/utilities/SKILL.md" in names
    assert "alphanus_bundled/utilities/tools.py" in names
    assert "alphanus_bundled/workspace-ops/SKILL.md" in names
