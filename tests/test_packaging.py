from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path


def test_built_wheel_contains_bundled_skills(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel_paths = list(tmp_path.glob("alphanus-*.whl"))
    assert len(wheel_paths) == 1

    with zipfile.ZipFile(wheel_paths[0]) as wheel:
        names = set(wheel.namelist())

    assert "alphanus_bundled/utilities/SKILL.md" in names
    assert "alphanus_bundled/utilities/tools.py" in names
    assert "alphanus_bundled/workspace-ops/SKILL.md" in names
