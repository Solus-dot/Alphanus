from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


class SkillProcessEnvBuilder:
    @staticmethod
    def build_base_env(
        *,
        workspace_root: Path,
        home_root: Path,
        memory_path: Path,
        python_executable: str,
        skills_dir: Path,
        config: dict[str, Any],
    ) -> dict[str, str]:
        env = os.environ.copy()
        env["ALPHANUS_WORKSPACE_ROOT"] = str(workspace_root)
        env["ALPHANUS_HOME_ROOT"] = str(home_root)
        env["ALPHANUS_MEMORY_PATH"] = str(memory_path)
        env["ALPHANUS_MEMORY_BACKEND"] = "lexical"
        env["ALPHANUS_SKILL_PYTHON"] = str(python_executable)
        env["ALPHANUS_CONFIG_JSON"] = json.dumps(config, ensure_ascii=False)

        repo_root = skills_dir.parent.resolve()
        src_root = (repo_root / "src").resolve()
        path_entries = [str(src_root)] if src_root.exists() else []
        path_entries.append(str(repo_root))
        existing = env.get("PYTHONPATH", "")
        prefix = os.pathsep.join(path_entries)
        env["PYTHONPATH"] = prefix if not existing else prefix + os.pathsep + existing

        npm_path = shutil.which("npm")
        if npm_path:
            try:
                proc = subprocess.run(
                    [npm_path, "root", "-g"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except Exception as exc:
                logging.debug("npm root probe failed: %s", exc)
                proc = None
            node_root = (proc.stdout or "").strip() if proc and proc.returncode == 0 else ""
            if node_root:
                existing_node_path = env.get("NODE_PATH", "")
                env["NODE_PATH"] = node_root if not existing_node_path else node_root + os.pathsep + existing_node_path
        return env
