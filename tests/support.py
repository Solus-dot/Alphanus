from __future__ import annotations

from pathlib import Path
from typing import Any

from core.memory import LexicalMemory
from core.project import ProjectRuntime
from skills.runtime import SkillRuntime


def write_skill(root: Path, skill_id: str, manifest: str, tools: str | None = None) -> Path:
    skill_dir = root / skill_id
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(manifest.strip(), encoding="utf-8")
    if tools is not None:
        (skill_dir / "tools.py").write_text(tools.strip(), encoding="utf-8")
    return skill_dir


def build_skill_runtime(
    tmp_path: Path,
    *,
    manifest: str,
    tools: str,
    skill_id: str = "project-ops",
    config: dict[str, Any] | None = None,
) -> SkillRuntime:
    workspace = tmp_path / "home" / "ws"
    skills = tmp_path / "skills"
    workspace.mkdir(parents=True)
    write_skill(skills, skill_id, manifest, tools)
    return SkillRuntime(
        skills_dir=str(skills),
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(workspace)),
        memory=LexicalMemory(storage_path=str(tmp_path / "memory.sqlite3")),
        config=config,
    )
