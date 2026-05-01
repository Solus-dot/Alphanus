from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from core.skill_parser import SKILL_DOC


class SkillDiscovery:
    @staticmethod
    def discover_skill_roots(skills_dir: Path) -> list[Path]:
        return [skills_dir]

    @staticmethod
    def discover_skill_dirs(root: Path, *, is_relative_to: Callable[[Path, Path], bool]) -> list[Path]:
        if not root.exists():
            return []
        candidates: list[Path] = []
        seen: set[str] = set()
        docs = [root / SKILL_DOC] if (root / SKILL_DOC).exists() else []
        if root.is_dir():
            try:
                docs.extend(sorted(root.rglob(SKILL_DOC)))
            except Exception as exc:
                logging.debug("rglob for SKILL.md failed in %s: %s", root, exc)
                docs = docs or []
        resolved_root = root.resolve()
        for skill_doc in docs:
            skill_dir = skill_doc.parent.resolve()
            if ".git" in skill_dir.parts:
                continue
            if not is_relative_to(skill_dir, resolved_root):
                continue
            rel = skill_doc.relative_to(resolved_root)
            if len(rel.parts) > 5:
                continue
            key = str(skill_dir)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(skill_dir)
        candidates.sort(key=lambda item: (len(item.relative_to(resolved_root).parts), str(item)))
        return candidates
