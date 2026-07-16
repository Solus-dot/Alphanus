from __future__ import annotations

import logging
from pathlib import Path

from skills.skill_parser import SKILL_DOC


class SkillDiscovery:
    @staticmethod
    def discover_skill_roots(skills_dirs: Path | list[Path]) -> list[Path]:
        raw_roots = skills_dirs if isinstance(skills_dirs, list) else [skills_dirs]
        roots: list[Path] = []
        seen: set[Path] = set()
        for root in raw_roots:
            resolved = root.expanduser().resolve()
            if resolved in seen:
                continue
            roots.append(resolved)
            seen.add(resolved)
        return roots

    @staticmethod
    def discover_skill_dirs(root: Path) -> list[Path]:
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
            if not skill_dir.is_relative_to(resolved_root):
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
