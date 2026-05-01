from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class SkillSelector:
    def __init__(self, runtime) -> None:
        self.runtime = runtime

    @staticmethod
    def selection_tokens(*values: Any) -> set[str]:
        tokens: set[str] = set()
        for value in values:
            if isinstance(value, (list, tuple, set)):
                tokens.update(SkillSelector.selection_tokens(*list(value)))
                continue
            text = str(value or "").strip().lower()
            if not text:
                continue
            for token in re.findall(r"[a-z0-9][a-z0-9_-]{1,}", text):
                tokens.add(token)
                if "_" in token:
                    tokens.update(part for part in token.split("_") if len(part) > 1)
                if "-" in token:
                    tokens.update(part for part in token.split("-") if len(part) > 1)
        return tokens

    def skill_selection_score(self, skill, ctx) -> int:
        score = 0
        explicit_skill_id = str(getattr(ctx, "explicit_skill_id", "")).strip().lower()
        if explicit_skill_id and skill.id.lower() == explicit_skill_id:
            score += 1000
        sticky_ids = {str(item).strip().lower() for item in getattr(ctx, "sticky_skill_ids", []) or [] if str(item).strip()}
        if skill.id.lower() in sticky_ids:
            score += 250

        user_tokens = self.selection_tokens(getattr(ctx, "user_input", ""))
        branch_tokens = self.selection_tokens(getattr(ctx, "branch_labels", []) or [])
        attachment_tokens = self.selection_tokens(*(Path(item).name for item in (getattr(ctx, "attachments", []) or [])))
        recent_tokens = self.selection_tokens(getattr(ctx, "recent_routing_hint", ""))
        skill_tokens = self.selection_tokens(
            skill.id,
            skill.name,
            skill.description,
            skill.tags,
            skill.categories,
            skill.produces,
            skill.allowed_tools,
            [entry.name for entry in self.runtime._skill_entrypoints(skill)],
        )

        score += 4 * len(user_tokens & skill_tokens)
        score += 2 * len(branch_tokens & skill_tokens)
        score += 2 * len(attachment_tokens & skill_tokens)
        score += 1 * len(recent_tokens & skill_tokens)
        return score

    def select_skills(self, ctx, top_n: int = 3):
        loaded = [
            skill
            for skill in self.runtime.skills_by_ids(list(getattr(ctx, "loaded_skill_ids", []) or []))
            if not skill.disable_model_invocation
        ]
        if not loaded:
            return []
        limit = max(1, int(top_n))
        if len(loaded) <= 1:
            return loaded[:limit]

        scored = [(self.skill_selection_score(skill, ctx), idx, skill) for idx, skill in enumerate(loaded)]
        if not any(score > 0 for score, _idx, _skill in scored):
            return loaded[:limit]
        scored.sort(key=lambda item: (-item[0], item[1], item[2].id))
        return [skill for _score, _idx, skill in scored[:limit]]
