from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def selection_tokens(*values: Any) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            tokens.update(selection_tokens(*value))
            continue
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{1,}", str(value or "").strip().lower()):
            tokens.add(token)
            tokens.update(part for part in re.split("[_-]", token) if len(part) > 1)
    return tokens


def skill_selection_score(skill, ctx) -> int:
    score = 1000 if skill.id.lower() == str(getattr(ctx, "explicit_skill_id", "")).strip().lower() else 0
    sticky_ids = {str(item).strip().lower() for item in getattr(ctx, "sticky_skill_ids", []) or [] if str(item).strip()}
    if skill.id.lower() in sticky_ids:
        score += 250
    skill_tokens = selection_tokens(skill.id, skill.description, skill.tags, skill.categories, skill.produces, skill.allowed_tools)
    score += 4 * len(selection_tokens(getattr(ctx, "user_input", "")) & skill_tokens)
    score += 2 * len(selection_tokens(getattr(ctx, "branch_labels", []) or []) & skill_tokens)
    score += 2 * len(selection_tokens(*(Path(item).name for item in getattr(ctx, "attachments", []) or [])) & skill_tokens)
    score += len(selection_tokens(getattr(ctx, "recent_routing_hint", "")) & skill_tokens)
    return score


def select_skills(runtime, ctx, top_n: int = 3):
    loaded = [
        skill for skill in runtime.skills_by_ids(list(getattr(ctx, "loaded_skill_ids", []) or [])) if not skill.disable_model_invocation
    ]
    limit = max(1, int(top_n))
    if len(loaded) <= 1:
        return loaded[:limit]
    scored = [(skill_selection_score(skill, ctx), index, skill) for index, skill in enumerate(loaded)]
    if not any(score > 0 for score, _index, _skill in scored):
        return loaded[:limit]
    scored.sort(key=lambda item: (-item[0], item[1], item[2].id))
    return [skill for _score, _index, skill in scored[:limit]]
