from __future__ import annotations


def select_skills(runtime, ctx, top_n: int = 3):
    loaded = [
        skill for skill in runtime.skills_by_ids(list(getattr(ctx, "loaded_skill_ids", []) or [])) if not skill.disable_model_invocation
    ]
    return loaded[: max(1, int(top_n))]
