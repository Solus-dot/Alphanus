from __future__ import annotations

from typing import Any

from core.skill_types import SkillManifest


class SkillToolSchemaBuilder:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    @staticmethod
    def cache_key(names: list[str], selected: list[SkillManifest], *, generation: int) -> tuple[Any, ...]:
        selected_ids = tuple(str(getattr(skill, "id", "")).strip() for skill in selected if str(getattr(skill, "id", "")).strip())
        active_skill_ids = tuple(
            sorted(dict.fromkeys(str(getattr(skill, "id", "")).strip() for skill in selected if str(getattr(skill, "id", "")).strip()))
        )
        return (
            generation,
            selected_ids,
            tuple(names),
            active_skill_ids,
        )

    def build(self, names: list[str], selected: list[SkillManifest] | None = None, ctx: Any | None = None) -> list[dict[str, Any]]:
        _ = ctx
        tools = []
        for name in names:
            reg = self.runtime._tool_registry[name]
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": reg.name,
                        "description": reg.description,
                        "parameters": reg.parameters,
                    },
                }
            )
        return tools
