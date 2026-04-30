from __future__ import annotations

from typing import Any, Callable, Dict, List

from core.skill_parser import SkillManifest


class SkillRegistry:
    @staticmethod
    def remove_skill_tools(tool_registry: dict[str, Any], skill_id: str) -> None:
        for tool_name, reg in list(tool_registry.items()):
            if getattr(reg, "skill_id", "") == skill_id:
                tool_registry.pop(tool_name, None)

    @staticmethod
    def rebuild_skill_index(
        enabled_skills: List[SkillManifest],
        *,
        skill_entrypoints: Callable[[SkillManifest], list[Any]],
        skill_runnable_scripts: Callable[[SkillManifest], tuple[str, ...]],
    ) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for skill in enabled_skills:
            index[skill.id] = {
                "id": skill.id,
                "name": skill.name,
                "description": skill.description,
                "tags": list(skill.tags),
                "categories": list(skill.categories),
                "tools": list(skill.allowed_tools),
                "produces": list(skill.produces),
                "entrypoints": [entry.name for entry in skill_entrypoints(skill)],
                "scripts": skill_runnable_scripts(skill),
                "execution_allowed": bool(skill.execution_allowed),
                "adapter": skill.adapter,
                "user_invocable": bool(skill.user_invocable),
                "model_invocable": not bool(skill.disable_model_invocation),
            }
        return index

    @staticmethod
    def register_tool(
        *,
        tool_registry: dict[str, Any],
        registered_tool_cls,
        tool_name: str,
        manifest: SkillManifest,
        tool_scope_for_name: Callable[[str], str],
        append_unique: Callable[[List[str], str], None],
        spec: Dict[str, Any],
        extra: Dict[str, Any],
        soft: bool = False,
    ) -> bool:
        warning_sink = manifest.validation_warnings if soft else manifest.validation_errors
        if tool_name in tool_registry:
            prev = tool_registry[tool_name]
            append_unique(
                warning_sink,
                f"duplicate tool '{tool_name}' already registered by {getattr(prev, 'skill_id', '')}",
            )
            return False

        capability = str(spec.get("capability", "")).strip()
        description = str(spec.get("description", "")).strip()
        parameters = spec.get("parameters")
        if not capability or not description or not isinstance(parameters, dict):
            append_unique(warning_sink, f"invalid tool spec '{tool_name}'")
            return False

        tool_registry[tool_name] = registered_tool_cls(
            name=tool_name,
            skill_id=manifest.id,
            tool_scope=tool_scope_for_name(tool_name),
            capability=capability,
            description=description,
            parameters=parameters,
            **extra,
        )
        return True
