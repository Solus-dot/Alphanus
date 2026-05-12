from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.skill_parser import SkillManifest


class SkillRegistry:
    @staticmethod
    def remove_skill_tools(tool_registry: dict[str, Any], skill_id: str) -> None:
        for tool_name, reg in list(tool_registry.items()):
            if getattr(reg, "skill_id", "") == skill_id:
                tool_registry.pop(tool_name, None)

    @staticmethod
    def register_tool(
        *,
        tool_registry: dict[str, Any],
        registered_tool_cls,
        tool_name: str,
        manifest: SkillManifest,
        tool_scope_for_name: Callable[[str], str],
        append_unique: Callable[[list[str], str], None],
        spec: dict[str, Any],
        extra: dict[str, Any],
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
