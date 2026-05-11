from __future__ import annotations

from typing import Any

from core.skill_parser import SkillManifest


class SkillToolSchemaBuilder:
    def __init__(self, runtime: Any, *, run_skill_tool_name: str) -> None:
        self.runtime = runtime
        self.run_skill_tool_name = run_skill_tool_name

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

    def dynamic_run_skill_schema(self, selected: list[SkillManifest], ctx: Any | None) -> dict[str, Any]:
        executable_skills = [
            skill
            for skill in selected
            if not skill.disable_model_invocation
            and (
                self.runtime._exposed_relevant_skill_entrypoints(skill, ctx)
                or self.runtime._exposed_relevant_skill_scripts(skill, ctx)
            )
        ]
        properties: dict[str, Any] = {
            "skill_id": {"type": "string"},
            "entrypoint": {"type": "string"},
            "script": {"type": "string"},
            "params": {"type": "object"},
            "argv": {"type": "array", "items": {"type": "string"}},
            "stdin": {"type": "string"},
            "timeout_s": {"type": "integer"},
        }
        if len(executable_skills) > 1:
            properties["skill_id"] = {"type": "string", "enum": [skill.id for skill in executable_skills]}

        entrypoint_names = sorted(
            dict.fromkeys(
                entrypoint.name
                for skill in executable_skills
                for entrypoint in self.runtime._exposed_relevant_skill_entrypoints(skill, ctx)
            )
        )
        if entrypoint_names:
            properties["entrypoint"] = {"type": "string", "enum": entrypoint_names}

        script_names = sorted(
            dict.fromkeys(
                rel_script for skill in executable_skills for rel_script in self.runtime._exposed_relevant_skill_scripts(skill, ctx)
            )
        )
        if script_names:
            properties["script"] = {"type": "string", "enum": script_names}

        required = ["skill_id"] if len(executable_skills) > 1 else []
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def build(self, names: list[str], selected: list[SkillManifest] | None = None, ctx: Any | None = None) -> list[dict[str, Any]]:
        tools = []
        for name in names:
            reg = self.runtime._tool_registry[name]
            parameters = reg.parameters
            description = reg.description
            if reg.name == self.run_skill_tool_name and selected is not None:
                parameters = self.dynamic_run_skill_schema(selected, ctx)
                available_paths: list[str] = []
                for skill in selected:
                    if skill.disable_model_invocation:
                        continue
                    for entrypoint in self.runtime._exposed_relevant_skill_entrypoints(skill, ctx):
                        available_paths.append(f"{skill.id}:{entrypoint.name}")
                    for rel_script in self.runtime._exposed_relevant_skill_scripts(skill, ctx):
                        available_paths.append(f"{skill.id}:{rel_script}")
                if available_paths:
                    description = (
                        f"{reg.description} Available executable paths: {', '.join(sorted(dict.fromkeys(available_paths))[:8])}."
                    )
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": reg.name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )
        return tools
