from __future__ import annotations

from core.skill_types import SkillContext, SkillManifest


class SkillToolPolicy:
    def __init__(
        self,
        runtime,
        *,
        core_tool_names: frozenset[str],
        always_available_tool_names: frozenset[str],
        skills_list_tool_name: str,
        skill_view_tool_name: str,
        request_user_input_tool_name: str,
    ) -> None:
        self.runtime = runtime
        self.core_tool_names = core_tool_names
        self.always_available_tool_names = always_available_tool_names
        self.skills_list_tool_name = skills_list_tool_name
        self.skill_view_tool_name = skill_view_tool_name
        self.request_user_input_tool_name = request_user_input_tool_name

    def _allowed_for_permission_mode(self, reg) -> bool:
        mode = self.runtime.permission_mode
        if mode == "danger-full-access":
            return True
        capability = str(reg.capability or "").strip().lower()
        if reg.name in self.always_available_tool_names:
            return True
        if capability == "run_shell_command":
            return mode == "project-write"
        if capability.startswith(("web_", "utility_")):
            return mode == "project-write"
        if capability.startswith("memory_") or mode == "project-write":
            return True
        return capability in {"project_read", "project_tree"} or capability.startswith(("skill_", "web_", "utility_"))

    def is_mutating(self, tool_name: str) -> bool:
        reg = self.runtime.tool_registration(tool_name)
        if reg is None:
            return False
        if reg.mutates is not None:
            return bool(reg.mutates)
        capability = str(reg.capability or "").strip().lower()
        if capability.startswith("project_") and capability != "project_read":
            return True
        return reg.tool_scope == "skill" and (capability.endswith("_runner") or capability == "skill_executor")

    def is_blocked_for_local_project(self, tool_name: str) -> bool:
        reg = self.runtime.tool_registration(tool_name)
        if reg is None:
            normalized = str(tool_name).strip()
            return normalized not in (self.core_tool_names | self.always_available_tool_names | {"shell_command"})
        capability = str(reg.capability or "").strip().lower()
        local_capabilities = (
            "project_",
            "memory_",
            "skill_",
            "local_search",
            "knowledge_",
            "retrieval_",
            "utility_file_search",
        )
        return not (capability.startswith(local_capabilities) or capability in {"run_shell_command", "user_input_requester"})

    def _optional_names(self, selected: list[SkillManifest]) -> list[str]:
        selected_map = {skill.id: skill for skill in selected}
        allowed: list[str] = []
        for tool_name, reg in self.runtime._tool_registry.items():
            if reg.skill_id == "__runtime__" or not self._allowed_for_permission_mode(reg):
                continue
            skill = selected_map.get(reg.skill_id)
            if not skill or skill.disable_model_invocation:
                continue
            if skill.allowed_tools and reg.name not in skill.allowed_tools:
                continue
            allowed.append(tool_name)
        return sorted(allowed)

    def optional_names(self, selected: list[SkillManifest], ctx: SkillContext | None = None) -> list[str]:
        _ = ctx
        return [] if self.runtime.is_read_only_mode() else self._optional_names(selected)

    def allowed_names(self, selected: list[SkillManifest], ctx: SkillContext | None = None) -> list[str]:
        runtime = self.runtime
        if runtime.is_read_only_mode():
            turn_core = {
                name for name in (set(runtime.model_exposed_tool_names()) | set(self._optional_names(selected))) if name in self.core_tool_names
            }
            safe_names = turn_core | {self.skills_list_tool_name, self.skill_view_tool_name}
            if runtime.config_model.runtime.ask_user_tool:
                safe_names.add(self.request_user_input_tool_name)
            return sorted(name for name in safe_names if name in runtime._tool_registry)

        names = set(runtime.model_exposed_tool_names())
        names.update(self.optional_names(selected, ctx=ctx))
        if not runtime.config_model.runtime.ask_user_tool:
            names.discard(self.request_user_input_tool_name)
        return sorted(name for name in names if name in runtime._tool_registry)

    def core_names(self, selected: list[SkillManifest], ctx: SkillContext | None = None) -> list[str]:
        return sorted(name for name in self.allowed_names(selected, ctx=ctx) if name in self.core_tool_names)
