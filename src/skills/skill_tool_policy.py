from __future__ import annotations

from core.skill_types import SkillContext, SkillManifest

_NETWORK_CAPABILITIES = frozenset({"utility_play_youtube", "utility_weather"})


def _allowed_for_permission_mode(runtime, reg) -> bool:
    mode = runtime.permission_mode
    capability = str(reg.capability or "").strip().lower()
    if (capability.startswith("web_") or capability in _NETWORK_CAPABILITIES) and not runtime.config_model.permissions.network:
        return False
    if mode == "danger-full-access" or reg.name in runtime.always_available_tool_names:
        return True
    if capability == "run_shell_command":
        return mode == "project-write"
    if capability.startswith(("web_", "utility_")):
        return mode == "project-write"
    if capability.startswith("memory_") or mode == "project-write":
        return True
    return capability in {"project_read", "project_tree"} or capability.startswith(("skill_", "web_", "utility_"))


def is_mutating(runtime, tool_name: str) -> bool:
    reg = runtime.tool_registration(tool_name)
    if reg is None:
        return False
    if reg.mutates is not None:
        return bool(reg.mutates)
    capability = str(reg.capability or "").strip().lower()
    return (capability.startswith("project_") and capability != "project_read") or (
        reg.tool_scope == "skill" and (capability.endswith("_runner") or capability == "skill_executor")
    )


def is_blocked_for_local_project(runtime, tool_name: str) -> bool:
    reg = runtime.tool_registration(tool_name)
    if reg is None:
        return str(tool_name).strip() not in (runtime.core_tool_names | runtime.always_available_tool_names | {"shell_command"})
    capability = str(reg.capability or "").strip().lower()
    return not (
        capability.startswith(("project_", "memory_", "skill_", "local_search", "knowledge_", "retrieval_", "utility_file_search"))
        or capability in {"run_shell_command", "user_input_requester"}
    )


def optional_names(runtime, selected: list[SkillManifest], ctx: SkillContext | None = None) -> list[str]:
    _ = ctx
    if runtime.is_read_only_mode():
        return []
    selected_map = {skill.id: skill for skill in selected}
    return sorted(
        name
        for name, reg in runtime._tool_registry.items()
        if reg.skill_id != "__runtime__"
        and _allowed_for_permission_mode(runtime, reg)
        and (skill := selected_map.get(reg.skill_id)) is not None
        and not skill.disable_model_invocation
        and (not skill.allowed_tools or reg.name in skill.allowed_tools)
    )


def allowed_names(runtime, selected: list[SkillManifest], ctx: SkillContext | None = None) -> list[str]:
    if runtime.is_read_only_mode():
        turn_core = {
            name
            for name in (set(runtime.model_exposed_tool_names()) | set(_optional_names(runtime, selected)))
            if name in runtime.core_tool_names
        }
        names = turn_core | {runtime.skills_list_tool_name, runtime.skill_view_tool_name}
        if runtime.config_model.runtime.ask_user_tool:
            names.add(runtime.request_user_input_tool_name)
    else:
        names = set(runtime.model_exposed_tool_names()) | set(optional_names(runtime, selected, ctx))
        if not runtime.config_model.runtime.ask_user_tool:
            names.discard(runtime.request_user_input_tool_name)
    return sorted(name for name in names if name in runtime._tool_registry)


def _optional_names(runtime, selected: list[SkillManifest]) -> list[str]:
    selected_map = {skill.id: skill for skill in selected}
    return [
        name
        for name, reg in runtime._tool_registry.items()
        if reg.skill_id != "__runtime__"
        and _allowed_for_permission_mode(runtime, reg)
        and (skill := selected_map.get(reg.skill_id)) is not None
        and not skill.disable_model_invocation
        and (not skill.allowed_tools or reg.name in skill.allowed_tools)
    ]


def core_names(runtime, selected: list[SkillManifest], ctx: SkillContext | None = None) -> list[str]:
    return sorted(name for name in allowed_names(runtime, selected, ctx) if name in runtime.core_tool_names)
