from __future__ import annotations

from pathlib import Path
from typing import Any


class SkillRunValidator:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def validate_run_skill_args(
        self,
        args: dict[str, Any],
        selected: list[Any],
        ctx: Any,
    ) -> dict[str, Any]:
        requested_entrypoint = str(args.get("entrypoint", "")).strip()
        requested_script = str(args.get("script", "")).strip()
        if bool(requested_entrypoint) == bool(requested_script):
            raise ValueError("run_skill requires exactly one of 'entrypoint' or 'script'")
        if requested_entrypoint:
            return self.validate_skill_entrypoint_args(args, selected, ctx)
        return self.validate_skill_script_args(args, selected, ctx)

    def validate_skill_script_args(
        self,
        args: dict[str, Any],
        selected: list[Any],
        ctx: Any,
    ) -> dict[str, Any]:
        runtime = self.runtime
        selected_with_scripts = [
            skill
            for skill in selected
            if runtime._exposed_relevant_skill_scripts(skill, ctx)
            and not skill.disable_model_invocation
            and skill.execution_allowed
        ]
        if not selected_with_scripts:
            raise PermissionError("No selected skills expose runnable bundled scripts")

        requested_skill_id = str(args.get("skill_id", "")).strip()
        if requested_skill_id:
            skill = next((item for item in selected_with_scripts if item.id == requested_skill_id), None)
            if skill is None:
                raise PermissionError(f"Skill '{requested_skill_id}' is not selected or has no runnable scripts")
        elif len(selected_with_scripts) == 1:
            skill = selected_with_scripts[0]
        else:
            skill_ids = ", ".join(skill.id for skill in selected_with_scripts[:4])
            raise ValueError(f"Multiple selected skills expose scripts; specify skill_id ({skill_ids})")

        requested_script = str(args.get("script", "")).strip()
        if not requested_script:
            raise ValueError("Missing required argument: script")
        runnable = runtime._exposed_relevant_skill_scripts(skill, ctx)
        chosen = ""
        if requested_script in runnable:
            chosen = requested_script
        else:
            matches = [rel for rel in runnable if Path(rel).name == requested_script or Path(rel).stem == requested_script]
            if len(matches) == 1:
                chosen = matches[0]
        if not chosen:
            raise PermissionError(f"Script '{requested_script}' is not available for skill '{skill.id}'")

        params = args.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("Invalid 'params': expected object")

        out = dict(args)
        out["skill_id"] = skill.id
        out["script"] = chosen
        out["params"] = params
        timeout_s = out.get("timeout_s")
        if timeout_s is None:
            out["timeout_s"] = 30
        return out

    def validate_skill_entrypoint_args(
        self,
        args: dict[str, Any],
        selected: list[Any],
        ctx: Any,
    ) -> dict[str, Any]:
        runtime = self.runtime
        selected_with_entrypoints = [
            skill
            for skill in selected
            if runtime._exposed_relevant_skill_entrypoints(skill, ctx)
            and not skill.disable_model_invocation
            and skill.execution_allowed
        ]
        if not selected_with_entrypoints:
            raise PermissionError("No selected skills expose runnable entrypoints")

        requested_skill_id = str(args.get("skill_id", "")).strip()
        if requested_skill_id:
            skill = next((item for item in selected_with_entrypoints if item.id == requested_skill_id), None)
            if skill is None:
                raise PermissionError(f"Skill '{requested_skill_id}' is not selected or has no runnable entrypoints")
        elif len(selected_with_entrypoints) == 1:
            skill = selected_with_entrypoints[0]
        else:
            skill_ids = ", ".join(skill.id for skill in selected_with_entrypoints[:4])
            raise ValueError(f"Multiple selected skills expose entrypoints; specify skill_id ({skill_ids})")

        requested_entrypoint = str(args.get("entrypoint", "")).strip()
        if not requested_entrypoint:
            raise ValueError("Missing required argument: entrypoint")
        candidates = runtime._exposed_relevant_skill_entrypoints(skill, ctx)
        entrypoint = next((item for item in candidates if item.name == requested_entrypoint), None)
        if entrypoint is None:
            raise PermissionError(f"Entrypoint '{requested_entrypoint}' is not available for skill '{skill.id}'")

        params = args.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("Invalid 'params': expected object")
        runtime._validate_schema_value("params", params, entrypoint.parameters)

        out = dict(args)
        out["skill_id"] = skill.id
        out["entrypoint"] = entrypoint.name
        out["params"] = params
        if out.get("timeout_s") is None:
            out["timeout_s"] = entrypoint.timeout_s
        return out
