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
    ) -> dict[str, Any]:
        requested_entrypoint = str(args.get("entrypoint", "")).strip()
        requested_script = str(args.get("script", "")).strip()
        if bool(requested_entrypoint) == bool(requested_script):
            raise ValueError("run_skill requires exactly one of 'entrypoint' or 'script'")
        if requested_entrypoint:
            return self.validate_skill_entrypoint_args(args, selected)
        return self.validate_skill_script_args(args, selected)

    def validate_skill_script_args(
        self,
        args: dict[str, Any],
        selected: list[Any],
    ) -> dict[str, Any]:
        runtime = self.runtime
        skill = self._select_skill(
            args,
            selected,
            runtime._exposed_relevant_skill_scripts,
            empty="No selected skills expose runnable bundled scripts",
            missing="not selected or has no runnable scripts",
            multiple="scripts",
        )

        requested_script = str(args.get("script", "")).strip()
        if not requested_script:
            raise ValueError("Missing required argument: script")
        runnable = runtime._exposed_relevant_skill_scripts(skill)
        chosen = ""
        if requested_script in runnable:
            chosen = requested_script
        else:
            matches = [rel for rel in runnable if Path(rel).name == requested_script or Path(rel).stem == requested_script]
            if len(matches) == 1:
                chosen = matches[0]
        if not chosen:
            raise PermissionError(f"Script '{requested_script}' is not available for skill '{skill.id}'")

        out = dict(args)
        out["skill_id"] = skill.id
        out["script"] = chosen
        out["params"] = self._params(args)
        if out.get("timeout_s") is None:
            out["timeout_s"] = 30
        return out

    def validate_skill_entrypoint_args(
        self,
        args: dict[str, Any],
        selected: list[Any],
    ) -> dict[str, Any]:
        runtime = self.runtime
        skill = self._select_skill(
            args,
            selected,
            runtime._exposed_relevant_skill_entrypoints,
            empty="No selected skills expose runnable entrypoints",
            missing="not selected or has no runnable entrypoints",
            multiple="entrypoints",
        )

        requested_entrypoint = str(args.get("entrypoint", "")).strip()
        if not requested_entrypoint:
            raise ValueError("Missing required argument: entrypoint")
        candidates = runtime._exposed_relevant_skill_entrypoints(skill)
        entrypoint = next((item for item in candidates if item.name == requested_entrypoint), None)
        if entrypoint is None:
            raise PermissionError(f"Entrypoint '{requested_entrypoint}' is not available for skill '{skill.id}'")

        params = self._params(args)
        runtime._validate_schema_value("params", params, entrypoint.parameters)

        out = dict(args)
        out["skill_id"] = skill.id
        out["entrypoint"] = entrypoint.name
        out["params"] = params
        if out.get("timeout_s") is None:
            out["timeout_s"] = entrypoint.timeout_s
        return out

    @staticmethod
    def _params(args: dict[str, Any]) -> dict[str, Any]:
        params = args.get("params")
        if params is None:
            return {}
        if not isinstance(params, dict):
            raise ValueError("Invalid 'params': expected object")
        return params

    @staticmethod
    def _select_skill(args, selected, exposed, *, empty: str, missing: str, multiple: str):
        candidates = [
            skill
            for skill in selected
            if exposed(skill) and not skill.disable_model_invocation and skill.execution_allowed
        ]
        if not candidates:
            raise PermissionError(empty)
        requested = str(args.get("skill_id", "")).strip()
        if requested:
            skill = next((item for item in candidates if item.id == requested), None)
            if skill is None:
                raise PermissionError(f"Skill '{requested}' is {missing}")
            return skill
        if len(candidates) == 1:
            return candidates[0]
        skill_ids = ", ".join(skill.id for skill in candidates[:4])
        raise ValueError(f"Multiple selected skills expose {multiple}; specify skill_id ({skill_ids})")
