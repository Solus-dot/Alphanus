from __future__ import annotations

import subprocess
import time
from typing import Any, cast


class SkillExecutor:
    def __init__(
        self,
        runtime,
        *,
        skills_list_tool_name: str,
        skill_view_tool_name: str,
        request_user_input_tool_name: str,
        ok_fn,
        err_fn,
        protocol_error_cls,
    ) -> None:
        self.runtime = runtime
        self._skills_list_tool_name = skills_list_tool_name
        self._skill_view_tool_name = skill_view_tool_name
        self._request_user_input_tool_name = request_user_input_tool_name
        self._ok = ok_fn
        self._err = err_fn
        self._protocol_error_cls = protocol_error_cls

    def _require_bundled_executable_skill(self, skill) -> None:
        """Reject executable code whose target is outside the reviewed bundle."""
        if skill is None or not skill.path:
            raise FileNotFoundError("Selected skill root is unavailable")
        if not skill.execution_allowed:
            raise PermissionError(f"Executable user skill '{skill.id}' is disabled; only reviewed bundled skills may execute")

    def execute_registered_tool(self, reg, args: dict[str, Any], env, ctx) -> Any:
        runtime = self.runtime
        if reg.name == self._skills_list_tool_name:
            return runtime.skills_list()
        if reg.name == self._skill_view_tool_name:
            return runtime.skill_view(str(args.get("name", "")).strip(), str(args.get("file_path", "")).strip(), ctx)
        if reg.name == self._request_user_input_tool_name:
            if not env.request_user_input:
                raise PermissionError("User input runtime is unavailable")
            return env.request_user_input(args)
        skill = runtime.get_skill(str(getattr(reg, "skill_id", "")).strip())
        if skill is not None:
            self._require_bundled_executable_skill(skill)
        if reg.module is None and reg.module_path:
            reg.module = runtime._load_module(reg.module_path, reg.module_name or f"alphanus_tools_{reg.skill_id}")
        if reg.module is None or not hasattr(reg.module, "execute"):
            raise self._protocol_error_cls(f"Tool '{reg.name}' has no callable execute() handler")
        return reg.module.execute(reg.name, args, env)

    def execute_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        selected,
        ctx,
        request_approval=None,
        request_user_input=None,
    ) -> dict[str, Any]:
        runtime = self.runtime
        start = time.perf_counter()
        try:
            reg, _owner = runtime._resolve_tool_call(tool_name, selected, ctx=ctx)
            normalized_args = runtime._prepare_tool_args(reg, args, selected)
            env = runtime.ToolExecutionEnv(
                project=runtime.project,
                memory=runtime.memory,
                config=runtime.config,
                debug=runtime.debug,
                request_approval=request_approval,
                request_user_input=request_user_input,
            )
            result = self.execute_registered_tool(reg, normalized_args, env, ctx)
            duration = int((time.perf_counter() - start) * 1000)
            return self.normalize_result(result, duration)
        except LookupError as exc:
            return self._err("E_UNSUPPORTED", str(exc), int((time.perf_counter() - start) * 1000))
        except ValueError as exc:
            return self._err("E_VALIDATION", str(exc), int((time.perf_counter() - start) * 1000))
        except FileNotFoundError as exc:
            return self._err("E_NOT_FOUND", str(exc), int((time.perf_counter() - start) * 1000))
        except PermissionError as exc:
            return self._err("E_POLICY", str(exc), int((time.perf_counter() - start) * 1000))
        except (TimeoutError, subprocess.TimeoutExpired) as exc:
            return self._err("E_TIMEOUT", str(exc), int((time.perf_counter() - start) * 1000))
        except self._protocol_error_cls as exc:
            return self._err("E_PROTOCOL", str(exc), int((time.perf_counter() - start) * 1000))
        except RuntimeError as exc:
            message = str(exc).strip() or "Tool raised RuntimeError"
            return self._err("E_IO", message, int((time.perf_counter() - start) * 1000))
        except Exception as exc:
            detail = str(exc).strip()
            message = f"Tool raised {exc.__class__.__name__}"
            if detail:
                message = f"{message}: {detail}"
            return self._err("E_IO", message, int((time.perf_counter() - start) * 1000))

    def normalize_result(self, result: Any, duration_ms: int) -> dict[str, Any]:
        if isinstance(result, dict) and {"ok", "data", "error"}.issubset(result.keys()):
            out = dict(result)
            raw_meta = out.get("meta")
            meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
            meta["duration_ms"] = int(meta.get("duration_ms", duration_ms))
            out["meta"] = meta
            return out
        return cast(dict[str, Any], self._ok(result, duration_ms))
