from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast


class SkillExecutor:
    def __init__(
        self,
        runtime,
        *,
        skills_list_tool_name: str,
        skill_view_tool_name: str,
        request_user_input_tool_name: str,
        run_skill_tool_name: str,
        ok_fn,
        err_fn,
        protocol_error_cls,
    ) -> None:
        self.runtime = runtime
        self._skills_list_tool_name = skills_list_tool_name
        self._skill_view_tool_name = skill_view_tool_name
        self._request_user_input_tool_name = request_user_input_tool_name
        self._run_skill_tool_name = run_skill_tool_name
        self._ok = ok_fn
        self._err = err_fn
        self._protocol_error_cls = protocol_error_cls

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
        if reg.name == self._run_skill_tool_name:
            return self.execute_run_skill_tool(args, env)
        if str(getattr(reg, "command_template", "")).strip():
            return self.execute_registered_command_tool(reg, args, env)
        if reg.module is None and reg.module_path:
            reg.module = runtime._load_module(reg.module_path, reg.module_name or f"alphanus_tools_{reg.skill_id}")
        if reg.module is None or not hasattr(reg.module, "execute"):
            raise self._protocol_error_cls(f"Tool '{reg.name}' has no callable execute() handler")
        return reg.module.execute(reg.name, args, env)

    def execute_registered_command_tool(self, reg, args: dict[str, Any], env) -> dict[str, Any]:
        runtime = self.runtime
        skill = runtime.get_skill(str(getattr(reg, "skill_id", "")).strip())
        if skill is None or not skill.path:
            raise FileNotFoundError("Selected skill root is unavailable")
        template_values: dict[str, Any] = {
            "workspace_root": str(runtime.workspace.workspace_root),
            "skill_root": str(skill.path),
        }
        template_values.update(args)
        command = self.resolve_entrypoint_placeholders(str(reg.command_template), template_values)
        timeout_s = max(1, int(getattr(reg, "timeout_s", 30) or 30))
        cwd_mode = str(getattr(reg, "cwd", "skill")).strip().lower() or "skill"
        command_cwd = str(skill.path) if cwd_mode == "skill" else str(runtime.workspace.workspace_root)
        run_data = self.run_shell_workflow_command(command, env, timeout_s, cwd=command_cwd)
        stdout = str(run_data.get("stdout", "") or "").strip()
        if stdout:
            candidate = stdout.splitlines()[-1].strip()
            try:
                parsed = json.loads(candidate)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                if {"ok", "data", "error"}.issubset(parsed.keys()):
                    return parsed
                parsed.setdefault("skill_id", skill.id)
                parsed.setdefault("tool_name", reg.name)
                parsed.setdefault("command", command)
                return parsed
        return {
            "skill_id": skill.id,
            "tool_name": reg.name,
            "command": command,
            "stdout": run_data.get("stdout", ""),
            "stderr": run_data.get("stderr", ""),
            "returncode": run_data.get("returncode", 0),
            "cwd": run_data.get("cwd", ""),
        }

    def execute_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        selected,
        ctx,
        confirm_shell=None,
        request_user_input=None,
    ) -> dict[str, Any]:
        runtime = self.runtime
        start = time.perf_counter()
        try:
            reg, _owner = runtime._resolve_tool_call(tool_name, selected, ctx=ctx)
            normalized_args = runtime._prepare_tool_args(reg, args, selected, ctx)
            env = runtime.ToolExecutionEnv(
                workspace=runtime.workspace,
                memory=runtime.memory,
                config=runtime.config,
                debug=runtime.debug,
                confirm_shell=confirm_shell,
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
            message = str(exc).strip() or "Action failed"
            return self._err("E_IO", message, int((time.perf_counter() - start) * 1000))
        except Exception as exc:
            message = str(exc) if runtime.debug else "Action failed"
            return self._err("E_IO", message, int((time.perf_counter() - start) * 1000))

    def execute_run_skill_tool(self, args: dict[str, Any], env) -> dict[str, Any]:
        if str(args.get("entrypoint", "")).strip():
            return self.execute_skill_entrypoint_tool(args, env)
        if str(args.get("script", "")).strip():
            return self.execute_skill_script_tool(args, env)
        raise ValueError("run_skill requires an entrypoint or script")

    def execute_skill_script_tool(self, args: dict[str, Any], env) -> dict[str, Any]:
        runtime = self.runtime
        skill = runtime.get_skill(str(args.get("skill_id", "")).strip())
        if skill is None or not skill.path:
            raise FileNotFoundError("Selected skill root is unavailable")

        rel_script = str(args.get("script", "")).strip()
        script_path = (skill.path / rel_script).resolve()
        if not runtime._is_relative_to(script_path, skill.path.resolve()):
            raise PermissionError("Skill script path escapes skill root")
        if not script_path.exists():
            raise FileNotFoundError(f"Skill script not found: {rel_script}")
        ext = script_path.suffix.lower()
        interpreter = runtime._script_interpreter(ext)
        if not interpreter:
            raise PermissionError(f"Unsupported skill script type: {script_path.suffix}")
        if (
            not Path(interpreter[0]).exists()
            and interpreter[0] != Path(sys.executable).resolve().as_posix()
            and shutil.which(interpreter[0]) is None
        ):
            raise FileNotFoundError(f"Missing interpreter for skill script: {interpreter[0]}")
        interpreter_cmd = cast(tuple[str, ...], interpreter)

        raw_argv = args.get("argv")
        argv = [str(item) for item in raw_argv] if isinstance(raw_argv, list) else []
        proc_env = dict(runtime._proc_env_base)
        proc_env["ALPHANUS_SELECTED_SKILL_ID"] = skill.id
        proc_env["ALPHANUS_SKILL_ROOT"] = str(skill.path)
        proc_env["ALPHANUS_SKILL_SCRIPT"] = rel_script
        params_payload = args.get("params")
        if not isinstance(params_payload, dict):
            params_payload = {}
        proc_env["ALPHANUS_TOOL_ARGS_JSON"] = json.dumps(params_payload, ensure_ascii=False)
        proc = subprocess.run(
            list(interpreter_cmd) + [str(script_path)] + argv,
            cwd=str(skill.path),
            capture_output=True,
            text=True,
            input=str(args.get("stdin") or ""),
            timeout=max(1, int(args.get("timeout_s", 30))),
            env=proc_env,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            msg = stderr or stdout or f"Skill script failed with exit code {proc.returncode}"
            lowered = msg.lower()
            if "permissionerror" in lowered or "operation not permitted" in lowered:
                raise PermissionError(msg)
            if "filenotfounderror" in lowered or "no such file or directory" in lowered:
                raise FileNotFoundError(msg)
            raise RuntimeError(msg)

        out = (proc.stdout or "").strip()
        if not out:
            return {
                "skill_id": skill.id,
                "script": rel_script,
                "stdout": "",
            }
        candidate = out.splitlines()[-1].strip()
        try:
            parsed = json.loads(candidate)
        except Exception:
            return {
                "skill_id": skill.id,
                "script": rel_script,
                "stdout": out,
                "stderr": (proc.stderr or "").strip(),
            }
        if isinstance(parsed, dict):
            parsed.setdefault("skill_id", skill.id)
            parsed.setdefault("script", rel_script)
            return parsed
        return {"skill_id": skill.id, "script": rel_script, "value": parsed}

    @staticmethod
    def resolve_entrypoint_placeholders(template: str, values: dict[str, Any]) -> str:
        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in values:
                raise ValueError(f"Missing template value: {key}")
            return shlex.quote(str(values[key]))

        return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, template)

    def run_shell_workflow_command(self, command: str, env, timeout_s: int, cwd: str | None = None) -> dict[str, Any]:
        caps = env.config.get("capabilities", {})
        dangerously_skip_permissions = bool(caps.get("dangerously_skip_permissions", False))
        shell_require_confirmation = bool(caps.get("shell_require_confirmation", True))
        if shell_require_confirmation and not dangerously_skip_permissions:
            if not env.confirm_shell:
                raise PermissionError("Shell confirmation callback is required")
            if not env.confirm_shell(command):
                raise PermissionError("Shell command rejected by user")
        allowed_cwd_roots = [cwd] if cwd else None
        result = env.workspace.run_shell_command(
            command,
            timeout_s=max(1, int(timeout_s)),
            cwd=cwd,
            allowed_cwd_roots=allowed_cwd_roots,
        )
        if not result.get("ok"):
            error = result.get("error") or {}
            message = str(error.get("message", "Shell workflow command failed"))
            code = str(error.get("code", ""))
            if code == "E_POLICY":
                raise PermissionError(message)
            if code == "E_TIMEOUT":
                raise TimeoutError(message)
            raise RuntimeError(message)
        return result["data"]

    def execute_skill_entrypoint_tool(self, args: dict[str, Any], env) -> dict[str, Any]:
        runtime = self.runtime
        skill = runtime.get_skill(str(args.get("skill_id", "")).strip())
        if skill is None or not skill.path:
            raise FileNotFoundError("Selected skill root is unavailable")
        entrypoint_name = str(args.get("entrypoint", "")).strip()
        entrypoint = next((item for item in runtime._skill_entrypoints(skill) if item.name == entrypoint_name), None)
        if entrypoint is None:
            raise FileNotFoundError(f"Skill entrypoint not found: {entrypoint_name}")

        raw_params = args.get("params")
        params = raw_params if isinstance(raw_params, dict) else {}
        template_values: dict[str, Any] = {
            "workspace_root": str(runtime.workspace.workspace_root),
            "skill_root": str(skill.path),
        }
        template_values.update(params)
        timeout_s = max(1, int(args.get("timeout_s", entrypoint.timeout_s)))

        install_results: list[dict[str, Any]] = []
        verify_results: list[dict[str, Any]] = []
        command_cwd = str(skill.path) if entrypoint.cwd == "skill" else str(runtime.workspace.workspace_root)
        for template in entrypoint.install:
            command = self.resolve_entrypoint_placeholders(template, template_values)
            install_results.append(self.run_shell_workflow_command(command, env, timeout_s, cwd=command_cwd))
        for template in entrypoint.verify:
            command = self.resolve_entrypoint_placeholders(template, template_values)
            verify_results.append(self.run_shell_workflow_command(command, env, timeout_s, cwd=command_cwd))
        command = self.resolve_entrypoint_placeholders(entrypoint.command, template_values)
        run_data = self.run_shell_workflow_command(command, env, timeout_s, cwd=command_cwd)
        return {
            "skill_id": skill.id,
            "entrypoint": entrypoint.name,
            "command": command,
            "install_results": install_results,
            "verify_results": verify_results,
            "stdout": run_data.get("stdout", ""),
            "stderr": run_data.get("stderr", ""),
            "returncode": run_data.get("returncode", 0),
            "cwd": run_data.get("cwd", ""),
        }

    def normalize_result(self, result: Any, duration_ms: int) -> dict[str, Any]:
        if isinstance(result, dict) and {"ok", "data", "error"}.issubset(result.keys()):
            out = dict(result)
            raw_meta = out.get("meta")
            meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
            meta["duration_ms"] = int(meta.get("duration_ms", duration_ms))
            out["meta"] = meta
            return out
        return self._ok(result, duration_ms)
