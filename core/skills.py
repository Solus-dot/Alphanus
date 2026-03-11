from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.memory import VectorMemory
from core.skill_parser import SKILL_DOC, SkillManifest, ToolCommandDef, extract_skill_doc, parse_agentskill_manifest
from core.workspace import WorkspaceManager


def _ok(data: Any, duration_ms: int) -> Dict[str, Any]:
    return {"ok": True, "data": data, "error": None, "meta": {"duration_ms": duration_ms}}


def _err(code: str, message: str, duration_ms: int) -> Dict[str, Any]:
    return {
        "ok": False,
        "data": None,
        "error": {"code": code, "message": message},
        "meta": {"duration_ms": duration_ms},
    }


class ToolProtocolError(RuntimeError):
    pass


def _recover_tool_args(raw_args: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw_args, dict):
        return {}
    out = dict(raw_args)
    raw = out.get("_raw")
    if not isinstance(raw, str) or not raw.strip():
        return out
    text = raw.strip()

    parsed: Optional[Dict[str, Any]] = None
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            parsed = value
    except Exception:
        parsed = None

    if parsed is None:
        m = re.search(r'"command"\s*:\s*"((?:\\.|[^"\\])*)"', text)
        if m:
            try:
                command = bytes(m.group(1), "utf-8").decode("unicode_escape")
            except Exception:
                command = m.group(1)
            parsed = {"command": command}

    if parsed:
        for key, value in parsed.items():
            if key not in out:
                out[key] = value
    return out


@dataclass(slots=True)
class SkillContext:
    user_input: str
    branch_labels: List[str]
    attachments: List[str]
    workspace_root: str
    memory_hits: List[Dict[str, Any]]


@dataclass(slots=True)
class ToolExecutionEnv:
    workspace: WorkspaceManager
    memory: VectorMemory
    config: Dict[str, Any]
    debug: bool
    confirm_shell: Optional[Callable[[str], bool]] = None


@dataclass(slots=True)
class RegisteredTool:
    name: str
    skill_id: str
    capability: str
    description: str
    parameters: Dict[str, Any]
    module: Optional[object] = None
    command: str = ""
    timeout_s: int = 30
    confirm_arg: str = ""
    cwd: str = ""


class SkillRuntime:
    def __init__(
        self,
        skills_dir: str,
        workspace: WorkspaceManager,
        memory: VectorMemory,
        config: Optional[dict] = None,
        debug: bool = False,
    ) -> None:
        self.skills_dir = Path(skills_dir).resolve()
        self.workspace = workspace
        self.memory = memory
        self.config = config or {}
        self.debug = debug
        self.skills_cfg = self.config.get("skills", {})
        self.selection_mode = str(self.skills_cfg.get("selection_mode", "all_enabled"))
        self.max_active_skills = int(self.skills_cfg.get("max_active_skills", 6))

        self.skills: Dict[str, SkillManifest] = {}
        self._tool_registry: Dict[str, RegisteredTool] = {}
        self._proc_env_base = self._build_proc_env_base()
        self.load_skills()

    def _build_proc_env_base(self) -> Dict[str, str]:
        env = os.environ.copy()
        env["ALPHANUS_WORKSPACE_ROOT"] = str(self.workspace.workspace_root)
        env["ALPHANUS_HOME_ROOT"] = str(self.workspace.home_root)
        env["ALPHANUS_MEMORY_PATH"] = str(self.memory.storage_path)
        env["ALPHANUS_MEMORY_MODEL"] = str(self.memory.model_name)
        env["ALPHANUS_MEMORY_BACKEND"] = str(self.memory.embedding_backend)
        env["ALPHANUS_MEMORY_EAGER_LOAD"] = "1" if bool(getattr(self.memory, "eager_load_encoder", False)) else "0"
        env["ALPHANUS_CONFIG_JSON"] = json.dumps(self.config, ensure_ascii=False)

        # Let skill scripts import project modules without per-script sys.path hacks.
        repo_root = str(self.skills_dir.parent)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = repo_root if not existing else repo_root + os.pathsep + existing
        return env

    @staticmethod
    def _prepare_command(command: str) -> Tuple[object, bool]:
        # Use shell=False for simple commands (safer + lower process overhead).
        if re.search(r"[|&;<>`$()]", command):
            return command, True
        try:
            parts = shlex.split(command)
        except ValueError:
            return command, True
        if not parts:
            return command, True
        return parts, False

    @staticmethod
    def _load_module(path: Path, module_name: str):
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_manifest(self, child: Path) -> Optional[SkillManifest]:
        skill_doc = child / SKILL_DOC
        if not skill_doc.exists():
            if self.debug:
                print(f"[skill] {child.name}: missing {SKILL_DOC}")
            return None
        return parse_agentskill_manifest(child, skill_doc)

    def _ensure_skill_prompt(self, manifest: SkillManifest) -> str:
        if manifest.prompt is not None:
            return manifest.prompt
        if not manifest.doc_path:
            manifest.prompt = ""
            return manifest.prompt
        _, prompt = extract_skill_doc(manifest.doc_path, include_prompt=True)
        manifest.prompt = prompt
        return manifest.prompt

    def _ensure_skill_hooks(self, manifest: SkillManifest) -> Optional[object]:
        if manifest.hooks is not None:
            return manifest.hooks
        hooks_path = manifest.hooks_path
        if not hooks_path or not hooks_path.exists():
            return None
        manifest.hooks = self._load_module(
            hooks_path,
            f"alphanus_hooks_{manifest.id.replace('-', '_')}",
        )
        return manifest.hooks

    def _remove_skill_tools(self, skill_id: str) -> None:
        for tool_name, reg in list(self._tool_registry.items()):
            if reg.skill_id == skill_id:
                self._tool_registry.pop(tool_name, None)

    def load_skills(self) -> None:
        self.skills = {}
        self._tool_registry = {}
        if not self.skills_dir.exists():
            return

        for child in sorted(self.skills_dir.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue

            manifest: Optional[SkillManifest] = None
            try:
                manifest = self._load_manifest(child)
                if manifest is None:
                    continue

                if manifest.id in self.skills:
                    raise ValueError(f"Duplicate skill id '{manifest.id}'")

                if not self._load_skill_tools(manifest):
                    continue

                self.skills[manifest.id] = manifest
            except Exception as exc:
                self._remove_skill_tools(manifest.id if manifest else child.name)
                if self.debug:
                    print(f"[skill] failed to load {child.name}: {exc}")

    def _load_skill_tools(self, manifest: SkillManifest) -> bool:
        if not manifest.path:
            return not manifest.required_tools

        allowed_tools = set(manifest.allowed_tools)
        local_registered: List[str] = []

        for spec in manifest.command_tools:
            if allowed_tools and spec.name not in allowed_tools:
                continue
            if spec.name in self._tool_registry:
                if self.debug:
                    prev = self._tool_registry[spec.name]
                    print(f"[skill] duplicate tool '{spec.name}' in {manifest.id}; already registered by {prev.skill_id}")
                continue
            self._tool_registry[spec.name] = RegisteredTool(
                name=spec.name,
                skill_id=manifest.id,
                capability=spec.capability,
                description=spec.description,
                parameters=spec.parameters,
                command=spec.command,
                timeout_s=spec.timeout_s,
                confirm_arg=spec.confirm_arg,
                cwd=str(manifest.path),
            )
            local_registered.append(spec.name)

        tools_path = manifest.path / "tools.py"
        if tools_path.exists():
            module = self._load_module(
                tools_path,
                f"alphanus_tools_{manifest.id.replace('-', '_')}",
            )
            if module is None:
                return False

            specs = getattr(module, "TOOL_SPECS", None)
            executor = getattr(module, "execute", None)
            if not isinstance(specs, dict) or not callable(executor):
                if self.debug:
                    print(f"[skill] {manifest.id} tools.py missing TOOL_SPECS dict or execute()")
                return False

            for tool_name, spec in specs.items():
                if allowed_tools and tool_name not in allowed_tools:
                    continue
                if tool_name in self._tool_registry:
                    if self.debug:
                        prev = self._tool_registry[tool_name]
                        print(f"[skill] duplicate tool '{tool_name}' in {manifest.id}; already registered by {prev.skill_id}")
                    continue

                capability = str(spec.get("capability", "")).strip()
                description = str(spec.get("description", "")).strip()
                parameters = spec.get("parameters")

                if not capability or not description or not isinstance(parameters, dict):
                    if self.debug:
                        print(f"[skill] invalid tool spec '{tool_name}' in {manifest.id}")
                    continue

                self._tool_registry[tool_name] = RegisteredTool(
                    name=tool_name,
                    skill_id=manifest.id,
                    capability=capability,
                    description=description,
                    parameters=parameters,
                    module=module,
                    cwd=str(manifest.path),
                )
                local_registered.append(tool_name)

        if manifest.required_tools:
            local_set = set(local_registered)
            missing = [name for name in manifest.required_tools if name not in local_set]
            if missing:
                for name in local_registered:
                    self._tool_registry.pop(name, None)
                if self.debug:
                    print(f"[skill] {manifest.id} missing required tools: {', '.join(missing)}")
                return False

        return True

    def list_skills(self) -> List[SkillManifest]:
        return sorted(self.skills.values(), key=lambda s: s.id)

    def set_enabled(self, skill_id: str, enabled: bool) -> bool:
        skill = self.skills.get(skill_id)
        if not skill:
            return False
        skill.enabled = enabled
        return True

    def get_skill(self, skill_id: str) -> Optional[SkillManifest]:
        return self.skills.get(skill_id)

    def score_skills(self, ctx: SkillContext) -> List[Tuple[int, SkillManifest]]:
        text = ctx.user_input.lower()
        attachments = " ".join(ctx.attachments).lower()
        scored: List[Tuple[int, SkillManifest]] = []

        for skill in self.skills.values():
            if not skill.enabled:
                continue
            score = 0

            for kw in skill.triggers.get("keywords", []):
                if kw.lower() in text:
                    score += 30

            for ext in skill.triggers.get("file_ext", []):
                ext_lower = ext.lower()
                if ext_lower in text or ext_lower in attachments:
                    score += 20

            if "memory" in skill.id and ctx.memory_hits:
                score += 10

            scored.append((score, skill))

        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return scored

    def select_skills(self, ctx: SkillContext, top_n: int = 3) -> List[SkillManifest]:
        if self.selection_mode == "all_enabled":
            enabled = [s for s in self.skills.values() if s.enabled]
            enabled.sort(key=lambda s: s.id)
            if self.max_active_skills <= 0:
                return enabled
            return enabled[: self.max_active_skills]

        scored = self.score_skills(ctx)
        limit = self.max_active_skills if self.max_active_skills > 0 else top_n
        return [skill for _, skill in scored[: max(1, limit)]]

    def compose_skill_block(
        self,
        selected: List[SkillManifest],
        ctx: SkillContext,
        context_limit: int,
        ratio: float = 0.15,
        hard_cap: int = 3,
    ) -> str:
        selected = selected[:hard_cap]
        if not selected:
            return ""

        sections: List[str] = []
        for skill in selected:
            extra = ""
            hooks = self._ensure_skill_hooks(skill)
            if hooks and hasattr(hooks, "pre_prompt"):
                try:
                    hook_out = hooks.pre_prompt(ctx)  # type: ignore[attr-defined]
                    if hook_out:
                        extra = str(hook_out).strip()
                except Exception:
                    pass
            body = self._ensure_skill_prompt(skill).strip()
            if extra:
                body += "\n\n" + extra
            sections.append(f"### Skill: {skill.name} ({skill.id})\n{body}")

        budget = max(200, int(context_limit * ratio * 4))
        out: List[str] = []
        used = 0

        for text in sections:
            if used >= budget:
                break
            remaining = budget - used
            if len(text) <= remaining:
                out.append(text)
                used += len(text)
                continue
            lines = text.splitlines()
            head = lines[0] if lines else ""
            body = "\n".join(lines[1:])
            allowed = max(0, remaining - len(head) - 1)
            snippet = body[:allowed].rstrip()
            out.append(head + "\n" + snippet)
            used = budget

        return "\n\n".join(out)

    def allowed_tool_names(self, selected: List[SkillManifest]) -> List[str]:
        selected_map = {skill.id: skill for skill in selected}
        allowed: List[str] = []
        for tool_name, reg in self._tool_registry.items():
            skill = selected_map.get(reg.skill_id)
            if not skill:
                continue
            if skill.disable_model_invocation:
                continue
            if skill.allowed_tools and reg.name not in skill.allowed_tools:
                continue
            allowed.append(tool_name)
        return sorted(allowed)

    def tools_for_skills(self, selected: List[SkillManifest]) -> List[Dict[str, Any]]:
        names = self.allowed_tool_names(selected)
        tools = []
        for name in names:
            reg = self._tool_registry[name]
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

    def _run_pre_action_hooks(
        self,
        selected: List[SkillManifest],
        ctx: SkillContext,
        action_name: str,
        args: Dict[str, Any],
    ) -> Tuple[bool, str]:
        for skill in selected:
            hooks = self._ensure_skill_hooks(skill)
            if not hooks or not hasattr(hooks, "pre_action"):
                continue
            try:
                allowed, reason = hooks.pre_action(ctx, action_name, args)  # type: ignore[attr-defined]
            except Exception:
                continue
            if not allowed:
                return False, str(reason or "Denied by skill policy")
        return True, ""

    def post_response(self, selected: List[SkillManifest], ctx: SkillContext, text: str) -> None:
        for skill in selected:
            hooks = self._ensure_skill_hooks(skill)
            if hooks and hasattr(hooks, "post_response"):
                try:
                    hooks.post_response(ctx, text)  # type: ignore[attr-defined]
                except Exception:
                    continue

    @staticmethod
    def _schema_type_matches(value: Any, expected: str) -> bool:
        if expected == "string":
            return isinstance(value, str)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "number":
            return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, float)
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "object":
            return isinstance(value, dict)
        if expected == "array":
            return isinstance(value, list)
        if expected == "null":
            return value is None
        return True

    def _validate_schema_value(self, field_name: str, value: Any, schema: Dict[str, Any]) -> None:
        enum = schema.get("enum")
        if isinstance(enum, list) and enum and value not in enum:
            raise ValueError(f"Invalid '{field_name}': expected one of {', '.join(map(str, enum[:10]))}")

        raw_type = schema.get("type")
        expected_types: List[str] = []
        if isinstance(raw_type, str):
            expected_types = [raw_type]
        elif isinstance(raw_type, list):
            expected_types = [str(item) for item in raw_type if isinstance(item, str)]

        if expected_types and not any(self._schema_type_matches(value, item) for item in expected_types):
            raise ValueError(f"Invalid '{field_name}': expected {' or '.join(expected_types)}")

        if isinstance(value, dict):
            props = schema.get("properties")
            if isinstance(props, dict):
                required = schema.get("required") or []
                if isinstance(required, list):
                    for item in required:
                        key = str(item).strip()
                        if key and key not in value:
                            raise ValueError(f"Missing required argument: {field_name}.{key}")
                for key, child in props.items():
                    if key in value and isinstance(child, dict):
                        self._validate_schema_value(f"{field_name}.{key}", value[key], child)
            if schema.get("additionalProperties") is False and isinstance(props, dict):
                unknown = [key for key in value if key not in props]
                if unknown:
                    raise ValueError(f"Unexpected arguments for '{field_name}': {', '.join(sorted(unknown)[:5])}")

        if isinstance(value, list):
            items = schema.get("items")
            if isinstance(items, dict):
                for idx, item in enumerate(value):
                    self._validate_schema_value(f"{field_name}[{idx}]", item, items)

    def _validate_tool_args(self, reg: RegisteredTool, args: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {key: value for key, value in args.items() if not str(key).startswith("_")}
        schema = reg.parameters if isinstance(reg.parameters, dict) else {}
        schema_type = schema.get("type")
        if schema_type and schema_type != "object":
            raise ValueError(f"Tool '{reg.name}' must declare an object parameters schema")

        required = schema.get("required") or []
        if isinstance(required, list):
            for item in required:
                key = str(item).strip()
                if key and key not in cleaned:
                    raise ValueError(f"Missing required argument: {key}")

        props = schema.get("properties")
        if isinstance(props, dict):
            for key, child_schema in props.items():
                if key in cleaned and isinstance(child_schema, dict):
                    self._validate_schema_value(key, cleaned[key], child_schema)
            if schema.get("additionalProperties") is False:
                unknown = [key for key in cleaned if key not in props]
                if unknown:
                    raise ValueError(f"Unexpected arguments: {', '.join(sorted(unknown)[:5])}")

        return cleaned

    def _resolve_tool_call(
        self,
        tool_name: str,
        selected: List[SkillManifest],
    ) -> Tuple[RegisteredTool, SkillManifest]:
        reg = self._tool_registry.get(tool_name)
        if not reg:
            raise LookupError(f"No adapter for tool '{tool_name}'")

        selected_map = {skill.id: skill for skill in selected}
        owner = selected_map.get(reg.skill_id)
        if not owner:
            raise PermissionError(f"Tool '{tool_name}' not allowed by active skills")
        if owner.disable_model_invocation:
            raise PermissionError(f"Tool '{tool_name}' is disabled for model invocation")
        if owner.allowed_tools and tool_name not in owner.allowed_tools:
            raise PermissionError(f"Tool '{tool_name}' not allowed by skill policy")
        return reg, owner

    def _prepare_tool_args(
        self,
        reg: RegisteredTool,
        args: Dict[str, Any],
        selected: List[SkillManifest],
        ctx: SkillContext,
    ) -> Dict[str, Any]:
        recovered = _recover_tool_args(args)
        validated = self._validate_tool_args(reg, recovered)
        allowed, reason = self._run_pre_action_hooks(selected, ctx, reg.name, validated)
        if not allowed:
            raise PermissionError(reason or "Denied by skill policy")
        return validated

    def _execute_registered_tool(
        self,
        reg: RegisteredTool,
        args: Dict[str, Any],
        env: ToolExecutionEnv,
    ) -> Any:
        if reg.command:
            return self._execute_command_tool(reg, args, env)
        if reg.module is None or not hasattr(reg.module, "execute"):
            raise ToolProtocolError(f"Tool '{reg.name}' has no callable execute() handler")
        return reg.module.execute(reg.name, args, env)

    def execute_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        selected: List[SkillManifest],
        ctx: SkillContext,
        confirm_shell: Optional[Callable[[str], bool]] = None,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            reg, _owner = self._resolve_tool_call(tool_name, selected)
            normalized_args = self._prepare_tool_args(reg, args, selected, ctx)
            env = ToolExecutionEnv(
                workspace=self.workspace,
                memory=self.memory,
                config=self.config,
                debug=self.debug,
                confirm_shell=confirm_shell,
            )
            result = self._execute_registered_tool(reg, normalized_args, env)
            duration = int((time.perf_counter() - start) * 1000)
            return self._normalize_result(result, duration)
        except LookupError as exc:
            return _err("E_UNSUPPORTED", str(exc), int((time.perf_counter() - start) * 1000))
        except ValueError as exc:
            return _err("E_VALIDATION", str(exc), int((time.perf_counter() - start) * 1000))
        except FileNotFoundError as exc:
            return _err("E_NOT_FOUND", str(exc), int((time.perf_counter() - start) * 1000))
        except PermissionError as exc:
            return _err("E_POLICY", str(exc), int((time.perf_counter() - start) * 1000))
        except (TimeoutError, subprocess.TimeoutExpired) as exc:
            return _err("E_TIMEOUT", str(exc), int((time.perf_counter() - start) * 1000))
        except ToolProtocolError as exc:
            return _err("E_PROTOCOL", str(exc), int((time.perf_counter() - start) * 1000))
        except Exception as exc:
            message = str(exc) if self.debug else "Action failed"
            return _err("E_IO", message, int((time.perf_counter() - start) * 1000))

    def _execute_command_tool(
        self,
        reg: RegisteredTool,
        args: Dict[str, Any],
        env: ToolExecutionEnv,
    ) -> Dict[str, Any]:
        caps = env.config.get("capabilities", {})
        dangerously_skip_permissions = bool(caps.get("dangerously_skip_permissions", False))
        shell_require_confirmation = bool(caps.get("shell_require_confirmation", True))
        require_confirmation = bool(reg.confirm_arg) and shell_require_confirmation and not dangerously_skip_permissions

        if require_confirmation:
            raw = args.get(reg.confirm_arg)
            command = str(raw).strip() if raw is not None else ""
            if not command:
                raise ValueError(f"Missing required confirmation argument: {reg.confirm_arg}")
            if not env.confirm_shell:
                raise PermissionError("Shell confirmation callback is required")
            if not env.confirm_shell(command):
                raise PermissionError("Shell command rejected by user")

        proc_env = dict(self._proc_env_base)
        proc_env["ALPHANUS_TOOL_NAME"] = reg.name
        proc_env["ALPHANUS_TOOL_ARGS_JSON"] = json.dumps(args, ensure_ascii=False)
        prepared_command, use_shell = self._prepare_command(reg.command)

        proc = subprocess.run(
            prepared_command,
            shell=use_shell,
            cwd=reg.cwd or str(self.skills_dir),
            capture_output=True,
            text=True,
            input=json.dumps(args, ensure_ascii=False),
            timeout=max(1, int(reg.timeout_s)),
            env=proc_env,
        )

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            msg = stderr or stdout or f"Tool command failed with exit code {proc.returncode}"
            lowered = msg.lower()
            if "permissionerror" in lowered or "operation not permitted" in lowered:
                raise PermissionError(msg)
            if "filenotfounderror" in lowered or "no such file or directory" in lowered:
                raise FileNotFoundError(msg)
            if "timeouterror" in lowered or "timed out" in lowered:
                raise TimeoutError(msg)
            raise RuntimeError(msg)

        out = (proc.stdout or "").strip()
        if not out:
            return {}
        candidate = out.splitlines()[-1].strip()

        try:
            parsed = json.loads(candidate)
        except Exception as exc:
            raise ToolProtocolError(
                "Tool command output is not valid JSON"
                + (f": {exc}" if self.debug else "")
            ) from exc

        if not isinstance(parsed, dict):
            return {"value": parsed}
        return parsed

    @staticmethod
    def _normalize_result(result: Any, duration_ms: int) -> Dict[str, Any]:
        if isinstance(result, dict) and {"ok", "data", "error"}.issubset(result.keys()):
            out = dict(result)
            meta = out.get("meta") if isinstance(out.get("meta"), dict) else {}
            meta["duration_ms"] = int(meta.get("duration_ms", duration_ms))
            out["meta"] = meta
            return out
        return _ok(result, duration_ms)
