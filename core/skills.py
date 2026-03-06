from __future__ import annotations

import importlib.util
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.memory import VectorMemory
from core.workspace import WorkspaceManager

SCHEMA_VERSION = "1.0.0"


def _major(version: str) -> int:
    return int((version or "0").split(".", 1)[0])


def _ok(data: Any, duration_ms: int) -> Dict[str, Any]:
    return {"ok": True, "data": data, "error": None, "meta": {"duration_ms": duration_ms}}


def _err(code: str, message: str, duration_ms: int) -> Dict[str, Any]:
    return {
        "ok": False,
        "data": None,
        "error": {"code": code, "message": message},
        "meta": {"duration_ms": duration_ms},
    }


@dataclass
class SkillManifest:
    id: str
    name: str
    version: str
    description: str
    enabled: bool
    priority: int
    schema_version: str
    triggers: Dict[str, List[str]] = field(default_factory=dict)
    prompt: str = ""
    path: Optional[Path] = None
    hooks: Optional[object] = None


@dataclass
class SkillContext:
    user_input: str
    branch_labels: List[str]
    attachments: List[str]
    workspace_root: str
    memory_hits: List[Dict[str, Any]]


@dataclass
class ToolExecutionEnv:
    workspace: WorkspaceManager
    memory: VectorMemory
    config: Dict[str, Any]
    debug: bool
    confirm_shell: Optional[Callable[[str], bool]] = None


@dataclass
class RegisteredTool:
    name: str
    skill_id: str
    capability: str
    description: str
    parameters: Dict[str, Any]
    module: object


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

        self.skills: Dict[str, SkillManifest] = {}
        self._tool_registry: Dict[str, RegisteredTool] = {}
        self.load_skills()

    @staticmethod
    def _load_module(path: Path, module_name: str):
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def load_skills(self) -> None:
        self.skills = {}
        self._tool_registry = {}
        if not self.skills_dir.exists():
            return

        for child in sorted(self.skills_dir.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            manifest_path = child / "skill.toml"
            prompt_path = child / "prompt.md"
            if not manifest_path.exists():
                continue
            try:
                with manifest_path.open("rb") as handle:
                    raw = tomllib.load(handle)

                schema_version = raw.get("schema_version", "1.0.0")
                if _major(schema_version) != _major(SCHEMA_VERSION):
                    raise ValueError(f"Unsupported skill schema version {schema_version}")

                manifest = SkillManifest(
                    id=raw["id"],
                    name=raw.get("name", raw["id"]),
                    version=raw.get("version", "0.0.0"),
                    description=raw.get("description", ""),
                    enabled=bool(raw.get("enabled", True)),
                    priority=int(raw.get("priority", 50)),
                    schema_version=schema_version,
                    triggers={
                        "keywords": list(raw.get("triggers", {}).get("keywords", [])),
                        "file_ext": list(raw.get("triggers", {}).get("file_ext", [])),
                        "capabilities": list(raw.get("triggers", {}).get("capabilities", [])),
                    },
                    prompt=prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else "",
                    path=child,
                )

                hooks_path = child / "hooks.py"
                if hooks_path.exists():
                    manifest.hooks = self._load_module(
                        hooks_path,
                        f"alphanus_hooks_{manifest.id.replace('-', '_')}",
                    )

                self.skills[manifest.id] = manifest
                self._load_skill_tools(manifest)
            except Exception as exc:
                if self.debug:
                    print(f"[skill] failed to load {child.name}: {exc}")

    def _load_skill_tools(self, manifest: SkillManifest) -> None:
        if not manifest.path:
            return
        tools_path = manifest.path / "tools.py"
        if not tools_path.exists():
            return

        module = self._load_module(
            tools_path,
            f"alphanus_tools_{manifest.id.replace('-', '_')}",
        )
        if module is None:
            return

        specs = getattr(module, "TOOL_SPECS", None)
        executor = getattr(module, "execute", None)
        if not isinstance(specs, dict) or not callable(executor):
            if self.debug:
                print(f"[skill] {manifest.id} tools.py missing TOOL_SPECS dict or execute()")
            return

        for tool_name, spec in specs.items():
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
            )

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
            score = skill.priority

            for kw in skill.triggers.get("keywords", []):
                if kw.lower() in text:
                    score += 30

            for ext in skill.triggers.get("file_ext", []):
                if ext.lower() in text or ext.lower() in attachments:
                    score += 20

            if "memory" in skill.id and ctx.memory_hits:
                score += 10

            scored.append((score, skill))

        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return scored

    def select_skills(self, ctx: SkillContext, top_n: int = 3) -> List[SkillManifest]:
        scored = self.score_skills(ctx)
        return [skill for _, skill in scored[: max(1, top_n)]]

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

        sections: List[Tuple[int, str]] = []
        for skill in selected:
            extra = ""
            if skill.hooks and hasattr(skill.hooks, "pre_prompt"):
                try:
                    hook_out = skill.hooks.pre_prompt(ctx)  # type: ignore[attr-defined]
                    if hook_out:
                        extra = str(hook_out).strip()
                except Exception:
                    pass
            body = skill.prompt.strip()
            if extra:
                body += "\n\n" + extra
            text = f"### Skill: {skill.name} ({skill.id})\n{body}"
            sections.append((skill.priority, text))

        budget = max(200, int(context_limit * ratio * 4))
        ordered = sorted(sections, key=lambda pair: pair[0], reverse=True)
        out: List[str] = []
        used = 0

        for _, text in ordered:
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
            if reg.capability not in skill.triggers.get("capabilities", []):
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
            hooks = skill.hooks
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
            hooks = skill.hooks
            if hooks and hasattr(hooks, "post_response"):
                try:
                    hooks.post_response(ctx, text)  # type: ignore[attr-defined]
                except Exception:
                    continue

    def execute_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        selected: List[SkillManifest],
        ctx: SkillContext,
        confirm_shell: Optional[Callable[[str], bool]] = None,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        reg = self._tool_registry.get(tool_name)
        if not reg:
            return _err("E_UNSUPPORTED", f"No adapter for tool '{tool_name}'", int((time.perf_counter() - start) * 1000))

        selected_map = {skill.id: skill for skill in selected}
        owner = selected_map.get(reg.skill_id)
        if not owner:
            return _err("E_POLICY", f"Tool '{tool_name}' not allowed by active skills", int((time.perf_counter() - start) * 1000))
        if reg.capability not in owner.triggers.get("capabilities", []):
            return _err("E_POLICY", f"Capability '{reg.capability}' not enabled for skill '{owner.id}'", int((time.perf_counter() - start) * 1000))

        allowed, reason = self._run_pre_action_hooks(selected, ctx, tool_name, args)
        if not allowed:
            return _err("E_POLICY", reason, int((time.perf_counter() - start) * 1000))

        env = ToolExecutionEnv(
            workspace=self.workspace,
            memory=self.memory,
            config=self.config,
            debug=self.debug,
            confirm_shell=confirm_shell,
        )

        try:
            result = reg.module.execute(tool_name, args, env)
            duration = int((time.perf_counter() - start) * 1000)
            return self._normalize_result(result, duration)
        except ValueError as exc:
            return _err("E_VALIDATION", str(exc), int((time.perf_counter() - start) * 1000))
        except FileNotFoundError as exc:
            return _err("E_NOT_FOUND", str(exc), int((time.perf_counter() - start) * 1000))
        except PermissionError as exc:
            return _err("E_POLICY", str(exc), int((time.perf_counter() - start) * 1000))
        except TimeoutError as exc:
            return _err("E_TIMEOUT", str(exc), int((time.perf_counter() - start) * 1000))
        except Exception as exc:
            message = str(exc) if self.debug else "Action failed"
            return _err("E_IO", message, int((time.perf_counter() - start) * 1000))

    @staticmethod
    def _normalize_result(result: Any, duration_ms: int) -> Dict[str, Any]:
        if isinstance(result, dict) and {"ok", "data", "error"}.issubset(result.keys()):
            out = dict(result)
            meta = out.get("meta") if isinstance(out.get("meta"), dict) else {}
            meta["duration_ms"] = int(meta.get("duration_ms", duration_ms))
            out["meta"] = meta
            return out
        return _ok(result, duration_ms)
