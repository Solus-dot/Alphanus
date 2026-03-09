from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.memory import VectorMemory
from core.workspace import WorkspaceManager

try:
    import yaml  # type: ignore[import-not-found]
except Exception:
    yaml = None

_SKILL_DOC = "SKILL.md"
_ALLOWED_CATEGORIES = {
    "coding",
    "data-science",
    "devops",
    "system",
    "productivity",
    "communication",
    "business",
    "education",
    "creative",
    "security",
    "custom",
}
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?(?:\d+\.\d+|\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?$")


def _ok(data: Any, duration_ms: int) -> Dict[str, Any]:
    return {"ok": True, "data": data, "error": None, "meta": {"duration_ms": duration_ms}}


def _err(code: str, message: str, duration_ms: int) -> Dict[str, Any]:
    return {
        "ok": False,
        "data": None,
        "error": {"code": code, "message": message},
        "meta": {"duration_ms": duration_ms},
    }


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in items:
        item = str(raw).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _split_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return _dedupe([str(item) for item in value])
    if isinstance(value, tuple):
        return _dedupe([str(item) for item in value])
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = _parse_yaml_inline_list(text)
                if isinstance(parsed, list):
                    return _dedupe([str(item) for item in parsed])
            except Exception:
                pass
        return _dedupe(_split_csv(text))
    return [str(value).strip()] if str(value).strip() else []


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "on"}:
            return True
        if lowered in {"false", "no", "0", "off"}:
            return False
    return default


def _split_kv(line: str) -> Optional[Tuple[str, str]]:
    if ":" not in line:
        return None
    key, value = line.split(":", 1)
    key = key.strip()
    if not key or not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
        return None
    return key, value.strip()


def _parse_yaml_inline_list(raw: str) -> List[Any]:
    text = raw.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError("Expected inline YAML list")
    inner = text[1:-1].strip()
    if not inner:
        return []

    parts: List[str] = []
    buff: List[str] = []
    quote: Optional[str] = None

    for ch in inner:
        if quote:
            buff.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            buff.append(ch)
            continue
        if ch == ",":
            parts.append("".join(buff).strip())
            buff = []
            continue
        buff.append(ch)

    tail = "".join(buff).strip()
    if tail:
        parts.append(tail)

    return [_parse_yaml_scalar(part) for part in parts if part]


def _parse_yaml_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""

    if value.startswith("[") and value.endswith("]"):
        return _parse_yaml_inline_list(value)

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        quote = value[0]
        body = value[1:-1]
        if quote == '"':
            body = (
                body.replace(r"\\n", "\n")
                .replace(r"\\t", "\t")
                .replace(r'\\"', '"')
                .replace(r"\\\\", "\\")
            )
        else:
            body = body.replace(r"\\'", "'").replace(r"\\\\", "\\")
        return body

    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "~", "none"}:
        return None

    if _INT_RE.match(value):
        try:
            return int(value)
        except Exception:
            pass
    if _FLOAT_RE.match(value):
        try:
            return float(value)
        except Exception:
            pass

    return value


def _prepare_yaml_lines(text: str) -> List[Tuple[int, int, str]]:
    out: List[Tuple[int, int, str]] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        leading = len(raw) - len(raw.lstrip(" "))
        if "\t" in raw[:leading]:
            raise ValueError(f"Tabs are not supported in YAML frontmatter (line {line_no})")
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        out.append((line_no, leading, stripped))
    return out


def _parse_yaml_map(lines: List[Tuple[int, int, str]], idx: int, indent: int) -> Tuple[Dict[str, Any], int]:
    out: Dict[str, Any] = {}

    while idx < len(lines):
        line_no, cur_indent, text = lines[idx]
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise ValueError(f"Unexpected indentation at line {line_no}")
        if text.startswith("- "):
            raise ValueError(f"List item found where mapping expected at line {line_no}")

        pair = _split_kv(text)
        if pair is None:
            raise ValueError(f"Invalid key/value syntax at line {line_no}")
        key, inline_value = pair
        idx += 1

        if inline_value:
            out[key] = _parse_yaml_scalar(inline_value)
            continue

        if idx >= len(lines) or lines[idx][1] <= cur_indent:
            out[key] = {}
            continue

        next_indent = lines[idx][1]
        if lines[idx][2].startswith("- "):
            nested, idx = _parse_yaml_list(lines, idx, next_indent)
        else:
            nested, idx = _parse_yaml_map(lines, idx, next_indent)
        out[key] = nested

    return out, idx


def _parse_yaml_list(lines: List[Tuple[int, int, str]], idx: int, indent: int) -> Tuple[List[Any], int]:
    items: List[Any] = []

    while idx < len(lines):
        line_no, cur_indent, text = lines[idx]
        if cur_indent < indent:
            break
        if cur_indent != indent or not text.startswith("- "):
            break

        item_text = text[2:].strip()
        idx += 1

        if item_text:
            pair = _split_kv(item_text)
            if pair is None:
                items.append(_parse_yaml_scalar(item_text))
                continue

            key, inline_value = pair
            item: Dict[str, Any] = {key: _parse_yaml_scalar(inline_value)}
            if idx < len(lines) and lines[idx][1] > cur_indent:
                nested, idx = _parse_yaml_map(lines, idx, lines[idx][1])
                item.update(nested)
            items.append(item)
            continue

        if idx >= len(lines) or lines[idx][1] <= cur_indent:
            items.append({})
            continue

        next_indent = lines[idx][1]
        if lines[idx][2].startswith("- "):
            nested, idx = _parse_yaml_list(lines, idx, next_indent)
        else:
            nested, idx = _parse_yaml_map(lines, idx, next_indent)
        items.append(nested)

    return items, idx


def _parse_yaml_frontmatter(text: str) -> Dict[str, Any]:
    if yaml is not None:
        parsed = yaml.safe_load(text)  # type: ignore[union-attr]
        if parsed is None:
            return {}
        if not isinstance(parsed, dict):
            raise ValueError("SKILL.md frontmatter must be a mapping")
        return parsed

    lines = _prepare_yaml_lines(text)
    if not lines:
        return {}

    parsed, idx = _parse_yaml_map(lines, 0, lines[0][1])
    if idx != len(lines):
        line_no, _, _ = lines[idx]
        raise ValueError(f"Unable to parse YAML frontmatter near line {line_no}")
    return parsed


def _extract_skill_doc(skill_doc: Path) -> Tuple[Dict[str, Any], str]:
    text = skill_doc.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter delimiter '---'")

    end = -1
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end = idx
            break
    if end < 0:
        raise ValueError("SKILL.md frontmatter is missing closing delimiter '---'")

    frontmatter_text = "\n".join(lines[1:end])
    prompt = "\n".join(lines[end + 1 :]).strip()
    frontmatter = _parse_yaml_frontmatter(frontmatter_text)
    return frontmatter, prompt


@dataclass(slots=True)
class SkillManifest:
    id: str
    name: str
    version: str
    description: str
    enabled: bool
    triggers: Dict[str, List[str]] = field(default_factory=dict)
    prompt: str = ""
    path: Optional[Path] = None
    hooks: Optional[object] = None
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    command_tools: List["ToolCommandDef"] = field(default_factory=list)
    disable_model_invocation: bool = False
    format: str = "agentskills"


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


@dataclass(slots=True)
class ToolCommandDef:
    name: str
    capability: str
    description: str
    parameters: Dict[str, Any]
    command: str
    timeout_s: int = 30
    confirm_arg: str = ""


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
        self.load_skills()

    @staticmethod
    def _load_module(path: Path, module_name: str):
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_agentskill_manifest(self, child: Path, skill_doc: Path) -> SkillManifest:
        frontmatter, prompt = _extract_skill_doc(skill_doc)

        skill_id = str(frontmatter.get("name", "")).strip()
        if not skill_id:
            raise ValueError("SKILL.md frontmatter requires 'name'")
        if skill_id != child.name:
            raise ValueError(f"SKILL.md name '{skill_id}' must match directory '{child.name}'")

        description = str(frontmatter.get("description", "")).strip()
        if not description:
            raise ValueError("SKILL.md frontmatter requires 'description'")

        version = str(frontmatter.get("version", "0.1.0")).strip()
        if version and not _SEMVER_RE.match(version):
            raise ValueError(f"Invalid semantic version: '{version}'")

        categories = _as_str_list(frontmatter.get("categories"))
        invalid_categories = [cat for cat in categories if cat not in _ALLOWED_CATEGORIES]
        if invalid_categories:
            joined = ", ".join(sorted(set(invalid_categories)))
            raise ValueError(f"Invalid categories: {joined}")

        tags = _as_str_list(frontmatter.get("tags"))

        tools_cfg_raw = frontmatter.get("tools", {})
        if tools_cfg_raw is None:
            tools_cfg_raw = {}
        if not isinstance(tools_cfg_raw, dict):
            raise ValueError("SKILL.md 'tools' must be a mapping")

        allowed_tools = _as_str_list(tools_cfg_raw.get("allowed-tools"))
        required_tools = _as_str_list(tools_cfg_raw.get("required-tools"))
        raw_defs = tools_cfg_raw.get("definitions") or []
        if raw_defs and not isinstance(raw_defs, list):
            raise ValueError("SKILL.md tools.definitions must be a list")
        command_tools: List[ToolCommandDef] = []
        for idx, raw in enumerate(raw_defs):
            if not isinstance(raw, dict):
                raise ValueError(f"SKILL.md tools.definitions[{idx}] must be a mapping")

            name = str(raw.get("name", "")).strip()
            capability = str(raw.get("capability", "")).strip()
            description = str(raw.get("description", "")).strip()
            command = str(raw.get("command", "")).strip()
            parameters = raw.get("parameters")
            timeout_s = int(raw.get("timeout-s", 30))
            confirm_arg = str(raw.get("confirm-arg", "")).strip()

            if not name:
                raise ValueError(f"SKILL.md tools.definitions[{idx}] missing name")
            if not capability:
                raise ValueError(f"SKILL.md tools.definitions[{idx}] missing capability")
            if not description:
                raise ValueError(f"SKILL.md tools.definitions[{idx}] missing description")
            if not command:
                raise ValueError(f"SKILL.md tools.definitions[{idx}] missing command")
            if not isinstance(parameters, dict):
                raise ValueError(f"SKILL.md tools.definitions[{idx}] parameters must be a mapping")
            if timeout_s <= 0:
                raise ValueError(f"SKILL.md tools.definitions[{idx}] timeout-s must be > 0")

            command_tools.append(
                ToolCommandDef(
                    name=name,
                    capability=capability,
                    description=description,
                    parameters=parameters,
                    command=command,
                    timeout_s=timeout_s,
                    confirm_arg=confirm_arg,
                )
            )
        disable_model_invocation = _coerce_bool(
            tools_cfg_raw.get("disable-model-invocation"),
            False,
        )

        ext_cfg_raw = frontmatter.get("x-alphanus", {})
        if ext_cfg_raw is None:
            ext_cfg_raw = {}
        if not isinstance(ext_cfg_raw, dict):
            raise ValueError("SKILL.md 'x-alphanus' must be a mapping")

        trigger_cfg = ext_cfg_raw.get("triggers", {})
        if trigger_cfg is None:
            trigger_cfg = {}
        if not isinstance(trigger_cfg, dict):
            raise ValueError("SKILL.md x-alphanus.triggers must be a mapping")

        enabled = _coerce_bool(ext_cfg_raw.get("enabled"), True)
        keywords = _as_str_list(trigger_cfg.get("keywords")) or list(tags)
        file_ext = _as_str_list(trigger_cfg.get("file_ext"))

        return SkillManifest(
            id=skill_id,
            name=skill_id,
            version=version,
            description=description,
            enabled=enabled,
            triggers={
                "keywords": keywords,
                "file_ext": file_ext,
            },
            prompt=prompt,
            path=child,
            tags=tags,
            categories=categories,
            allowed_tools=allowed_tools,
            required_tools=required_tools,
            command_tools=command_tools,
            disable_model_invocation=disable_model_invocation,
            format="agentskills",
        )

    def _load_manifest(self, child: Path) -> Optional[SkillManifest]:
        skill_doc = child / _SKILL_DOC
        if not skill_doc.exists():
            if self.debug:
                print(f"[skill] {child.name}: missing {_SKILL_DOC}")
            return None
        return self._load_agentskill_manifest(child, skill_doc)

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

                hooks_path = child / "hooks.py"
                if hooks_path.exists():
                    manifest.hooks = self._load_module(
                        hooks_path,
                        f"alphanus_hooks_{manifest.id.replace('-', '_')}",
                    )

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
                if ext.lower() in text or ext.lower() in attachments:
                    score += 20

            if "memory" in skill.id and ctx.memory_hits:
                score += 10

            scored.append((score, skill))

        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return scored

    def select_skills(self, ctx: SkillContext, top_n: int = 3) -> List[SkillManifest]:
        enabled = [s for s in self.skills.values() if s.enabled]
        enabled.sort(key=lambda s: s.id)

        if self.selection_mode == "all_enabled":
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
        if owner.disable_model_invocation:
            return _err("E_POLICY", f"Tool '{tool_name}' is disabled for model invocation", int((time.perf_counter() - start) * 1000))
        if owner.allowed_tools and tool_name not in owner.allowed_tools:
            return _err("E_POLICY", f"Tool '{tool_name}' not allowed by skill policy", int((time.perf_counter() - start) * 1000))

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
            if reg.command:
                result = self._execute_command_tool(reg, args, env)
            else:
                if reg.module is None:
                    raise RuntimeError(f"Tool '{tool_name}' has no module or command executor")
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

        proc_env = os.environ.copy()
        proc_env["ALPHANUS_TOOL_NAME"] = reg.name
        proc_env["ALPHANUS_TOOL_ARGS_JSON"] = json.dumps(args, ensure_ascii=False)
        proc_env["ALPHANUS_WORKSPACE_ROOT"] = str(env.workspace.workspace_root)
        proc_env["ALPHANUS_HOME_ROOT"] = str(env.workspace.home_root)
        proc_env["ALPHANUS_MEMORY_PATH"] = str(env.memory.storage_path)
        proc_env["ALPHANUS_MEMORY_MODEL"] = str(env.memory.model_name)
        proc_env["ALPHANUS_MEMORY_BACKEND"] = str(env.memory.embedding_backend)
        proc_env["ALPHANUS_MEMORY_EAGER_LOAD"] = "1" if bool(getattr(env.memory, "eager_load_encoder", False)) else "0"
        proc_env["ALPHANUS_CONFIG_JSON"] = json.dumps(env.config, ensure_ascii=False)

        proc = subprocess.run(
            reg.command,
            shell=True,
            cwd=reg.cwd or str(self.skills_dir),
            capture_output=True,
            text=True,
            input=json.dumps(args, ensure_ascii=False),
            timeout=max(1, int(reg.timeout_s)),
            env=proc_env,
        )

        out = (proc.stdout or "").strip()
        if not out:
            raise RuntimeError("Tool command produced no JSON output")
        # Allow scripts to print diagnostics as long as the last line is the JSON result.
        candidate = out.splitlines()[-1].strip()

        try:
            parsed = json.loads(candidate)
        except Exception as exc:
            raise RuntimeError(
                "Tool command output is not valid JSON"
                + (f": {exc}" if self.debug else "")
            ) from exc

        if not isinstance(parsed, dict):
            raise RuntimeError("Tool command must return a JSON object result")

        if proc.returncode != 0 and not parsed.get("ok"):
            error = parsed.get("error") if isinstance(parsed.get("error"), dict) else {}
            if not error:
                parsed["error"] = {
                    "code": "E_IO",
                    "message": f"Tool command failed with exit code {proc.returncode}",
                }
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
