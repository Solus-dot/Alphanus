from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

SKILL_DOC = "SKILL.md"
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")


def _dedupe(items: List[str]) -> List[str]:
    return list(dict.fromkeys(str(raw).strip() for raw in items if str(raw).strip()))


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return _dedupe([str(item) for item in value])
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return _dedupe([part.strip() for part in text.split(",") if part.strip()])
    return [str(value).strip()] if str(value).strip() else []


def _as_tool_name_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return _dedupe([str(item) for item in value])
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return _dedupe([part for part in re.split(r"[\s,]+", text) if part])
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


def extract_skill_doc(skill_doc: Path, include_prompt: bool = True) -> Tuple[Dict[str, Any], Optional[str]]:
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
    prompt = "\n".join(lines[end + 1 :]).strip() if include_prompt else None
    parsed = yaml.safe_load(frontmatter_text)
    if parsed is None:
        frontmatter: Dict[str, Any] = {}
    elif isinstance(parsed, dict):
        frontmatter = parsed
    else:
        raise ValueError("SKILL.md frontmatter must be a mapping")
    return frontmatter, prompt


@dataclass(slots=True)
class ToolCommandDef:
    name: str
    capability: str
    description: str
    parameters: Dict[str, Any]
    command: str


@dataclass(slots=True)
class SkillEntrypointDef:
    name: str
    description: str
    command: str
    parameters: Dict[str, Any]
    intents: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    install: List[str] = field(default_factory=list)
    verify: List[str] = field(default_factory=list)
    timeout_s: int = 30
    cwd: str = "workspace"


@dataclass(slots=True)
class SkillManifest:
    id: str
    name: str
    version: str
    description: str
    enabled: bool
    requirements: Dict[str, List[str]] = field(default_factory=dict)
    triggers: Dict[str, List[str]] = field(default_factory=dict)
    prompt: Optional[str] = None
    path: Optional[Path] = None
    doc_path: Optional[Path] = None
    hooks_path: Optional[Path] = None
    hooks: Optional[object] = None
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    execution_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    allowed_tools: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    command_tools: List[ToolCommandDef] = field(default_factory=list)
    entrypoints: List[SkillEntrypointDef] = field(default_factory=list)
    disable_model_invocation: bool = False
    user_invocable: bool = True
    argument_hint: str = ""
    format: str = "agentskills"
    source_tier: str = "bundled"
    available: bool = True
    availability_code: str = "ready"
    availability_reason: str = ""
    trust_level: str = "trusted"
    execution_allowed: bool = True
    adapter: str = "agentskills"
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    shadowed_by: str = ""
    shadowing: List[str] = field(default_factory=list)
    blocked_features: List[str] = field(default_factory=list)
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    bundled_files: List[str] = field(default_factory=list)
    vendor_flavor: str = "agentskills"


def parse_agentskill_manifest(child: Path, skill_doc: Path, include_prompt: bool = False) -> SkillManifest:
    frontmatter, prompt = extract_skill_doc(skill_doc, include_prompt=include_prompt)

    skill_id = str(frontmatter.get("name", "")).strip()
    if not skill_id:
        raise ValueError("SKILL.md frontmatter requires 'name'")
    if skill_id != child.name:
        raise ValueError(f"SKILL.md name '{skill_id}' must match directory '{child.name}'")

    description = str(frontmatter.get("description", "")).strip()
    if not description:
        raise ValueError("SKILL.md frontmatter requires 'description'")

    metadata_raw = frontmatter.get("metadata", {})
    if metadata_raw is None:
        metadata_raw = {}
    if not isinstance(metadata_raw, dict):
        raise ValueError("SKILL.md 'metadata' must be a mapping")

    version_raw = metadata_raw.get("version", frontmatter.get("version", "0.1.0"))
    version = str(version_raw).strip()
    if version and not _SEMVER_RE.match(version):
        raise ValueError(f"Invalid semantic version: '{version}'")

    categories = _dedupe(_as_str_list(frontmatter.get("categories") or metadata_raw.get("categories")))
    tags = _as_str_list(frontmatter.get("tags") or metadata_raw.get("tags"))
    produces = _dedupe(
        _as_str_list(
            frontmatter.get("produces")
            or frontmatter.get("artifacts")
            or metadata_raw.get("produces")
            or metadata_raw.get("artifacts")
        )
    )

    requirements_raw = frontmatter.get("requirements") or metadata_raw.get("requirements") or {}
    if requirements_raw is None:
        requirements_raw = {}
    if not isinstance(requirements_raw, dict):
        raise ValueError("SKILL.md requirements must be a mapping")
    requirements = {
        "os": _as_str_list(requirements_raw.get("os")),
        "env": _as_str_list(requirements_raw.get("env")),
        "commands": _as_str_list(
            requirements_raw.get("commands")
            or requirements_raw.get("binaries")
            or requirements_raw.get("bins")
        ),
    }
    vendor_flavor = str(frontmatter.get("format") or metadata_raw.get("format") or "agentskills").strip() or "agentskills"
    vendor_requires: Dict[str, Any] = {}
    for vendor_key in ("openclaw", "claude", "opencode"):
        candidate = metadata_raw.get(vendor_key)
        if isinstance(candidate, dict):
            vendor_requires = candidate.get("requires") if isinstance(candidate.get("requires"), dict) else {}
            if vendor_requires:
                vendor_flavor = vendor_key
                break
    if vendor_requires:
        requirements["os"] = _dedupe(requirements["os"] + _as_str_list(vendor_requires.get("os")))
        requirements["env"] = _dedupe(requirements["env"] + _as_str_list(vendor_requires.get("env")))
        requirements["commands"] = _dedupe(
            requirements["commands"]
            + _as_str_list(vendor_requires.get("commands") or vendor_requires.get("bins") or vendor_requires.get("binaries"))
            + _as_str_list(vendor_requires.get("anyBins"))
        )

    execution_raw = frontmatter.get("execution") or metadata_raw.get("execution") or {}
    if execution_raw is None:
        execution_raw = {}
    if not isinstance(execution_raw, dict):
        raise ValueError("SKILL.md execution must be a mapping")
    execution_dependencies_raw = execution_raw.get("dependencies") or {}
    if execution_dependencies_raw is None:
        execution_dependencies_raw = {}
    if not isinstance(execution_dependencies_raw, dict):
        raise ValueError("SKILL.md execution.dependencies must be a mapping")
    execution_dependencies = {
        "python": _as_str_list(execution_dependencies_raw.get("python") or execution_dependencies_raw.get("pip")),
        "commands": _as_str_list(
            execution_dependencies_raw.get("commands")
            or execution_dependencies_raw.get("binaries")
            or execution_dependencies_raw.get("system")
        ),
    }
    global_install = _as_str_list(
        execution_raw.get("install")
        or execution_raw.get("install_commands")
        or execution_raw.get("installs")
    )
    global_verify = _as_str_list(
        execution_raw.get("verify")
        or execution_raw.get("verify_commands")
        or execution_raw.get("preflight")
    )
    raw_entrypoints = execution_raw.get("entrypoints") or []
    if raw_entrypoints and not isinstance(raw_entrypoints, list):
        raise ValueError("SKILL.md execution.entrypoints must be a list")
    entrypoints: List[SkillEntrypointDef] = []
    for idx, raw in enumerate(raw_entrypoints):
        if not isinstance(raw, dict):
            raise ValueError(f"SKILL.md execution.entrypoints[{idx}] must be a mapping")
        name = str(raw.get("name", "")).strip()
        command = str(raw.get("command", "")).strip()
        if not name:
            raise ValueError(f"SKILL.md execution.entrypoints[{idx}] missing name")
        if not command:
            raise ValueError(f"SKILL.md execution.entrypoints[{idx}] missing command")
        tool = str(raw.get("tool", "shell_command")).strip() or "shell_command"
        if tool != "shell_command":
            raise ValueError(f"SKILL.md execution.entrypoints[{idx}] unsupported tool '{tool}'")
        description = str(raw.get("description", "")).strip() or name
        parameters = raw.get("parameters")
        if parameters is None:
            parameters = {"type": "object", "properties": {}, "required": []}
        if not isinstance(parameters, dict):
            raise ValueError(f"SKILL.md execution.entrypoints[{idx}] parameters must be a mapping")
        timeout_s = int(raw.get("timeout-s", 30))
        if timeout_s <= 0:
            raise ValueError(f"SKILL.md execution.entrypoints[{idx}] timeout-s must be > 0")
        cwd = str(raw.get("cwd", "workspace")).strip().lower() or "workspace"
        if cwd not in {"workspace", "skill"}:
            raise ValueError(f"SKILL.md execution.entrypoints[{idx}] cwd must be 'workspace' or 'skill'")
        intents = _as_str_list(raw.get("intents") or raw.get("intent") or ["general"])
        produces_for_entry = _dedupe(_as_str_list(raw.get("produces") or raw.get("artifacts") or produces))
        install = _as_str_list(raw.get("install") or raw.get("install_commands") or global_install)
        verify = _as_str_list(raw.get("verify") or raw.get("verify_commands") or raw.get("preflight") or global_verify)
        entrypoints.append(
            SkillEntrypointDef(
                name=name,
                description=description,
                command=command,
                parameters=parameters,
                intents=intents,
                produces=produces_for_entry,
                install=install,
                verify=verify,
                timeout_s=timeout_s,
                cwd=cwd,
            )
        )

    tools_cfg_raw = frontmatter.get("tools", {})
    if tools_cfg_raw is None:
        tools_cfg_raw = {}
    if not isinstance(tools_cfg_raw, dict):
        raise ValueError("SKILL.md 'tools' must be a mapping")
    metadata_tools_raw = metadata_raw.get("tools", {})
    if metadata_tools_raw is None:
        metadata_tools_raw = {}
    if not isinstance(metadata_tools_raw, dict):
        raise ValueError("SKILL.md metadata.tools must be a mapping")

    allowed_tools = _as_tool_name_list(
        frontmatter.get("allowed-tools")
        or metadata_raw.get("allowed-tools")
        or metadata_tools_raw.get("allowed-tools")
        or tools_cfg_raw.get("allowed-tools")
    )
    required_tools = _as_tool_name_list(
        frontmatter.get("required-tools")
        or metadata_raw.get("required-tools")
        or metadata_tools_raw.get("required-tools")
        or tools_cfg_raw.get("required-tools")
    )
    raw_defs = metadata_tools_raw.get("definitions") or tools_cfg_raw.get("definitions") or []
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
            )
        )
    disable_model_invocation = _coerce_bool(
        frontmatter.get(
            "disable-model-invocation",
            metadata_raw.get(
                "disable-model-invocation",
                metadata_tools_raw.get("disable-model-invocation", tools_cfg_raw.get("disable-model-invocation")),
            ),
        ),
        False,
    )
    user_invocable = _coerce_bool(
        frontmatter.get("user-invocable", metadata_raw.get("user-invocable")),
        True,
    )
    argument_hint = str(frontmatter.get("argument-hint") or metadata_raw.get("argument-hint") or "").strip()

    trigger_cfg = metadata_raw.get("triggers", frontmatter.get("triggers", {}))
    if trigger_cfg is None:
        trigger_cfg = {}
    if not isinstance(trigger_cfg, dict):
        raise ValueError("SKILL.md triggers must be a mapping")

    enabled = _coerce_bool(metadata_raw.get("enabled", frontmatter.get("enabled")), True)
    keywords = _as_str_list(trigger_cfg.get("keywords")) or list(tags)
    file_ext = _as_str_list(trigger_cfg.get("file_ext"))

    return SkillManifest(
        id=skill_id,
        name=skill_id,
        version=version,
        description=description,
        enabled=enabled,
        requirements=requirements,
        triggers={"keywords": keywords, "file_ext": file_ext},
        prompt=prompt,
        path=child,
        doc_path=skill_doc,
        hooks_path=child / "hooks.py",
        tags=tags,
        categories=categories,
        produces=produces,
        execution_dependencies=execution_dependencies,
        allowed_tools=allowed_tools,
        required_tools=required_tools,
        command_tools=command_tools,
        entrypoints=entrypoints,
        disable_model_invocation=disable_model_invocation,
        user_invocable=user_invocable,
        argument_hint=argument_hint,
        format=vendor_flavor,
        adapter=vendor_flavor,
        frontmatter=dict(frontmatter),
        metadata=dict(metadata_raw),
        vendor_flavor=vendor_flavor,
    )
