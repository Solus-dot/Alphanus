from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

SKILL_DOC = "SKILL.md"
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")


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


def extract_skill_doc(skill_doc: Path) -> Tuple[Dict[str, Any], str]:
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
    timeout_s: int = 30
    confirm_arg: str = ""


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
    command_tools: List[ToolCommandDef] = field(default_factory=list)
    disable_model_invocation: bool = False
    format: str = "agentskills"


def parse_agentskill_manifest(child: Path, skill_doc: Path) -> SkillManifest:
    frontmatter, prompt = extract_skill_doc(skill_doc)

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
        metadata_tools_raw.get("disable-model-invocation", tools_cfg_raw.get("disable-model-invocation")),
        False,
    )

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
        triggers={"keywords": keywords, "file_ext": file_ext},
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
