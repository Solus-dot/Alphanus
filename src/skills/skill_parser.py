from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.coercion import coerce_bool

SKILL_DOC = "SKILL.md"
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(str(raw).strip() for raw in items if str(raw).strip()))


def _as_str_list(value: Any) -> list[str]:
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


def _as_tool_name_list(value: Any) -> list[str]:
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


def extract_skill_doc(skill_doc: Path, include_prompt: bool = True) -> tuple[dict[str, Any], str | None]:
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
        frontmatter: dict[str, Any] = {}
    elif isinstance(parsed, dict):
        frontmatter = parsed
    else:
        raise ValueError("SKILL.md frontmatter must be a mapping")
    return frontmatter, prompt


@dataclass(slots=True)
class SkillManifest:
    id: str
    version: str
    description: str
    enabled: bool
    requirements: dict[str, list[str]] = field(default_factory=dict)
    prompt: str | None = None
    path: Path | None = None
    doc_path: Path | None = None
    tags: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    produces: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    disable_model_invocation: bool = False
    user_invocable: bool = True
    available: bool = True
    availability_code: str = "ready"
    availability_reason: str = ""
    execution_allowed: bool = True
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    frontmatter: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    bundled_files: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)


def parse_agentskill_manifest(child: Path, skill_doc: Path, include_prompt: bool = False) -> SkillManifest:
    frontmatter, prompt = extract_skill_doc(skill_doc, include_prompt=include_prompt)
    warnings: list[str] = []

    skill_id = str(frontmatter.get("name", "")).strip()
    if not skill_id:
        raise ValueError("SKILL.md frontmatter requires 'name'")
    aliases: list[str] = []
    child_name = str(child.name).strip()
    if child_name and child_name != skill_id:
        aliases.append(child_name)

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
            frontmatter.get("produces") or frontmatter.get("artifacts") or metadata_raw.get("produces") or metadata_raw.get("artifacts")
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
        "commands": _as_str_list(requirements_raw.get("commands") or requirements_raw.get("binaries") or requirements_raw.get("bins")),
    }
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
    disable_model_invocation = coerce_bool(
        frontmatter.get(
            "disable-model-invocation",
            metadata_raw.get(
                "disable-model-invocation",
                metadata_tools_raw.get("disable-model-invocation", tools_cfg_raw.get("disable-model-invocation")),
            ),
        ),
        False,
    )
    user_invocable = coerce_bool(
        frontmatter.get("user-invocable", metadata_raw.get("user-invocable")),
        True,
    )

    enabled = coerce_bool(metadata_raw.get("enabled", frontmatter.get("enabled")), True)

    return SkillManifest(
        id=skill_id,
        version=version,
        description=description,
        enabled=enabled,
        requirements=requirements,
        prompt=prompt,
        path=child,
        doc_path=skill_doc,
        tags=tags,
        categories=categories,
        produces=produces,
        allowed_tools=allowed_tools,
        required_tools=required_tools,
        disable_model_invocation=disable_model_invocation,
        user_invocable=user_invocable,
        validation_warnings=warnings,
        frontmatter=dict(frontmatter),
        metadata=dict(metadata_raw),
        aliases=_dedupe(aliases),
    )
