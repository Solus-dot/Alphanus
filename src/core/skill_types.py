from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.message_types import JSONValue


@dataclass(slots=True)
class SkillContext:
    user_input: str
    branch_labels: list[str]
    attachments: list[str]
    project_root: str
    memory_hits: list[dict[str, JSONValue]]
    retrieval_hits: list[dict[str, JSONValue]] = field(default_factory=list)
    loaded_skill_ids: list[str] = field(default_factory=list)
    recent_routing_hint: str = ""
    sticky_skill_ids: list[str] = field(default_factory=list)
    explicit_skill_id: str = ""
    explicit_skill_args: str = ""
    context_summary: str = ""
    relevant_skill_ids: list[str] = field(default_factory=list)


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
