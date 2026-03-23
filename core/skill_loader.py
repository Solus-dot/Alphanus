from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from core.skill_parser import SKILL_DOC, SkillManifest, extract_skill_doc, parse_agentskill_manifest


@dataclass(slots=True)
class SkillStub:
    id: str
    name: str
    description: str
    path: Path
    doc_path: Path
    version: str
    enabled: bool
    compatibility: str
    requirements: Dict[str, List[str]]
    triggers: Dict[str, List[str]]
    tags: List[str]
    categories: List[str]
    allowed_tools: List[str]
    required_tools: List[str]
    command_tools: List[Any]
    disable_model_invocation: bool
    format: str
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    bundled_files: List[str] = field(default_factory=list)


@dataclass(slots=True)
class LoadedSkill:
    stub: SkillStub
    instructions_markdown: str
    resources: List[str]
    scripts: List[str]
    capabilities: Dict[str, Any]


def discover_skills(
    roots: Iterable[str | Path],
    *,
    on_error: Callable[[Path, Exception], None] | None = None,
) -> List[SkillStub]:
    stubs: List[SkillStub] = []
    seen_paths: set[Path] = set()

    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if root in seen_paths or not root.exists():
            continue
        seen_paths.add(root)

        for child in sorted(root.iterdir(), key=lambda path: path.name):
            if not child.is_dir():
                continue
            skill_doc = child / SKILL_DOC
            if not skill_doc.exists():
                continue
            try:
                stubs.append(_build_stub(child, skill_doc))
            except Exception as exc:
                if on_error is not None:
                    on_error(child, exc)

    return stubs


def activate_skill(stub: SkillStub) -> LoadedSkill:
    _, prompt = extract_skill_doc(stub.doc_path, include_prompt=True)
    bundled_files = list(stub.bundled_files)
    scripts = [path for path in bundled_files if path.startswith("scripts/")]
    resources = [
        path
        for path in bundled_files
        if path.startswith("references/") or path.startswith("assets/")
    ]
    capabilities = {
        "allowed_tools": list(stub.allowed_tools),
        "required_tools": list(stub.required_tools),
        "command_tools": [spec.name for spec in stub.command_tools],
        "has_tools_py": (stub.path / "tools.py").exists(),
        "has_hooks_py": (stub.path / "hooks.py").exists(),
        "bundled_files": bundled_files,
        "scripts": scripts,
        "resources": resources,
    }
    return LoadedSkill(
        stub=stub,
        instructions_markdown=prompt or "",
        resources=resources,
        scripts=scripts,
        capabilities=capabilities,
    )


def stub_to_manifest(stub: SkillStub) -> SkillManifest:
    return SkillManifest(
        id=stub.id,
        name=stub.name,
        version=stub.version,
        description=stub.description,
        enabled=stub.enabled,
        compatibility=stub.compatibility,
        requirements={key: list(value) for key, value in stub.requirements.items()},
        triggers={key: list(value) for key, value in stub.triggers.items()},
        prompt=None,
        path=stub.path,
        doc_path=stub.doc_path,
        hooks_path=stub.path / "hooks.py",
        tags=list(stub.tags),
        categories=list(stub.categories),
        allowed_tools=list(stub.allowed_tools),
        required_tools=list(stub.required_tools),
        command_tools=list(stub.command_tools),
        disable_model_invocation=stub.disable_model_invocation,
        format=stub.format,
        frontmatter=dict(stub.frontmatter),
        metadata=dict(stub.metadata),
        bundled_files=list(stub.bundled_files),
    )


def _build_stub(child: Path, skill_doc: Path) -> SkillStub:
    manifest = parse_agentskill_manifest(child, skill_doc, include_prompt=False)
    frontmatter, _ = extract_skill_doc(skill_doc, include_prompt=False)
    metadata = frontmatter.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    bundled_files = sorted(
        str(path.relative_to(child))
        for path in child.rglob("*")
        if path.is_file() and path.name != SKILL_DOC
    )

    return SkillStub(
        id=manifest.id,
        name=manifest.name,
        description=manifest.description,
        path=child,
        doc_path=skill_doc,
        version=manifest.version,
        enabled=manifest.enabled,
        compatibility=manifest.compatibility,
        requirements={key: list(value) for key, value in manifest.requirements.items()},
        triggers={key: list(value) for key, value in manifest.triggers.items()},
        tags=list(manifest.tags),
        categories=list(manifest.categories),
        allowed_tools=list(manifest.allowed_tools),
        required_tools=list(manifest.required_tools),
        command_tools=list(manifest.command_tools),
        disable_model_invocation=manifest.disable_model_invocation,
        format=manifest.format,
        frontmatter=frontmatter,
        metadata=dict(metadata),
        bundled_files=bundled_files,
    )
