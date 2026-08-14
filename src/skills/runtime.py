from __future__ import annotations

import ast
import importlib.util
import logging
import os
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import skills.skill_executor as skill_executor
import skills.skill_selector as skill_selector
import skills.skill_tool_policy as skill_tool_policy
from core.config_model import ConfigSchema, config_schema
from core.memory import LexicalMemory
from core.message_types import ApprovalRequestFn, JSONValue, UserInputRequestFn
from core.project import ProjectRuntime
from core.skill_types import SkillContext, SkillManifest
from core.tool_results import ToolResult, error_result, ok_result
from skills.skill_discovery import discover_skill_dirs, discover_skill_roots
from skills.skill_parser import SKILL_DOC, extract_skill_doc, parse_agentskill_manifest

_CORE_TOOL_NAMES = frozenset(
    {
        "shell_command",
        "read_file",
        "read_files",
        "list_files",
        "find_files",
        "search_code",
        "project_tree",
        "create_directory",
        "create_file",
        "edit_file",
        "delete_path",
    }
)

_REQUEST_USER_INPUT_TOOL_NAME = "request_user_input"
_SKILLS_LIST_TOOL_NAME = "skills_list"
_SKILL_VIEW_TOOL_NAME = "skill_view"
_ALWAYS_AVAILABLE_TOOL_NAMES = frozenset({_SKILLS_LIST_TOOL_NAME, _SKILL_VIEW_TOOL_NAME, _REQUEST_USER_INPUT_TOOL_NAME})


class ToolProtocolError(RuntimeError):
    pass


def _ok(data: object, duration_ms: int) -> ToolResult:
    return ok_result(cast(JSONValue, data), duration_ms=duration_ms)


def _err(code: str, message: str, duration_ms: int) -> ToolResult:
    return error_result(code, message, duration_ms=duration_ms)


@dataclass(slots=True)
class ToolExecutionEnv:
    project: ProjectRuntime
    memory: LexicalMemory
    config: dict[str, JSONValue]
    debug: bool
    request_approval: ApprovalRequestFn | None = None
    request_user_input: UserInputRequestFn | None = None
    stop_event: Any = None


@dataclass(slots=True)
class RegisteredTool:
    name: str
    skill_id: str
    tool_scope: str
    capability: str
    description: str
    parameters: dict[str, object]
    mutates: bool | None = None
    actions: tuple[str, ...] = ()
    module: object | None = None
    module_path: Path | None = None
    module_name: str = ""


class SkillRuntime:
    def __init__(
        self,
        skills_dir: str,
        project: ProjectRuntime,
        memory: LexicalMemory,
        config: ConfigSchema | Mapping[str, Any] | None = None,
        bundled_skills_dir: str | None = None,
        extra_skill_dirs: list[str] | None = None,
        debug: bool = False,
    ) -> None:
        self.skills_dir = Path(skills_dir).resolve()
        self.bundled_skills_dir = Path(bundled_skills_dir).resolve() if bundled_skills_dir else None
        self._configured_extra_skill_dirs = [Path(item).expanduser().resolve() for item in extra_skill_dirs or [] if str(item).strip()]
        self.project = project
        self.memory = memory
        self.debug = debug
        self.ToolExecutionEnv = ToolExecutionEnv
        self.ToolProtocolError = ToolProtocolError
        self.core_tool_names = _CORE_TOOL_NAMES
        self.always_available_tool_names = _ALWAYS_AVAILABLE_TOOL_NAMES
        self.skills_list_tool_name = _SKILLS_LIST_TOOL_NAME
        self.skill_view_tool_name = _SKILL_VIEW_TOOL_NAME
        self.request_user_input_tool_name = _REQUEST_USER_INPUT_TOOL_NAME
        self.reload_config(config or {})
        self.generation = 0

        self.skill_roots = self._discover_skill_roots()
        self.skills: dict[str, SkillManifest] = {}
        self._all_skills: list[SkillManifest] = []
        self._tool_registry: dict[str, RegisteredTool] = {}
        self._skill_alias_index: dict[str, str] = {}
        self._skill_alias_collisions: dict[str, tuple[str, ...]] = {}
        self._list_skills_cache: tuple[SkillManifest, ...] | None = None
        self._enabled_skills_cache: tuple[SkillManifest, ...] | None = None
        self._skill_catalog_cache: dict[int, str] = {}
        self._tools_schema_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        self.load_skills()

    def reload_config(self, config: ConfigSchema | Mapping[str, Any] | None) -> None:
        self.config_model = config_schema(config)
        self.config = self.config_model.model_dump()
        config_paths = self.config_model.skills.paths
        self.extra_skill_dirs = list(self._configured_extra_skill_dirs)
        for raw_path in config_paths:
            text = str(raw_path).strip()
            if text:
                path = Path(os.path.expanduser(text)).resolve()
                if path not in self.extra_skill_dirs and path != self.skills_dir and path != self.bundled_skills_dir:
                    self.extra_skill_dirs.append(path)
        permission_mode = self.config_model.permissions.mode.strip().lower()
        if permission_mode not in {"read-only", "project-write", "danger-full-access"}:
            permission_mode = "project-write"
        self.permission_mode = permission_mode
        self.project.permission_mode = permission_mode
        self.project.sandbox_config = replace(
            self.project.sandbox_config,
            mode=permission_mode,
            network=self.config_model.permissions.network,
        )

    def is_read_only_mode(self) -> bool:
        return self.permission_mode == "read-only"

    def _discover_skill_roots(self) -> list[Path]:
        roots = [self.skills_dir, *self.extra_skill_dirs]
        if self.bundled_skills_dir is not None:
            roots.append(self.bundled_skills_dir)
        return discover_skill_roots(roots)

    @staticmethod
    def _load_module(path: Path, module_name: str):
        spec = importlib.util.spec_from_file_location(module_name, str(path), submodule_search_locations=[str(path.parent)])
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module

    @staticmethod
    def _read_tool_specs(path: Path) -> dict[str, Any] | None:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            logging.debug("Failed to parse tools.py at %s: %s", path, exc)
            return None

        has_execute = False
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "execute":
                has_execute = True
                break
        if not has_execute:
            return None

        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"TOOL_SPECS", "TOOL_SPEC_ROWS"}:
                    try:
                        value = ast.literal_eval(node.value)
                    except Exception as exc:
                        if target.id == "TOOL_SPECS":
                            logging.debug("Failed to evaluate TOOL_SPECS literal at %s: %s", path, exc)
                        continue
                    if target.id == "TOOL_SPECS":
                        return value if isinstance(value, dict) else None
                    if isinstance(value, dict):
                        specs: dict[str, Any] = {}
                        for name, row in value.items():
                            if not isinstance(row, tuple) or len(row) not in {6, 7}:
                                return None
                            capability, mutates, actions, description, properties, required, *options = row
                            parameters = {"type": "object", "properties": properties}
                            include_required = bool(required) if not options else required is not None
                            if include_required:
                                parameters["required"] = list(required)
                            if not options or options[0]:
                                parameters["additionalProperties"] = False
                            specs[str(name)] = {
                                "capability": capability,
                                "mutates": mutates,
                                "actions": list(actions),
                                "description": description,
                                "parameters": parameters,
                            }
                        return specs
        return None

    @staticmethod
    def _bundled_files_for_path(root: Path) -> list[str]:
        return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.name != SKILL_DOC)

    def _load_manifest(self, child: Path) -> SkillManifest | None:
        skill_doc = child / SKILL_DOC
        if not skill_doc.exists():
            if self.debug:
                print(f"[skill] {child.name}: missing {SKILL_DOC}")
            return None
        manifest = parse_agentskill_manifest(child, skill_doc, include_prompt=False)
        manifest.bundled_files = self._bundled_files_for_path(child)
        reviewed_root = self.bundled_skills_dir or (Path(__file__).resolve().parents[2] / "bundled-skills")
        manifest.execution_allowed = child.resolve().is_relative_to(reviewed_root.resolve())
        return manifest

    def _ensure_skill_prompt(self, manifest: SkillManifest) -> str:
        if manifest.prompt is not None:
            return manifest.prompt
        if not manifest.doc_path:
            manifest.prompt = ""
            return manifest.prompt
        _, prompt = extract_skill_doc(manifest.doc_path, include_prompt=True)
        manifest.prompt = prompt
        return manifest.prompt or ""

    def _remove_skill_tools(self, skill_id: str) -> None:
        self._tool_registry = {name: reg for name, reg in self._tool_registry.items() if reg.skill_id != skill_id}

    def _invalidate_skill_caches(self) -> None:
        self._list_skills_cache = None
        self._enabled_skills_cache = None
        self._skill_catalog_cache = {}
        self._tools_schema_cache = {}

    def _refresh_skill_runtime_indexes(self) -> None:
        self._rebuild_skill_alias_index()

    @staticmethod
    def _append_unique(items: list[str], value: str) -> None:
        text = str(value).strip()
        if text and text not in items:
            items.append(text)

    def _rebuild_skill_alias_index(self) -> None:
        alias_map: dict[str, list[str]] = {}
        for skill in self.list_skills():
            aliases = {
                str(skill.id).strip(),
                *(str(item).strip() for item in (getattr(skill, "aliases", []) or [])),
            }
            for alias in aliases:
                key = alias.lower()
                if not key:
                    continue
                alias_map.setdefault(key, []).append(skill.id)
        self._skill_alias_index = {}
        self._skill_alias_collisions = {}
        for key, ids in alias_map.items():
            unique_ids = tuple(sorted(dict.fromkeys(ids)))
            if len(unique_ids) == 1:
                self._skill_alias_index[key] = unique_ids[0]
            elif len(unique_ids) > 1:
                self._skill_alias_collisions[key] = unique_ids

    def _register_tool(
        self,
        tool_name: str,
        manifest: SkillManifest,
        spec: dict[str, Any],
        *,
        soft: bool = False,
        **extra: Any,
    ) -> bool:
        warning_sink = manifest.validation_warnings if soft else manifest.validation_errors
        if tool_name in self._tool_registry:
            self._append_unique(warning_sink, f"duplicate tool '{tool_name}' already registered")
            return False
        capability = str(spec.get("capability", "")).strip()
        description = str(spec.get("description", "")).strip()
        parameters = spec.get("parameters")
        if not capability or not description or not isinstance(parameters, dict):
            self._append_unique(warning_sink, f"invalid tool spec '{tool_name}'")
            return False
        mutates_raw = spec.get("mutates")
        actions_raw = spec.get("actions", [])
        self._tool_registry[tool_name] = RegisteredTool(
            name=tool_name,
            skill_id=manifest.id,
            tool_scope="core" if tool_name in _CORE_TOOL_NAMES else "skill",
            capability=capability,
            description=description,
            parameters=parameters,
            mutates=mutates_raw if isinstance(mutates_raw, bool) else None,
            actions=tuple(dict.fromkeys(str(item).strip().lower() for item in actions_raw if str(item).strip()))
            if isinstance(actions_raw, list)
            else (),
            **extra,
        )
        return True

    def _register_runtime_tools(self) -> None:
        def register(name: str, scope: str, capability: str, description: str, mutates: bool, actions, properties, **schema) -> None:
            self._tool_registry[name] = RegisteredTool(
                name=name,
                skill_id="__runtime__",
                tool_scope=scope,
                capability=capability,
                description=description,
                mutates=mutates,
                actions=actions,
                parameters={"type": "object", "properties": properties, **schema},
            )

        register(
            _SKILLS_LIST_TOOL_NAME,
            "core",
            "skill_catalog_reader",
            "List available skills with minimal metadata. Use this to discover a relevant skill before loading it.",
            False,
            ("list", "read"),
            {},
        )
        register(
            _SKILL_VIEW_TOOL_NAME,
            "core",
            "skill_loader",
            "Load a skill's full SKILL.md content or read one linked file inside the skill.",
            False,
            ("read",),
            {"name": {"type": "string"}, "file_path": {"type": "string"}},
            required=["name"],
        )
        register(
            _REQUEST_USER_INPUT_TOOL_NAME,
            "skill",
            "user_input_requester",
            "Ask the user a structured follow-up question and pause the current workflow.",
            False,
            (),
            {
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}},
                "header": {"type": "string"},
            },
            required=["question"],
        )

    def load_skills(self) -> None:
        runtime = self
        previous_enabled = {skill_id: skill.enabled for skill_id, skill in runtime.skills.items()}
        runtime.generation += 1
        runtime.skill_roots = runtime._discover_skill_roots()
        runtime.skills = {}
        runtime._all_skills = []
        runtime._tool_registry = {}
        runtime._skill_alias_index = {}
        runtime._skill_alias_collisions = {}
        runtime._invalidate_skill_caches()
        runtime._register_runtime_tools()
        if not any(root.exists() for root in runtime.skill_roots):
            runtime._refresh_skill_runtime_indexes()
            return

        for root in runtime.skill_roots:
            if not root.exists():
                continue
            for child in discover_skill_dirs(root):
                manifest: SkillManifest | None = None
                try:
                    manifest = runtime._load_manifest(child)
                    if manifest is None:
                        continue

                    if manifest.id in previous_enabled:
                        manifest.enabled = previous_enabled[manifest.id]

                    (
                        manifest.available,
                        manifest.availability_code,
                        manifest.availability_reason,
                    ) = runtime._check_skill_availability(manifest)

                    existing = runtime.skills.get(manifest.id)
                    if existing is not None:
                        source = runtime.skill_source_label(existing) or existing.id
                        incoming = runtime.skill_source_label(manifest) or manifest.id
                        runtime._append_unique(
                            existing.validation_errors,
                            f"duplicate skill id '{manifest.id}' ignored from {incoming}; using {source}",
                        )
                        continue

                    if manifest.available and manifest.execution_allowed and not runtime._load_skill_tools(manifest):
                        manifest.available = False
                        manifest.execution_allowed = False
                        if not manifest.availability_code or manifest.availability_code == "ready":
                            manifest.availability_code = "invalid"
                        if not manifest.availability_reason:
                            manifest.availability_reason = (
                                manifest.validation_errors[0] if manifest.validation_errors else "skill load failed"
                            )

                    runtime.skills[manifest.id] = manifest
                    runtime._all_skills.append(manifest)
                except Exception as exc:
                    runtime._remove_skill_tools(manifest.id if manifest else child.name)
                    if manifest is not None:
                        runtime._append_unique(manifest.validation_errors, str(exc))
                        manifest.available = False
                        manifest.execution_allowed = False
                        manifest.availability_code = "invalid"
                        manifest.availability_reason = str(exc)
                        runtime.skills[manifest.id] = manifest
                        runtime._all_skills.append(manifest)
                    elif runtime.debug:
                        print(f"[skill] failed to load {child.name}: {exc}")
        runtime._refresh_skill_runtime_indexes()

    @staticmethod
    def _current_os_aliases() -> set[str]:
        aliases = {sys.platform.lower(), os.name.lower()}
        if sys.platform.startswith("darwin"):
            aliases.update({"darwin", "mac", "macos", "osx"})
        elif sys.platform.startswith("linux"):
            aliases.update({"linux", "posix"})
        return aliases

    def _check_skill_availability(self, manifest: SkillManifest) -> tuple[bool, str, str]:
        requirements = manifest.requirements if isinstance(manifest.requirements, dict) else {}

        required_os = [item.lower() for item in requirements.get("os", []) if str(item).strip()]
        if required_os:
            aliases = self._current_os_aliases()
            if not any(item in aliases for item in required_os):
                return False, "os", f"requires os: {', '.join(required_os)}"

        missing_env = [name for name in requirements.get("env", []) if name and not os.environ.get(name)]
        if missing_env:
            return False, "env", f"missing env: {', '.join(missing_env)}"

        missing_commands = [name for name in requirements.get("commands", []) if name and shutil.which(name) is None]
        if missing_commands:
            return False, "command", f"missing commands: {', '.join(missing_commands)}"

        return True, "ready", ""

    def _load_skill_tools(self, manifest: SkillManifest) -> bool:
        if not manifest.path:
            return not manifest.required_tools

        allowed_tools = set(manifest.allowed_tools)
        local_registered: list[str] = []
        local_seen: set[str] = set()

        tools_path = manifest.path / "tools.py"
        if tools_path.exists():
            specs = self._read_tool_specs(tools_path)
            if not isinstance(specs, dict):
                self._append_unique(manifest.validation_errors, "tools.py missing TOOL_SPECS dict or execute()")
                return False

            module_name = f"alphanus_tools_{manifest.id.replace('-', '_')}"

            for tool_name, spec in specs.items():
                if allowed_tools and tool_name not in allowed_tools:
                    continue
                if tool_name in local_seen:
                    self._append_unique(manifest.validation_errors, f"duplicate tool '{tool_name}' within skill")
                    continue
                local_seen.add(tool_name)
                if self._register_tool(
                    tool_name,
                    manifest,
                    spec,
                    module_path=tools_path,
                    module_name=module_name,
                ):
                    local_registered.append(tool_name)

        if manifest.required_tools:
            local_set = set(local_registered)
            missing = [name for name in manifest.required_tools if name not in local_set]
            if missing:
                for name in local_registered:
                    self._tool_registry.pop(name, None)
                self._append_unique(manifest.validation_errors, f"missing required tools: {', '.join(missing)}")
                return False

        if manifest.validation_errors:
            for name in local_registered:
                self._tool_registry.pop(name, None)
            return False

        return True

    def list_skills(self) -> list[SkillManifest]:
        if self._list_skills_cache is None:
            self._list_skills_cache = tuple(
                sorted(
                    self._all_skills,
                    key=lambda s: (s.id, self.skill_source_label(s)),
                )
            )
        return list(self._list_skills_cache)

    def _skill_location(self, skill: SkillManifest) -> tuple[str, str]:
        path = skill.path or skill.doc_path
        if not path:
            return "unknown", ""
        try:
            resolved = path.resolve()
            locations = [(self.skills_dir, "user/skills", "skills")]
            locations.extend((root, "configured", "configured") for root in self.extra_skill_dirs)
            if self.bundled_skills_dir is not None:
                locations.append((self.bundled_skills_dir, "bundled", "bundled-skills"))
            locations.append((self.project.project_root, "project", "project"))
            for root, provenance, prefix in locations:
                root = root.resolve()
                if resolved.is_relative_to(root):
                    return provenance, f"{prefix}/{resolved.relative_to(root)}"
            return "external", str(resolved)
        except Exception:
            return "external", str(path)

    def skill_source_label(self, skill: SkillManifest) -> str:
        return self._skill_location(skill)[1]

    def skill_provenance_label(self, skill: SkillManifest) -> str:
        return self._skill_location(skill)[0]

    @staticmethod
    def skill_status_label(skill: SkillManifest) -> tuple[str, str]:
        if not skill.available:
            return "blocked", "yellow"
        if skill.enabled:
            return "on", "green"
        return "off", "red"

    def enabled_skills(self) -> list[SkillManifest]:
        if self._enabled_skills_cache is None:
            self._enabled_skills_cache = tuple(skill for skill in self.skills.values() if skill.enabled and skill.available)
        return list(self._enabled_skills_cache)

    def skills_by_ids(self, skill_ids: list[str]) -> list[SkillManifest]:
        out: list[SkillManifest] = []
        seen = set()
        for skill_id in skill_ids:
            key = str(skill_id).strip()
            if not key or key in seen:
                continue
            skill = self.skills.get(key)
            if not skill or not skill.enabled or not skill.available:
                continue
            out.append(skill)
            seen.add(key)
        return out

    def skill_catalog_text(self, max_tags: int = 3) -> str:
        cached = self._skill_catalog_cache.get(max_tags)
        if cached is not None:
            return cached
        catalog = self.skill_catalog_text_for(self.enabled_skills(), max_tags=max_tags)
        self._skill_catalog_cache[max_tags] = catalog
        return catalog

    def compose_skill_index(self, ctx: SkillContext | None = None, *, full: bool = False) -> str:
        if full or ctx is None:
            skills = sorted(self.enabled_skills(), key=lambda item: item.id)
        else:
            skill_ids: list[str] = []
            for source in (
                getattr(ctx, "loaded_skill_ids", []) or [],
                getattr(ctx, "relevant_skill_ids", []) or [],
                getattr(ctx, "sticky_skill_ids", []) or [],
            ):
                for skill_id in source:
                    normalized = str(skill_id).strip()
                    if normalized and normalized not in skill_ids:
                        skill_ids.append(normalized)
            skills = self.skills_by_ids(skill_ids)
            if not skills:
                skills = sorted(self.enabled_skills(), key=lambda item: item.id)
        if not skills:
            return ""

        lines = [f"  - {skill.id}: {skill.description}" for skill in skills]
        title = "## Skills (mandatory)" if full or ctx is None else "## Relevant skills (mandatory)"
        return (
            f"{title}\n"
            "Before replying, scan the skills below. If one clearly matches the task, "
            "load it with skill_view(name) and follow its instructions.\n"
            "If the user asks to create, edit, save, read, or manage local files and a listed skill matches that work, "
            "load that skill before saying file tools are unavailable.\n"
            "Do not call namespaced skill tools like skill-name:tool_name. Skill tools are only usable after "
            "skill_view loads the skill and exposes their exact unqualified function names in the current tool list.\n"
            "If none match, proceed normally without loading a skill.\n\n"
            "<available_skills>\n" + "\n".join(lines) + "\n</available_skills>"
        )

    def model_exposed_tool_names(self) -> list[str]:
        # Skill-native tools are exposed only through turn selection
        # (loaded skills), not globally through enablement alone.
        return sorted(name for name in _ALWAYS_AVAILABLE_TOOL_NAMES if name in self._tool_registry)

    def skill_catalog_text_for(self, skills: list[SkillManifest], max_tags: int = 3) -> str:
        lines: list[str] = []
        for skill in skills:
            tags = ", ".join(skill.tags[:max_tags])
            tag_text = f" tags: {tags}." if tags else ""
            tools = ", ".join(skill.allowed_tools[:4])
            tool_text = f" tools: {tools}." if tools else ""
            produces = ", ".join(skill.produces[:3])
            produce_text = f" produces: {produces}." if produces else ""
            tier = self.skill_provenance_label(skill)
            location = self.skill_source_label(skill)
            location_text = f" location: {location}." if location else ""
            lines.append(f"- {skill.id}: {skill.description}. source: {tier}.{tag_text}{produce_text}{tool_text}{location_text}")
        return "\n".join(lines)

    def skill_health_report(self) -> list[dict[str, Any]]:
        return [
            {
                "id": skill.id,
                "name": skill.id,
                "enabled": bool(skill.enabled),
                "available": bool(skill.available),
                "provenance": self.skill_provenance_label(skill),
                "execution_allowed": bool(skill.execution_allowed),
                "adapter": "agentskills",
                "status": self.skill_status_label(skill)[0],
                "source": self.skill_source_label(skill),
                "availability_code": skill.availability_code or "ready",
                "availability_reason": skill.availability_reason or "ready",
                "tools": self._reported_skill_tools(skill),
                "scripts": [],
                "entrypoints": [],
                "user_invocable": bool(skill.user_invocable),
                "model_invocable": not bool(skill.disable_model_invocation),
                "validation_errors": list(skill.validation_errors),
                "validation_warnings": list(getattr(skill, "validation_warnings", []) or []),
                "aliases": list(getattr(skill, "aliases", []) or []),
                "alias_conflicts": sorted(alias for alias, ids in self._skill_alias_collisions.items() if skill.id in ids),
            }
            for skill in self.list_skills()
        ]

    def set_enabled(self, skill_id: str, enabled: bool) -> bool:
        skill = self.skills.get(skill_id)
        if not skill:
            return False
        if skill.enabled == enabled:
            return True
        skill.enabled = enabled
        self.generation += 1
        self._invalidate_skill_caches()
        self._refresh_skill_runtime_indexes()
        return True

    def get_skill(self, skill_id: str) -> SkillManifest | None:
        return self.skills.get(skill_id)

    def resolve_skill_reference(self, skill_ref: str) -> SkillManifest | None:
        ref = str(skill_ref).strip()
        if not ref:
            return None
        exact = self.get_skill(ref)
        if exact is not None:
            return exact
        lowered = ref.lower()
        collided_ids = self._skill_alias_collisions.get(lowered)
        if collided_ids:
            return None
        alias_hit = self._skill_alias_index.get(lowered)
        if alias_hit:
            resolved = self.get_skill(alias_hit)
            if resolved is not None:
                return resolved
        return None

    def _reported_skill_tools(self, skill: SkillManifest) -> list[str]:
        if not getattr(skill, "execution_allowed", True):
            return []
        return sorted(reg.name for reg in self._tool_registry.values() if reg.skill_id == skill.id)

    def select_skills(self, ctx: SkillContext, top_n: int = 3) -> list[SkillManifest]:
        return skill_selector.select_skills(self, ctx, top_n=top_n)

    def tool_registration(self, tool_name: str) -> RegisteredTool | None:
        return self._tool_registry.get(str(tool_name).strip())

    def tool_is_mutating(self, tool_name: str) -> bool:
        return skill_tool_policy.is_mutating(self, tool_name)

    def tool_is_blocked_for_local_project(self, tool_name: str) -> bool:
        return skill_tool_policy.is_blocked_for_local_project(self, tool_name)

    def compose_skill_block(
        self,
        selected: list[SkillManifest],
        context_limit: int,
        ratio: float = 0.15,
        hard_cap: int = 0,
    ) -> str:
        if hard_cap > 0:
            selected = selected[:hard_cap]
        if not selected:
            return ""

        sections: list[str] = []
        for skill in selected:
            body = self._ensure_skill_prompt(skill).strip()
            sections.append(f"### Skill: {skill.id}\n{body}")

        budget = max(200, int(context_limit * ratio * 4))
        out: list[str] = []
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
            snippet = self._safe_prompt_snippet(body, allowed)
            out.append(head + "\n" + snippet)
            used = budget

        return "\n\n".join(out)

    def _skill_linked_files(self, skill: SkillManifest) -> dict[str, list[str]]:
        linked: dict[str, list[str]] = {}
        for rel in sorted(skill.bundled_files):
            parts = Path(rel).parts
            key = parts[0] if len(parts) > 1 else "root"
            linked.setdefault(key, []).append(rel)
        return linked

    def skills_list(self) -> dict[str, Any]:
        skills = [
            {
                "name": skill.id,
                "description": skill.description,
            }
            for skill in sorted(self.enabled_skills(), key=lambda item: item.id)
        ]
        return {
            "skills": skills,
            "count": len(skills),
            "hint": "Use skill_view(name) to load a skill's full instructions or inspect one linked file.",
        }

    def _resolve_enabled_skill(self, name: str) -> SkillManifest:
        key = str(name or "").strip()
        if not key:
            raise ValueError("Missing required argument: name")
        collided_ids = self._skill_alias_collisions.get(key.lower())
        if collided_ids:
            rendered = ", ".join(collided_ids[:4])
            raise FileNotFoundError(f"Skill reference '{key}' is ambiguous ({rendered})")
        skill = self.resolve_skill_reference(key)
        if skill is None or not skill.enabled or not skill.available:
            raise FileNotFoundError(f"Skill '{key}' not found")
        return skill

    def skill_view(self, name: str, file_path: str, ctx: SkillContext) -> dict[str, Any]:
        skill = self._resolve_enabled_skill(name)
        relpath = str(file_path or "").strip()
        if not relpath:
            if skill.id not in ctx.loaded_skill_ids:
                ctx.loaded_skill_ids.append(skill.id)
            return {
                "skill_id": skill.id,
                "name": skill.id,
                "description": skill.description,
                "content": self._ensure_skill_prompt(skill).strip(),
                "linked_files": self._skill_linked_files(skill),
                "loaded": True,
                "loaded_skill_ids": list(ctx.loaded_skill_ids),
            }
        if not skill.path:
            raise FileNotFoundError(f"Skill root unavailable: {skill.id}")
        target = (skill.path / relpath).resolve()
        if not target.is_relative_to(skill.path.resolve()):
            raise PermissionError("Skill file path escapes skill root")
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"Skill file not found: {relpath}")
        return {
            "skill_id": skill.id,
            "name": skill.id,
            "file_path": relpath,
            "content": target.read_text(encoding="utf-8"),
            "loaded": False,
            "loaded_skill_ids": list(ctx.loaded_skill_ids),
        }

    @staticmethod
    def _safe_prompt_snippet(body: str, allowed: int) -> str:
        if allowed <= 0 or not body:
            return ""
        if len(body) <= allowed:
            return body.rstrip()

        kept: list[str] = []
        used = 0
        fence_balance = 0
        for line in body.splitlines():
            addition = len(line) if not kept else len(line) + 1
            if used + addition > allowed:
                break
            kept.append(line)
            used += addition
            if line.lstrip().startswith("```"):
                fence_balance += 1

        while kept and fence_balance % 2 == 1:
            removed = kept.pop()
            if removed.lstrip().startswith("```"):
                fence_balance -= 1

        if kept:
            return "\n".join(kept).rstrip()

        clipped = body[:allowed].rstrip()
        if clipped.count("```") % 2 == 1:
            clipped = clipped.rsplit("```", 1)[0].rstrip()
        return clipped

    def core_tool_names_for_turn(
        self,
        selected: list[SkillManifest],
        ctx: SkillContext | None = None,
    ) -> list[str]:
        return skill_tool_policy.core_names(self, selected, ctx=ctx)

    def optional_tool_names(self, selected: list[SkillManifest], ctx: SkillContext | None = None) -> list[str]:
        return skill_tool_policy.optional_names(self, selected, ctx=ctx)

    def allowed_tool_names(
        self,
        selected: list[SkillManifest],
        ctx: SkillContext | None = None,
    ) -> list[str]:
        return skill_tool_policy.allowed_names(self, selected, ctx=ctx)

    def tools_for_turn(
        self,
        selected: list[SkillManifest],
        ctx: SkillContext | None = None,
    ) -> list[dict[str, Any]]:
        names = self.allowed_tool_names(selected, ctx=ctx)
        selected_ids = tuple(skill.id for skill in selected)
        cache_key = (self.generation, selected_ids, tuple(names))
        if cache_key not in self._tools_schema_cache:
            self._tools_schema_cache[cache_key] = [
                {
                    "type": "function",
                    "function": {"name": reg.name, "description": reg.description, "parameters": reg.parameters},
                }
                for name in names
                for reg in [self._tool_registry[name]]
            ]
        return self._tools_schema_cache[cache_key]

    def execute_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        selected: list[SkillManifest],
        ctx: SkillContext,
        request_approval: ApprovalRequestFn | None = None,
        request_user_input: UserInputRequestFn | None = None,
        stop_event: Any = None,
    ) -> dict[str, Any]:
        return skill_executor.execute_tool_call(
            self,
            tool_name,
            args,
            selected,
            ctx,
            request_approval=request_approval,
            request_user_input=request_user_input,
            stop_event=stop_event,
        )
