from __future__ import annotations

import ast
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from core.memory import LexicalMemory
from core.message_types import JSONValue, ShellConfirmationFn, UserInputRequestFn
from core.runtime_config import SkillsRuntimeConfig
from core.tool_results import ToolResult, error_result, ok_result
from core.workspace import WorkspaceManager
from skills.skill_discovery import SkillDiscovery
from skills.skill_executor import SkillExecutor
from skills.skill_inventory import SkillInventoryLoader
from skills.skill_parser import SKILL_DOC, SkillEntrypointDef, SkillManifest, extract_skill_doc, parse_agentskill_manifest
from skills.skill_process_env import SkillProcessEnvBuilder
from skills.skill_registry import SkillRegistry
from skills.skill_run_validation import SkillRunValidator
from skills.skill_script_inspector import SkillScriptInspector
from skills.skill_selector import SkillSelector
from skills.skill_tool_schema import SkillToolSchemaBuilder

_CORE_TOOL_NAMES = frozenset(
    {
        "shell_command",
        "read_file",
        "read_files",
        "list_files",
        "search_code",
        "workspace_tree",
        "create_directory",
        "create_file",
        "edit_file",
        "delete_path",
    }
)

_RUN_SKILL_TOOL_NAME = "run_skill"
_REQUEST_USER_INPUT_TOOL_NAME = "request_user_input"
_SKILLS_LIST_TOOL_NAME = "skills_list"
_SKILL_VIEW_TOOL_NAME = "skill_view"
_ALWAYS_AVAILABLE_TOOL_NAMES = frozenset(
    {
        _SKILLS_LIST_TOOL_NAME,
        _SKILL_VIEW_TOOL_NAME,
        _REQUEST_USER_INPUT_TOOL_NAME,
    }
)


class ToolProtocolError(RuntimeError):
    pass


def _ok(data: object, duration_ms: int) -> ToolResult:
    return ok_result(cast(JSONValue, data), duration_ms=duration_ms)


def _err(code: str, message: str, duration_ms: int) -> ToolResult:
    return error_result(code, message, duration_ms=duration_ms)


def _recover_tool_args(raw_args: dict[str, object]) -> dict[str, object]:
    if not isinstance(raw_args, dict):
        return {}
    out = dict(raw_args)
    raw = out.get("_raw")
    if not isinstance(raw, str) or not raw.strip():
        return out
    text = raw.strip()

    parsed: dict[str, object] | None = None
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
    branch_labels: list[str]
    attachments: list[str]
    workspace_root: str
    memory_hits: list[dict[str, JSONValue]]
    retrieval_hits: list[dict[str, JSONValue]] = field(default_factory=list)
    loaded_skill_ids: list[str] = field(default_factory=list)
    recent_routing_hint: str = ""
    sticky_skill_ids: list[str] = field(default_factory=list)
    explicit_skill_id: str = ""
    explicit_skill_args: str = ""


@dataclass(slots=True)
class ToolExecutionEnv:
    workspace: WorkspaceManager
    memory: LexicalMemory
    config: dict[str, JSONValue]
    debug: bool
    confirm_shell: ShellConfirmationFn | None = None
    request_user_input: UserInputRequestFn | None = None


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
    cwd: str = ""
    command_template: str = ""
    timeout_s: int = 30


class SkillRuntime:
    def __init__(
        self,
        skills_dir: str,
        workspace: WorkspaceManager,
        memory: LexicalMemory,
        config: dict | None = None,
        bundled_skills_dir: str | None = None,
        extra_skill_dirs: list[str] | None = None,
        debug: bool = False,
    ) -> None:
        self.skills_dir = Path(skills_dir).resolve()
        self.bundled_skills_dir = Path(bundled_skills_dir).resolve() if bundled_skills_dir else None
        self._configured_extra_skill_dirs = [Path(item).expanduser().resolve() for item in extra_skill_dirs or [] if str(item).strip()]
        self.workspace = workspace
        self.memory = memory
        self.config = {}
        self.runtime_config = SkillsRuntimeConfig()
        self.debug = debug
        self.skills_cfg = {}
        self.tools_cfg = {}
        self.python_executable = sys.executable
        self.ToolExecutionEnv = ToolExecutionEnv
        self.reload_config(config or {})
        self.generation = 0

        self.skill_roots = self._discover_skill_roots()
        self.skills: dict[str, SkillManifest] = {}
        self._all_skills: list[SkillManifest] = []
        self._tool_registry: dict[str, RegisteredTool] = {}
        self._skill_alias_index: dict[str, str] = {}
        self._skill_alias_collisions: dict[str, tuple[str, ...]] = {}
        self._skill_prefix_index: dict[str, str] = {}
        self._skill_fuzzy_index: dict[str, str] = {}
        self._skill_fuzzy_collisions: dict[str, tuple[str, ...]] = {}
        self._list_skills_cache: tuple[SkillManifest, ...] | None = None
        self._enabled_skills_cache: tuple[SkillManifest, ...] | None = None
        self._skill_catalog_cache: dict[int, str] = {}
        self._runnable_scripts_cache: dict[tuple[str, str], tuple[str, ...]] = {}
        self._tools_schema_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        self._python_module_probe_cache: dict[str, bool] = {}
        self._script_inspector = SkillScriptInspector(self)
        self._tool_schema_builder = SkillToolSchemaBuilder(self, run_skill_tool_name=_RUN_SKILL_TOOL_NAME)
        self._inventory_loader = SkillInventoryLoader(self)
        self._skill_executor = SkillExecutor(
            self,
            skills_list_tool_name=_SKILLS_LIST_TOOL_NAME,
            skill_view_tool_name=_SKILL_VIEW_TOOL_NAME,
            request_user_input_tool_name=_REQUEST_USER_INPUT_TOOL_NAME,
            run_skill_tool_name=_RUN_SKILL_TOOL_NAME,
            ok_fn=_ok,
            err_fn=_err,
            protocol_error_cls=ToolProtocolError,
        )
        self._run_validator = SkillRunValidator(self)
        self._proc_env_base = self._build_proc_env_base()
        self.selector = SkillSelector(self)
        self.load_skills()

    def reload_config(self, config: dict | None) -> None:
        self.config = config or {}
        self.skills_cfg = self.config.get("skills", {}) if isinstance(self.config.get("skills"), dict) else {}
        self.tools_cfg = self.config.get("tools", {}) if isinstance(self.config.get("tools"), dict) else {}
        self.runtime_config = SkillsRuntimeConfig.from_config(self.config)
        self.python_executable = self.runtime_config.python_executable
        raw_paths = self.skills_cfg.get("paths", [])
        config_paths = raw_paths if isinstance(raw_paths, list) else []
        self.extra_skill_dirs = list(self._configured_extra_skill_dirs)
        for raw_path in config_paths:
            text = str(raw_path).strip()
            if text:
                path = Path(os.path.expanduser(text)).resolve()
                if path not in self.extra_skill_dirs and path != self.skills_dir and path != self.bundled_skills_dir:
                    self.extra_skill_dirs.append(path)
        runtime_cfg = self.config.get("runtime", {}) if isinstance(self.config.get("runtime"), dict) else {}
        profile_raw = str(runtime_cfg.get("profile", "standard")).strip().lower()
        self.runtime_profile = "minimal" if profile_raw in {"minimal", "safe", "minimal_reliable"} else "standard"
        capabilities_cfg = self.config.get("capabilities", {}) if isinstance(self.config.get("capabilities"), dict) else {}
        permission_profile_raw = str(capabilities_cfg.get("permission_profile", "full")).strip().lower()
        if permission_profile_raw in {"safe", "minimal", "readonly", "read-only"}:
            self.permission_profile = "safe"
        elif permission_profile_raw in {"workspace", "standard"}:
            self.permission_profile = "workspace"
        else:
            self.permission_profile = "full"

    def refresh_process_env(self) -> None:
        self._proc_env_base = self._build_proc_env_base()

    def is_minimal_profile(self) -> bool:
        return self.runtime_profile == "minimal"

    def _build_proc_env_base(self) -> dict[str, str]:
        return SkillProcessEnvBuilder.build_base_env(
            workspace_root=self.workspace.workspace_root,
            home_root=self.workspace.home_root,
            memory_path=self.memory.storage_path,
            python_executable=self.python_executable,
            skills_dir=self.skills_dir,
            bundled_skills_dir=self.bundled_skills_dir,
            config=self.config,
        )

    def _discover_skill_roots(self) -> list[Path]:
        roots = [self.skills_dir, *self.extra_skill_dirs]
        if self.bundled_skills_dir is not None:
            roots.append(self.bundled_skills_dir)
        return SkillDiscovery.discover_skill_roots(roots)

    def _discover_skill_dirs(self, root: Path) -> list[Path]:
        return SkillDiscovery.discover_skill_dirs(root, is_relative_to=self._is_relative_to)

    @staticmethod
    def _load_module(path: Path, module_name: str):
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
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
                if isinstance(target, ast.Name) and target.id == "TOOL_SPECS":
                    try:
                        value = ast.literal_eval(node.value)
                    except Exception as exc:
                        logging.debug("Failed to evaluate TOOL_SPECS literal at %s: %s", path, exc)
                        return None
                    return value if isinstance(value, dict) else None
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
        manifest.execution_allowed = True
        manifest.adapter = str(manifest.adapter or manifest.format or "agentskills")
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
        SkillRegistry.remove_skill_tools(self._tool_registry, skill_id)

    def _invalidate_skill_caches(self) -> None:
        self._list_skills_cache = None
        self._enabled_skills_cache = None
        self._skill_catalog_cache = {}
        self._runnable_scripts_cache = {}
        self._tools_schema_cache = {}

    def _refresh_skill_runtime_indexes(self) -> None:
        for skill in self.enabled_skills():
            self._skill_runnable_scripts(skill)
        self._rebuild_skill_alias_index()

    @staticmethod
    def _append_unique(items: list[str], value: str) -> None:
        text = str(value).strip()
        if text and text not in items:
            items.append(text)

    def _rebuild_skill_alias_index(self) -> None:
        alias_map: dict[str, list[str]] = {}
        prefix_map: dict[str, list[str]] = {}
        fuzzy_map: dict[str, list[str]] = {}
        for skill in self.list_skills():
            aliases = {
                str(skill.id).strip(),
                str(skill.name).strip(),
                *(str(item).strip() for item in (getattr(skill, "aliases", []) or [])),
            }
            for alias in aliases:
                key = alias.lower()
                if not key:
                    continue
                alias_map.setdefault(key, []).append(skill.id)
                for idx in range(1, len(key) + 1):
                    prefix_map.setdefault(key[:idx], []).append(skill.id)
                normalized = re.sub(r"[^a-z0-9]+", "", key)
                if normalized:
                    fuzzy_map.setdefault(normalized, []).append(skill.id)
        self._skill_alias_index = {}
        self._skill_alias_collisions = {}
        for key, ids in alias_map.items():
            unique_ids = tuple(sorted(dict.fromkeys(ids)))
            if len(unique_ids) == 1:
                self._skill_alias_index[key] = unique_ids[0]
            elif len(unique_ids) > 1:
                self._skill_alias_collisions[key] = unique_ids
        self._skill_prefix_index = {
            key: unique_ids[0] for key, ids in prefix_map.items() if len(unique_ids := tuple(sorted(dict.fromkeys(ids)))) == 1
        }
        self._skill_fuzzy_index = {}
        self._skill_fuzzy_collisions = {}
        for key, ids in fuzzy_map.items():
            unique_ids = tuple(sorted(dict.fromkeys(ids)))
            if len(unique_ids) == 1:
                self._skill_fuzzy_index[key] = unique_ids[0]
            elif len(unique_ids) > 1:
                self._skill_fuzzy_collisions[key] = unique_ids

    def _register_tool(
        self,
        tool_name: str,
        manifest: SkillManifest,
        spec: dict[str, Any],
        *,
        soft: bool = False,
        **extra: Any,
    ) -> bool:
        return SkillRegistry.register_tool(
            tool_registry=self._tool_registry,
            registered_tool_cls=RegisteredTool,
            tool_name=tool_name,
            manifest=manifest,
            tool_scope_for_name=lambda name: "core" if name in _CORE_TOOL_NAMES else "skill",
            append_unique=self._append_unique,
            spec=spec,
            extra=extra,
            soft=soft,
        )

    def _register_runtime_tools(self) -> None:
        self._tool_registry[_SKILLS_LIST_TOOL_NAME] = RegisteredTool(
            name=_SKILLS_LIST_TOOL_NAME,
            skill_id="__runtime__",
            tool_scope="core",
            capability="skill_catalog_reader",
            description="List available skills with minimal metadata. Use this to discover a relevant skill before loading it.",
            mutates=False,
            actions=("list", "read"),
            parameters={
                "type": "object",
                "properties": {},
            },
        )
        self._tool_registry[_SKILL_VIEW_TOOL_NAME] = RegisteredTool(
            name=_SKILL_VIEW_TOOL_NAME,
            skill_id="__runtime__",
            tool_scope="core",
            capability="skill_loader",
            description="Load a skill's full SKILL.md content or read one linked file inside the skill.",
            mutates=False,
            actions=("read",),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "file_path": {"type": "string"},
                },
                "required": ["name"],
            },
        )
        self._tool_registry[_REQUEST_USER_INPUT_TOOL_NAME] = RegisteredTool(
            name=_REQUEST_USER_INPUT_TOOL_NAME,
            skill_id="__runtime__",
            tool_scope="skill",
            capability="user_input_requester",
            description="Ask the user a structured follow-up question and pause the current workflow.",
            mutates=False,
            actions=(),
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "header": {"type": "string"},
                },
                "required": ["question"],
            },
        )
        self._tool_registry[_RUN_SKILL_TOOL_NAME] = RegisteredTool(
            name=_RUN_SKILL_TOOL_NAME,
            skill_id="__runtime__",
            tool_scope="skill",
            capability="skill_executor",
            description="Run the selected skill's single declared executable path. Use either an entrypoint or a bundled script, not both.",
            mutates=True,
            actions=("run",),
            parameters={
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string"},
                    "entrypoint": {"type": "string"},
                    "script": {"type": "string"},
                    "params": {"type": "object"},
                    "argv": {"type": "array", "items": {"type": "string"}},
                    "stdin": {"type": "string"},
                    "timeout_s": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        )

    def load_skills(self) -> None:
        self._inventory_loader.load_skills()

    @staticmethod
    def _current_os_aliases() -> set[str]:
        aliases = {sys.platform.lower(), os.name.lower()}
        if sys.platform.startswith("darwin"):
            aliases.update({"darwin", "mac", "macos", "osx"})
        elif sys.platform.startswith("linux"):
            aliases.update({"linux", "posix"})
        elif sys.platform.startswith("win"):
            aliases.update({"windows", "win32", "nt"})
        return aliases

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

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
                    cwd=str(manifest.path),
                ):
                    local_registered.append(tool_name)

        for definition in list(getattr(manifest, "tool_definitions", []) or []):
            if not isinstance(definition, dict):
                continue
            tool_name = str(definition.get("name", "")).strip()
            if not tool_name:
                continue
            spec = definition.get("spec")
            if not isinstance(spec, dict):
                continue
            command_template = str(definition.get("command", "")).strip()
            if not command_template:
                continue
            cwd = str(definition.get("cwd", "skill")).strip().lower() or "skill"
            if cwd not in {"workspace", "skill"}:
                self._append_unique(manifest.validation_warnings, f"tool '{tool_name}' has unsupported cwd '{cwd}'; defaulting to skill")
                cwd = "skill"
            timeout_raw = definition.get("timeout_s", 30)
            try:
                timeout_s = int(timeout_raw)
            except (TypeError, ValueError):
                self._append_unique(manifest.validation_warnings, f"tool '{tool_name}' has invalid timeout {timeout_raw!r}; defaulting to 30")
                timeout_s = 30
            if timeout_s <= 0:
                self._append_unique(manifest.validation_warnings, f"tool '{tool_name}' has non-positive timeout {timeout_s}; defaulting to 30")
                timeout_s = 30
            if self._register_tool(
                tool_name,
                manifest,
                spec,
                soft=True,
                cwd=cwd,
                command_template=command_template,
                timeout_s=timeout_s,
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

    def skill_source_label(self, skill: SkillManifest) -> str:
        path = skill.path or skill.doc_path
        if not path:
            return ""
        try:
            resolved = path.resolve()
            if self._is_relative_to(resolved, self.skills_dir.resolve()):
                return f"skills/{resolved.relative_to(self.skills_dir.resolve())}"
            for root in self.extra_skill_dirs:
                if self._is_relative_to(resolved, root.resolve()):
                    return f"configured/{resolved.relative_to(root.resolve())}"
            if self.bundled_skills_dir is not None and self._is_relative_to(resolved, self.bundled_skills_dir.resolve()):
                return f"bundled-skills/{resolved.relative_to(self.bundled_skills_dir.resolve())}"
            if self._is_relative_to(resolved, self.workspace.workspace_root.resolve()):
                return f"workspace/{resolved.relative_to(self.workspace.workspace_root.resolve())}"
            if self._is_relative_to(resolved, self.workspace.home_root.resolve()):
                return f"home/{resolved.relative_to(self.workspace.home_root.resolve())}"
            return str(resolved.relative_to(self.skills_dir.parent))
        except Exception:
            return str(path)

    def skill_provenance_label(self, skill: SkillManifest) -> str:
        path = skill.path or skill.doc_path
        if not path:
            return "unknown"
        try:
            resolved = path.resolve()
            if self._is_relative_to(resolved, self.skills_dir.resolve()):
                return "user/skills"
            for root in self.extra_skill_dirs:
                if self._is_relative_to(resolved, root.resolve()):
                    return "configured"
            if self.bundled_skills_dir is not None and self._is_relative_to(resolved, self.bundled_skills_dir.resolve()):
                return "bundled"
            if self._is_relative_to(resolved, self.workspace.workspace_root.resolve()):
                return "workspace"
            if self._is_relative_to(resolved, self.workspace.home_root.resolve()):
                return "home"
            return "external"
        except Exception:
            return "external"

    @staticmethod
    def skill_status_label(skill: SkillManifest) -> tuple[str, str]:
        if not skill.available:
            return "blocked", "yellow"
        if skill.enabled:
            return "on", "green"
        return "off", "red"

    def enabled_skills(self) -> list[SkillManifest]:
        if self._enabled_skills_cache is None:
            self._enabled_skills_cache = tuple(
                skill for skill in self.skills.values() if skill.enabled and skill.available and skill.execution_allowed
            )
        return list(self._enabled_skills_cache)

    def skills_by_ids(self, skill_ids: list[str]) -> list[SkillManifest]:
        out: list[SkillManifest] = []
        seen = set()
        for skill_id in skill_ids:
            key = str(skill_id).strip()
            if not key or key in seen:
                continue
            skill = self.skills.get(key)
            if not skill or not skill.enabled or not skill.available or not skill.execution_allowed:
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

    def compose_skill_index(self) -> str:
        skills = sorted(self.enabled_skills(), key=lambda item: item.id)
        if not skills:
            return ""

        lines = [f"  - {skill.id}: {skill.description}" for skill in skills]
        return (
            "## Skills (mandatory)\n"
            "Before replying, scan the skills below. If one clearly matches the task, "
            "load it with skill_view(name) and follow its instructions.\n"
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
            entrypoints = ", ".join(entry.name for entry in self._skill_entrypoints(skill)[:3])
            entrypoint_text = f" entrypoints: {entrypoints}." if entrypoints else ""
            produces = ", ".join(skill.produces[:3])
            produce_text = f" produces: {produces}." if produces else ""
            tier = self.skill_provenance_label(skill)
            location = self.skill_source_label(skill)
            location_text = f" location: {location}." if location else ""
            lines.append(
                f"- {skill.id}: {skill.description}. source: {tier}.{tag_text}{produce_text}{tool_text}{entrypoint_text}{location_text}"
            )
        return "\n".join(lines)

    def skill_health_report(self) -> list[dict[str, Any]]:
        return [
            {
                "id": skill.id,
                "name": skill.name,
                "enabled": bool(skill.enabled),
                "available": bool(skill.available),
                "provenance": self.skill_provenance_label(skill),
                "execution_allowed": bool(skill.execution_allowed),
                "adapter": skill.adapter,
                "status": self.skill_status_label(skill)[0],
                "source": self.skill_source_label(skill),
                "availability_code": skill.availability_code or "ready",
                "availability_reason": skill.availability_reason or "ready",
                "tools": self._reported_skill_tools(skill),
                "scripts": self._reported_skill_scripts(skill),
                "entrypoints": [entry.name for entry in self._reported_skill_entrypoints(skill)],
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
        prefix_hit = self._skill_prefix_index.get(lowered)
        if prefix_hit:
            resolved = self.get_skill(prefix_hit)
            if resolved is not None:
                return resolved
        normalized = re.sub(r"[^a-z0-9]+", "", lowered)
        if normalized:
            if normalized in self._skill_fuzzy_collisions:
                return None
            fuzzy_hit = self._skill_fuzzy_index.get(normalized)
            if fuzzy_hit:
                resolved = self.get_skill(fuzzy_hit)
                if resolved is not None:
                    return resolved
        return None

    @staticmethod
    def _is_skill_script_candidate(relpath: str) -> bool:
        return SkillScriptInspector.is_skill_script_candidate(relpath)

    def _skill_runnable_scripts(self, skill: SkillManifest) -> list[str]:
        return self._script_inspector.skill_runnable_scripts(skill)

    def _script_block_reason(self, skill: SkillManifest, rel_script: str) -> str:
        return self._script_inspector.script_block_reason(skill, rel_script)

    def _blocked_skill_scripts(self, skill: SkillManifest) -> list[dict[str, str]]:
        return self._script_inspector.blocked_skill_scripts(skill)

    def _script_is_cli_entry(self, skill: SkillManifest, rel_script: str) -> bool:
        return self._script_inspector.script_is_cli_entry(skill, rel_script)

    def _script_has_local_module(self, skill: SkillManifest, script_path: Path, module_name: str) -> bool:
        return self._script_inspector.script_has_local_module(skill, script_path, module_name)

    def _python_script_missing_modules(self, skill: SkillManifest, rel_script: str) -> list[str]:
        return self._script_inspector.python_script_missing_modules(skill, rel_script)

    def _script_interpreter(self, ext: str) -> list[str] | None:
        return self._script_inspector.script_interpreter(ext)

    def _python_module_available(self, module_name: str) -> bool:
        return self._script_inspector.python_module_available(module_name)

    def _skill_entrypoints(self, skill: SkillManifest) -> list[SkillEntrypointDef]:
        return list(getattr(skill, "entrypoints", []) or [])

    def _reported_skill_tools(self, skill: SkillManifest) -> list[str]:
        if not getattr(skill, "execution_allowed", True):
            return []
        return sorted(reg.name for reg in self._tool_registry.values() if reg.skill_id == skill.id)

    def _reported_skill_scripts(self, skill: SkillManifest) -> list[str]:
        if not getattr(skill, "execution_allowed", True):
            return []
        return self._skill_runnable_scripts(skill)

    def _reported_skill_entrypoints(self, skill: SkillManifest) -> list[SkillEntrypointDef]:
        if not getattr(skill, "execution_allowed", True):
            return []
        return self._skill_entrypoints(skill)

    def _exposed_relevant_skill_scripts(self, skill: SkillManifest) -> list[str]:
        if skill.allowed_tools and _RUN_SKILL_TOOL_NAME not in skill.allowed_tools:
            return []
        return self._reported_skill_scripts(skill)

    def _exposed_relevant_skill_entrypoints(self, skill: SkillManifest) -> list[SkillEntrypointDef]:
        if skill.allowed_tools and _RUN_SKILL_TOOL_NAME not in skill.allowed_tools:
            return []
        return self._reported_skill_entrypoints(skill)

    def select_skills(self, ctx: SkillContext, top_n: int = 3) -> list[SkillManifest]:
        return self.selector.select_skills(ctx, top_n=top_n)

    def tool_registration(self, tool_name: str) -> RegisteredTool | None:
        return self._tool_registry.get(str(tool_name).strip())

    def _tool_allowed_for_permission_profile(self, reg: RegisteredTool) -> bool:
        profile = str(getattr(self, "permission_profile", "full") or "full").strip().lower()
        if profile == "full":
            return True

        capability = str(reg.capability or "").strip().lower()
        if reg.name in {_SKILLS_LIST_TOOL_NAME, _SKILL_VIEW_TOOL_NAME, _REQUEST_USER_INPUT_TOOL_NAME}:
            return True
        if reg.name == _RUN_SKILL_TOOL_NAME:
            return False
        if capability == "run_shell_command":
            return False
        if capability.startswith(("web_", "utility_")):
            return False
        if capability.startswith("memory_"):
            return True
        if profile == "workspace":
            return capability.startswith("workspace_") or capability.startswith("skill_")
        # safe profile: read-only workspace, memory helpers, and skill catalog/load utilities.
        return capability in {"workspace_read", "workspace_tree"} or capability.startswith("skill_")

    def tool_is_mutating(self, tool_name: str) -> bool:
        reg = self.tool_registration(tool_name)
        if reg is None:
            return False
        if reg.mutates is not None:
            return bool(reg.mutates)
        capability = str(reg.capability or "").strip().lower()
        if capability.startswith("workspace_") and capability != "workspace_read":
            return True
        if reg.tool_scope == "skill" and (capability.endswith("_runner") or capability == "skill_executor"):
            return True
        return False

    def tool_is_blocked_for_local_workspace(self, tool_name: str) -> bool:
        reg = self.tool_registration(tool_name)
        if reg is None:
            normalized_name = str(tool_name).strip()
            return normalized_name not in (_CORE_TOOL_NAMES | _ALWAYS_AVAILABLE_TOOL_NAMES | {_RUN_SKILL_TOOL_NAME, "shell_command"})
        capability = str(reg.capability or "").strip().lower()
        if capability.startswith(("workspace_", "memory_", "skill_")):
            return False
        if capability in {"run_shell_command", "user_input_requester"}:
            return False
        return True

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
            sections.append(f"### Skill: {skill.name} ({skill.id})\n{body}")

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

    def _rebase_vendor_paths(self, text: str, skill: SkillManifest) -> str:
        out = str(text or "")
        skill_root = str(skill.path or "")
        if not skill_root:
            return out
        replacements: dict[str, str] = {}
        discovered_roots: list[Path] = []
        for root in self.skill_roots:
            try:
                resolved_root = root.resolve()
            except OSError:
                continue
            if resolved_root not in discovered_roots:
                discovered_roots.append(resolved_root)
        for candidate in self.list_skills():
            if not candidate.path:
                continue
            resolved_candidate = candidate.path.resolve()
            for root in discovered_roots:
                if not self._is_relative_to(resolved_candidate, root):
                    continue
                source = root / candidate.id
                replacements[str(source)] = str(candidate.path)
                try:
                    home_relative = source.relative_to(self.workspace.home_root.resolve())
                except ValueError:
                    continue
                replacements[f"~/{home_relative.as_posix()}"] = str(candidate.path)
        for source, target in sorted(replacements.items(), key=lambda item: -len(item[0])):
            out = out.replace(source, target)
        return out

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
        if not self._is_relative_to(target, skill.path.resolve()):
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
        return sorted(name for name in self.allowed_tool_names(selected, ctx=ctx) if name in _CORE_TOOL_NAMES)

    def _optional_tool_names_for_turn(
        self,
        selected: list[SkillManifest],
    ) -> list[str]:
        selected_map = {skill.id: skill for skill in selected}
        allowed: list[str] = []
        for tool_name, reg in self._tool_registry.items():
            if tool_name == _RUN_SKILL_TOOL_NAME:
                if not self._tool_allowed_for_permission_profile(reg):
                    continue
                if any(
                    (self._exposed_relevant_skill_entrypoints(skill) or self._exposed_relevant_skill_scripts(skill))
                    for skill in selected
                    if not skill.disable_model_invocation
                ):
                    allowed.append(tool_name)
                continue
            if reg.skill_id == "__runtime__":
                continue
            if not self._tool_allowed_for_permission_profile(reg):
                continue
            skill = selected_map.get(reg.skill_id)
            if not skill:
                continue
            if skill.disable_model_invocation:
                continue
            if skill.allowed_tools and reg.name not in skill.allowed_tools:
                continue
            allowed.append(tool_name)
        return sorted(allowed)

    def optional_tool_names(self, selected: list[SkillManifest], ctx: SkillContext | None = None) -> list[str]:
        _ = ctx
        if self.is_minimal_profile():
            return []
        return self._optional_tool_names_for_turn(selected)

    def _tool_names_for_turn(
        self,
        selected: list[SkillManifest],
        ctx: SkillContext | None = None,
    ) -> list[str]:
        if self.is_minimal_profile():
            turn_core_names = {
                name
                for name in (set(self.model_exposed_tool_names()) | set(self._optional_tool_names_for_turn(selected)))
                if name in _CORE_TOOL_NAMES
            }
            safe_names = set(turn_core_names)
            safe_names.update({_SKILLS_LIST_TOOL_NAME, _SKILL_VIEW_TOOL_NAME})
            runtime_cfg = self.config.get("runtime", {}) if isinstance(self.config.get("runtime"), dict) else {}
            if bool(runtime_cfg.get("ask_user_tool", True)):
                safe_names.add(_REQUEST_USER_INPUT_TOOL_NAME)
            return sorted(name for name in safe_names if name in self._tool_registry)

        names = set(self.model_exposed_tool_names())
        names.update(self.optional_tool_names(selected, ctx=ctx))
        runtime_cfg = self.config.get("runtime", {}) if isinstance(self.config.get("runtime"), dict) else {}
        if not bool(runtime_cfg.get("ask_user_tool", True)):
            names.discard(_REQUEST_USER_INPUT_TOOL_NAME)
        return sorted(name for name in names if name in self._tool_registry)

    def allowed_tool_names(
        self,
        selected: list[SkillManifest],
        ctx: SkillContext | None = None,
    ) -> list[str]:
        return self._tool_names_for_turn(selected, ctx=ctx)

    def _tool_schemas(
        self,
        names: list[str],
        selected: list[SkillManifest] | None = None,
        ctx: SkillContext | None = None,
    ) -> list[dict[str, Any]]:
        return self._tool_schema_builder.build(names, selected=selected, ctx=ctx)

    def tools_for_turn(
        self,
        selected: list[SkillManifest],
        ctx: SkillContext | None = None,
    ) -> list[dict[str, Any]]:
        names = self.allowed_tool_names(selected, ctx=ctx)
        cache_key = SkillToolSchemaBuilder.cache_key(names, selected, generation=self.generation)
        cached = self._tools_schema_cache.get(cache_key)
        if cached is not None:
            return cached
        tools = self._tool_schemas(names, selected=selected, ctx=ctx)
        self._tools_schema_cache[cache_key] = tools
        return tools

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

    def _validate_schema_value(self, field_name: str, value: Any, schema: dict[str, Any]) -> None:
        enum = schema.get("enum")
        if isinstance(enum, list) and enum and value not in enum:
            raise ValueError(f"Invalid '{field_name}': expected one of {', '.join(map(str, enum[:10]))}")

        raw_type = schema.get("type")
        expected_types: list[str] = []
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

    def _validate_tool_args(self, reg: RegisteredTool, args: dict[str, Any]) -> dict[str, Any]:
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
        selected: list[SkillManifest],
        ctx: SkillContext | None = None,
    ) -> tuple[RegisteredTool, SkillManifest | None]:
        reg = self._tool_registry.get(tool_name)
        if not reg:
            raise LookupError(f"No adapter for tool '{tool_name}'")

        if reg.name in self._tool_names_for_turn(selected, ctx=ctx):
            if reg.name in _ALWAYS_AVAILABLE_TOOL_NAMES:
                return reg, None
            return reg, self.skills.get(reg.skill_id)
        raise PermissionError(f"Tool '{tool_name}' is not enabled by the current skill configuration")

    def _prepare_tool_args(
        self,
        reg: RegisteredTool,
        args: dict[str, Any],
        selected: list[SkillManifest],
    ) -> dict[str, Any]:
        recovered = _recover_tool_args(args)
        validated = self._validate_tool_args(reg, recovered)
        if reg.name == _RUN_SKILL_TOOL_NAME:
            validated = self._run_validator.validate_run_skill_args(validated, selected)
        return validated

    def execute_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        selected: list[SkillManifest],
        ctx: SkillContext,
        confirm_shell: ShellConfirmationFn | None = None,
        request_user_input: UserInputRequestFn | None = None,
    ) -> dict[str, Any]:
        return self._skill_executor.execute_tool_call(
            tool_name,
            args,
            selected,
            ctx,
            confirm_shell=confirm_shell,
            request_user_input=request_user_input,
        )
