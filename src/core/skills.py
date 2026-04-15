from __future__ import annotations

import logging

import ast
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, TypedDict

from core.message_types import ChatMessage, JSONValue, ShellConfirmationFn, ToolCallDelta, UserInputRequestFn
from core.memory import VectorMemory
from core.runtime_config import SkillsRuntimeConfig
from core.skill_parser import SKILL_DOC, SkillEntrypointDef, SkillManifest, extract_skill_doc, parse_agentskill_manifest
from core.workspace import WorkspaceManager

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
        "run_checks",
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
_SCRIPT_INTERPRETER_BY_EXT = {
    ".py": [sys.executable],
    ".sh": ["bash"],
    ".js": ["node"],
    ".mjs": ["node"],
}

def _ok(data: object, duration_ms: int) -> dict[str, object]:
    return {"ok": True, "data": data, "error": None, "meta": {"duration_ms": duration_ms}}


def _err(code: str, message: str, duration_ms: int) -> dict[str, object]:
    return {
        "ok": False,
        "data": None,
        "error": {"code": code, "message": message},
        "meta": {"duration_ms": duration_ms},
    }


class ToolProtocolError(RuntimeError):
    pass


def _recover_tool_args(raw_args: dict[str, object]) -> dict[str, object]:
    if not isinstance(raw_args, dict):
        return {}
    out = dict(raw_args)
    raw = out.get("_raw")
    if not isinstance(raw, str) or not raw.strip():
        return out
    text = raw.strip()

    parsed: Optional[dict[str, object]] = None
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
    memory_hits: list[dict[str, JSONValue]]
    loaded_skill_ids: List[str] = field(default_factory=list)
    recent_routing_hint: str = ""
    sticky_skill_ids: List[str] = field(default_factory=list)
    explicit_skill_id: str = ""
    explicit_skill_args: str = ""


@dataclass(slots=True)
class ToolExecutionEnv:
    workspace: WorkspaceManager
    memory: VectorMemory
    config: dict[str, JSONValue]
    debug: bool
    confirm_shell: Optional[ShellConfirmationFn] = None
    request_user_input: Optional[UserInputRequestFn] = None


@dataclass(slots=True)
class RegisteredTool:
    name: str
    skill_id: str
    tool_scope: str
    capability: str
    description: str
    parameters: dict[str, object]
    module: Optional[object] = None
    module_path: Optional[Path] = None
    module_name: str = ""
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
        self.config = {}
        self.runtime_config = SkillsRuntimeConfig()
        self.debug = debug
        self.skills_cfg = {}
        self.tools_cfg = {}
        self.python_executable = sys.executable
        self.reload_config(config or {})
        self.generation = 0

        self.skill_roots = self._discover_skill_roots()
        self.skills: Dict[str, SkillManifest] = {}
        self._all_skills: List[SkillManifest] = []
        self._skill_index: Dict[str, Dict[str, Any]] = {}
        self._tool_registry: Dict[str, RegisteredTool] = {}
        self._list_skills_cache: Optional[Tuple[SkillManifest, ...]] = None
        self._enabled_skills_cache: Optional[Tuple[SkillManifest, ...]] = None
        self._skill_catalog_cache: Dict[int, str] = {}
        self._skill_cards_cache: Dict[Tuple[Tuple[str, ...], bool], str] = {}
        self._runnable_scripts_cache: Dict[Tuple[str, str], Tuple[str, ...]] = {}
        self._tools_schema_cache: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        self._python_module_probe_cache: Dict[str, bool] = {}
        self._proc_env_base = self._build_proc_env_base()
        self.load_skills()

    def reload_config(self, config: Optional[dict]) -> None:
        self.config = config or {}
        self.skills_cfg = self.config.get("skills", {}) if isinstance(self.config.get("skills"), dict) else {}
        self.tools_cfg = self.config.get("tools", {}) if isinstance(self.config.get("tools"), dict) else {}
        self.runtime_config = SkillsRuntimeConfig.from_config(self.config)
        self.python_executable = self.runtime_config.python_executable

    def _build_proc_env_base(self) -> Dict[str, str]:
        env = os.environ.copy()
        env["ALPHANUS_WORKSPACE_ROOT"] = str(self.workspace.workspace_root)
        env["ALPHANUS_HOME_ROOT"] = str(self.workspace.home_root)
        env["ALPHANUS_MEMORY_PATH"] = str(self.memory.storage_path)
        env["ALPHANUS_MEMORY_MODEL"] = str(self.memory.model_name)
        env["ALPHANUS_MEMORY_EAGER_LOAD"] = "1" if bool(getattr(self.memory, "eager_load_encoder", False)) else "0"
        env["ALPHANUS_SKILL_PYTHON"] = str(self.python_executable)
        env["ALPHANUS_CONFIG_JSON"] = json.dumps(self.config, ensure_ascii=False)

        # Prepend project src and repo root to PYTHONPATH for skill script imports.
        repo_root = self.skills_dir.parent.resolve()
        src_root = (repo_root / "src").resolve()
        path_entries = [str(src_root)] if src_root.exists() else []
        path_entries.append(str(repo_root))
        existing = env.get("PYTHONPATH", "")
        prefix = os.pathsep.join(path_entries)
        env["PYTHONPATH"] = prefix if not existing else prefix + os.pathsep + existing

        npm_path = shutil.which("npm")
        if npm_path:
            try:
                proc = subprocess.run(
                    [npm_path, "root", "-g"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except Exception as exc:
                logging.debug("npm root probe failed: %s", exc)
                proc = None
            node_root = (proc.stdout or "").strip() if proc and proc.returncode == 0 else ""
            if node_root:
                existing_node_path = env.get("NODE_PATH", "")
                env["NODE_PATH"] = node_root if not existing_node_path else node_root + os.pathsep + existing_node_path
        return env

    def _discover_skill_roots(self) -> List[Path]:
        # Skills are loaded from the configured skills root.
        return [self.skills_dir]

    def _discover_skill_dirs(self, root: Path) -> List[Path]:
        if not root.exists():
            return []
        candidates: List[Path] = []
        seen: set[str] = set()
        docs = [root / SKILL_DOC] if (root / SKILL_DOC).exists() else []
        if root.is_dir():
            try:
                docs.extend(sorted(root.rglob(SKILL_DOC)))
            except Exception as exc:
                logging.debug("rglob for SKILL.md failed in %s: %s", root, exc)
                docs = docs or []
        for skill_doc in docs:
            skill_dir = skill_doc.parent.resolve()
            if ".git" in skill_dir.parts:
                continue
            if not self._is_relative_to(skill_dir, root.resolve()):
                continue
            rel = skill_doc.relative_to(root.resolve())
            if len(rel.parts) > 5:
                continue
            key = str(skill_dir)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(skill_dir)
        candidates.sort(key=lambda item: (len(item.relative_to(root.resolve()).parts), str(item)))
        return candidates

    @staticmethod
    def _load_module(path: Path, module_name: str):
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _read_tool_specs(path: Path) -> Optional[Dict[str, Any]]:
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
    def _bundled_files_for_path(root: Path) -> List[str]:
        return sorted(
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and path.name != SKILL_DOC
        )

    def _load_manifest(self, child: Path) -> Optional[SkillManifest]:
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
        return manifest.prompt

    def _remove_skill_tools(self, skill_id: str) -> None:
        for tool_name, reg in list(self._tool_registry.items()):
            if reg.skill_id == skill_id:
                self._tool_registry.pop(tool_name, None)

    def _invalidate_skill_caches(self) -> None:
        self._list_skills_cache = None
        self._enabled_skills_cache = None
        self._skill_catalog_cache = {}
        self._skill_cards_cache = {}
        self._runnable_scripts_cache = {}
        self._tools_schema_cache = {}

    @staticmethod
    def _append_unique(items: List[str], value: str) -> None:
        text = str(value).strip()
        if text and text not in items:
            items.append(text)

    def _rebuild_skill_index(self) -> None:
        self._skill_index = {}
        for skill in self.enabled_skills():
            self._skill_index[skill.id] = {
                "id": skill.id,
                "name": skill.name,
                "description": skill.description,
                "tags": list(skill.tags),
                "categories": list(skill.categories),
                "tools": list(skill.allowed_tools),
                "produces": list(skill.produces),
                "entrypoints": [entry.name for entry in self._skill_entrypoints(skill)],
                "scripts": self._skill_runnable_scripts(skill),
                "execution_allowed": bool(skill.execution_allowed),
                "adapter": skill.adapter,
                "user_invocable": bool(skill.user_invocable),
                "model_invocable": not bool(skill.disable_model_invocation),
            }

    def _register_tool(self, tool_name: str, manifest: SkillManifest, spec: Dict[str, Any], **extra: Any) -> bool:
        if tool_name in self._tool_registry:
            prev = self._tool_registry[tool_name]
            self._append_unique(
                manifest.validation_errors,
                f"duplicate tool '{tool_name}' already registered by {prev.skill_id}",
            )
            return False

        capability = str(spec.get("capability", "")).strip()
        description = str(spec.get("description", "")).strip()
        parameters = spec.get("parameters")
        if not capability or not description or not isinstance(parameters, dict):
            self._append_unique(manifest.validation_errors, f"invalid tool spec '{tool_name}'")
            return False

        self._tool_registry[tool_name] = RegisteredTool(
            name=tool_name,
            skill_id=manifest.id,
            tool_scope=self._tool_scope_for_name(tool_name),
            capability=capability,
            description=description,
            parameters=parameters,
            **extra,
        )
        return True

    @staticmethod
    def _tool_scope_for_name(tool_name: str) -> str:
        return "core" if tool_name in _CORE_TOOL_NAMES else "skill"

    def _register_runtime_tools(self) -> None:
        self._tool_registry[_SKILLS_LIST_TOOL_NAME] = RegisteredTool(
            name=_SKILLS_LIST_TOOL_NAME,
            skill_id="__runtime__",
            tool_scope="core",
            capability="skill_catalog_reader",
            description="List available skills with minimal metadata. Use this to discover a relevant skill before loading it.",
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
        previous_enabled = {skill_id: skill.enabled for skill_id, skill in self.skills.items()}
        self.generation += 1
        self.skill_roots = self._discover_skill_roots()
        self.skills = {}
        self._all_skills = []
        self._skill_index = {}
        self._tool_registry = {}
        self._invalidate_skill_caches()
        self._register_runtime_tools()
        if not any(root.exists() for root in self.skill_roots):
            return

        for root in self.skill_roots:
            if not root.exists():
                continue
            for child in self._discover_skill_dirs(root):
                manifest: Optional[SkillManifest] = None
                try:
                    manifest = self._load_manifest(child)
                    if manifest is None:
                        continue

                    if manifest.id in previous_enabled:
                        manifest.enabled = previous_enabled[manifest.id]

                    (
                        manifest.available,
                        manifest.availability_code,
                        manifest.availability_reason,
                    ) = self._check_skill_availability(manifest)

                    existing = self.skills.get(manifest.id)
                    if existing is not None:
                        source = self.skill_source_label(existing) or existing.id
                        incoming = self.skill_source_label(manifest) or manifest.id
                        self._append_unique(
                            existing.validation_errors,
                            f"duplicate skill id '{manifest.id}' ignored from {incoming}; using {source}",
                        )
                        continue

                    if manifest.available and manifest.execution_allowed and not self._load_skill_tools(manifest):
                        manifest.available = False
                        manifest.execution_allowed = False
                        if not manifest.availability_code or manifest.availability_code == "ready":
                            manifest.availability_code = "invalid"
                        if not manifest.availability_reason:
                            manifest.availability_reason = manifest.validation_errors[0] if manifest.validation_errors else "skill load failed"

                    self.skills[manifest.id] = manifest
                    self._all_skills.append(manifest)
                except Exception as exc:
                    self._remove_skill_tools(manifest.id if manifest else child.name)
                    if manifest is not None:
                        self._append_unique(manifest.validation_errors, str(exc))
                        manifest.available = False
                        manifest.execution_allowed = False
                        manifest.availability_code = "invalid"
                        manifest.availability_reason = str(exc)
                        self.skills[manifest.id] = manifest
                        self._all_skills.append(manifest)
                    elif self.debug:
                        print(f"[skill] failed to load {child.name}: {exc}")
        self._rebuild_skill_index()

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

    def _check_skill_availability(self, manifest: SkillManifest) -> Tuple[bool, str, str]:
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
        local_registered: List[str] = []
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

    def list_skills(self) -> List[SkillManifest]:
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
                return "repo/skills"
            if self._is_relative_to(resolved, self.workspace.workspace_root.resolve()):
                return "workspace"
            if self._is_relative_to(resolved, self.workspace.home_root.resolve()):
                return "home"
            return "external"
        except Exception:
            return "external"

    @staticmethod
    def skill_status_label(skill: SkillManifest) -> Tuple[str, str]:
        if not skill.available:
            return "blocked", "yellow"
        if skill.enabled:
            return "on", "green"
        return "off", "red"

    def enabled_skills(self) -> List[SkillManifest]:
        if self._enabled_skills_cache is None:
            self._enabled_skills_cache = tuple(
                skill
                for skill in self.skills.values()
                if skill.enabled and skill.available and skill.execution_allowed
            )
        return list(self._enabled_skills_cache)

    def skills_by_ids(self, skill_ids: List[str]) -> List[SkillManifest]:
        out: List[SkillManifest] = []
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
            "If none match, proceed normally without loading a skill.\n\n"
            "<available_skills>\n"
            + "\n".join(lines)
            + "\n</available_skills>"
        )

    def model_exposed_tool_names(self) -> List[str]:
        # Skill-native tools are exposed only through turn selection
        # (loaded skills), not globally through enablement alone.
        return sorted(name for name in _ALWAYS_AVAILABLE_TOOL_NAMES if name in self._tool_registry)

    def skill_catalog_text_for(self, skills: List[SkillManifest], max_tags: int = 3) -> str:
        lines: List[str] = []
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

    def skill_health_report(self) -> List[Dict[str, Any]]:
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
        self._rebuild_skill_index()
        return True

    def get_skill(self, skill_id: str) -> Optional[SkillManifest]:
        return self.skills.get(skill_id)

    def resolve_skill_reference(self, skill_ref: str) -> Optional[SkillManifest]:
        ref = str(skill_ref).strip()
        if not ref:
            return None
        exact = self.get_skill(ref)
        if exact is not None:
            return exact
        lowered = ref.lower()
        for skill in self.list_skills():
            if skill.id.lower() == lowered or skill.name.lower() == lowered:
                return skill
        prefix_matches = [
            skill
            for skill in self.list_skills()
            if skill.id.lower().startswith(lowered) or skill.name.lower().startswith(lowered)
        ]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        normalized = re.sub(r"[^a-z0-9]+", "", lowered)
        fuzzy = [
            skill
            for skill in self.list_skills()
            if normalized
            and (
                re.sub(r"[^a-z0-9]+", "", skill.id.lower()) == normalized
                or re.sub(r"[^a-z0-9]+", "", skill.name.lower()) == normalized
            )
        ]
        if len(fuzzy) == 1:
            return fuzzy[0]
        return None

    @staticmethod
    def _is_skill_script_candidate(relpath: str) -> bool:
        normalized = str(relpath or "").strip()
        if not normalized:
            return False
        name = Path(normalized).name
        if name in {"tools.py", "hooks.py", "__init__.py"}:
            return False
        if normalized.startswith("scripts/"):
            return True
        return "/" not in normalized and Path(normalized).suffix.lower() in _SCRIPT_INTERPRETER_BY_EXT

    def _skill_runnable_scripts(self, skill: SkillManifest) -> List[str]:
        cache_key = (skill.id, str(skill.path or ""))
        cached = self._runnable_scripts_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        runnable: List[str] = []
        for rel in skill.bundled_files:
            if not self._is_skill_script_candidate(rel):
                continue
            if not self._script_is_cli_entry(skill, rel):
                continue
            ext = Path(rel).suffix.lower()
            interpreter = self._script_interpreter(ext)
            if not interpreter:
                continue
            if not Path(interpreter[0]).exists() and interpreter[0] != Path(sys.executable).resolve().as_posix() and shutil.which(interpreter[0]) is None:
                continue
            if self._python_script_missing_modules(skill, rel):
                continue
            runnable.append(rel)
        deduped = tuple(sorted(dict.fromkeys(runnable)))
        self._runnable_scripts_cache[cache_key] = deduped
        return list(deduped)

    def _script_block_reason(self, skill: SkillManifest, rel_script: str) -> str:
        if not skill.path:
            return "skill root is unavailable"
        script_path = (skill.path / rel_script).resolve()
        if not self._is_relative_to(script_path, skill.path.resolve()):
            return "script path escapes skill root"
        if not script_path.exists():
            return "script file is missing"
        ext = script_path.suffix.lower()
        interpreter = self._script_interpreter(ext)
        if not interpreter:
            return f"unsupported script type: {script_path.suffix}"
        if not Path(interpreter[0]).exists() and interpreter[0] != Path(sys.executable).resolve().as_posix() and shutil.which(interpreter[0]) is None:
            return f"missing interpreter: {interpreter[0]}"
        missing_modules = self._python_script_missing_modules(skill, rel_script)
        if missing_modules:
            return f"missing python modules: {', '.join(missing_modules)}"
        return ""

    def _blocked_skill_scripts(self, skill: SkillManifest) -> List[Dict[str, str]]:
        blocked: List[Dict[str, str]] = []
        for rel in sorted(rel for rel in skill.bundled_files if self._is_skill_script_candidate(rel)):
            if not self._script_is_cli_entry(skill, rel):
                continue
            if Path(rel).suffix.lower() not in _SCRIPT_INTERPRETER_BY_EXT:
                continue
            reason = self._script_block_reason(skill, rel)
            if reason:
                blocked.append({"script": rel, "reason": reason})
        return blocked

    def _script_is_cli_entry(self, skill: SkillManifest, rel_script: str) -> bool:
        if not skill.path:
            return False
        script_path = (skill.path / rel_script).resolve()
        ext = script_path.suffix.lower()
        if ext in {".sh", ".js", ".mjs"}:
            return True
        if ext != ".py" or not script_path.exists():
            return False
        return script_path.name != "__init__.py"

    def _script_has_local_module(self, skill: SkillManifest, script_path: Path, module_name: str) -> bool:
        if not skill.path:
            return False
        bases = [script_path.parent, skill.path / "scripts", skill.path]
        for base in bases:
            if (base / f"{module_name}.py").exists():
                return True
            module_dir = base / module_name
            if module_dir.exists() and module_dir.is_dir():
                return True
        return False

    def _python_script_missing_modules(self, skill: SkillManifest, rel_script: str) -> List[str]:
        if not skill.path:
            return []
        script_path = (skill.path / rel_script).resolve()
        if script_path.suffix.lower() != ".py" or not script_path.exists():
            return []
        try:
            tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
        except Exception:
            return []

        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = str(alias.name or "").split(".", 1)[0].strip()
                    if name:
                        imported.add(name)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                name = str(node.module).split(".", 1)[0].strip()
                if name:
                    imported.add(name)

        missing: List[str] = []
        stdlib = getattr(sys, "stdlib_module_names", set())
        for name in sorted(imported):
            if name in stdlib:
                continue
            if self._script_has_local_module(skill, script_path, name):
                continue
            if not self._python_module_available(name):
                missing.append(name)
        return missing

    def _script_interpreter(self, ext: str) -> Optional[List[str]]:
        normalized = str(ext or "").lower()
        if normalized == ".py":
            return [self.python_executable]
        base = _SCRIPT_INTERPRETER_BY_EXT.get(normalized)
        return list(base) if base else None

    def _python_module_available(self, module_name: str) -> bool:
        cached = self._python_module_probe_cache.get(module_name)
        if cached is not None:
            return cached

        python_exec = str(self.python_executable or sys.executable)
        if python_exec == sys.executable:
            try:
                available = importlib.util.find_spec(module_name) is not None
            except Exception:
                available = False
            self._python_module_probe_cache[module_name] = available
            return available

        try:
            proc = subprocess.run(
                [python_exec, "-c", "import importlib.util,sys; raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)", module_name],
                capture_output=True,
                text=True,
                timeout=5,
                env=self._proc_env_base,
            )
            available = proc.returncode == 0
        except Exception:
            available = False
        self._python_module_probe_cache[module_name] = available
        return available

    def _skill_entrypoints(self, skill: SkillManifest) -> List[SkillEntrypointDef]:
        return list(getattr(skill, "entrypoints", []) or [])

    def _reported_skill_tools(self, skill: SkillManifest) -> List[str]:
        if not getattr(skill, "execution_allowed", True):
            return []
        return sorted(
            reg.name
            for reg in self._tool_registry.values()
            if reg.skill_id == skill.id
        )

    def _reported_skill_scripts(self, skill: SkillManifest) -> List[str]:
        if not getattr(skill, "execution_allowed", True):
            return []
        return self._skill_runnable_scripts(skill)

    def _reported_skill_entrypoints(self, skill: SkillManifest) -> List[SkillEntrypointDef]:
        if not getattr(skill, "execution_allowed", True):
            return []
        return self._skill_entrypoints(skill)

    def _exposed_relevant_skill_scripts(self, skill: SkillManifest, ctx: Optional[SkillContext]) -> List[str]:
        if skill.allowed_tools and _RUN_SKILL_TOOL_NAME not in skill.allowed_tools:
            return []
        return self._reported_skill_scripts(skill)

    def _exposed_relevant_skill_entrypoints(self, skill: SkillManifest, ctx: Optional[SkillContext]) -> List[SkillEntrypointDef]:
        if skill.allowed_tools and _RUN_SKILL_TOOL_NAME not in skill.allowed_tools:
            return []
        return self._reported_skill_entrypoints(skill)

    def select_skills(self, ctx: SkillContext, top_n: int = 3) -> List[SkillManifest]:
        loaded = [skill for skill in self.skills_by_ids(list(getattr(ctx, "loaded_skill_ids", []) or [])) if not skill.disable_model_invocation]
        if not loaded:
            return []
        limit = max(1, int(top_n))
        if len(loaded) <= 1:
            return loaded[:limit]

        scored = [
            (self._skill_selection_score(skill, ctx), idx, skill)
            for idx, skill in enumerate(loaded)
        ]
        if not any(score > 0 for score, _idx, _skill in scored):
            return loaded[:limit]
        scored.sort(key=lambda item: (-item[0], item[1], item[2].id))
        return [skill for _score, _idx, skill in scored[:limit]]

    def tool_registration(self, tool_name: str) -> Optional[RegisteredTool]:
        return self._tool_registry.get(str(tool_name).strip())

    def tool_is_mutating(self, tool_name: str) -> bool:
        reg = self.tool_registration(tool_name)
        if reg is None:
            return False
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
            return normalized_name not in (
                _CORE_TOOL_NAMES
                | _ALWAYS_AVAILABLE_TOOL_NAMES
                | {_RUN_SKILL_TOOL_NAME, "shell_command"}
            )
        capability = str(reg.capability or "").strip().lower()
        if capability.startswith(("workspace_", "memory_", "skill_")):
            return False
        if capability in {"run_shell_command", "user_input_requester"}:
            return False
        return True

    def skill_cards_text(self, skills: List[SkillManifest]) -> str:
        key = (tuple(getattr(skill, "id", "") for skill in skills), True)
        cached = self._skill_cards_cache.get(key)
        if cached is not None:
            return cached
        lines: List[str] = []
        for skill in skills:
            allowed_tools = list(getattr(skill, "allowed_tools", []) or [])
            bundled_files = list(getattr(skill, "bundled_files", []) or [])
            tool_text = ", ".join(allowed_tools[:4]) or "none"
            resource_count = len([rel for rel in bundled_files if not rel.startswith("scripts/")])
            lines.append(
                f"- {getattr(skill, 'id', '')}: {getattr(skill, 'description', '')} | tools: {tool_text} | resources: {resource_count} | "
                f"execution: {'yes' if getattr(skill, 'execution_allowed', True) else 'no'} | "
                f"adapter: {getattr(skill, 'adapter', 'agentskills')} | user_invocable: {'yes' if getattr(skill, 'user_invocable', True) else 'no'} | "
                f"model_invocable: {'no' if getattr(skill, 'disable_model_invocation', False) else 'yes'}"
            )
        rendered = "\n".join(lines)
        self._skill_cards_cache[key] = rendered
        return rendered

    def compose_skill_block(
        self,
        selected: List[SkillManifest],
        ctx: SkillContext,
        context_limit: int,
        ratio: float = 0.15,
        hard_cap: int = 0,
    ) -> str:
        if hard_cap > 0:
            selected = selected[:hard_cap]
        if not selected:
            return ""

        sections: List[str] = []
        for skill in selected:
            body = self._ensure_skill_prompt(skill).strip()
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
            snippet = self._safe_prompt_snippet(body, allowed)
            out.append(head + "\n" + snippet)
            used = budget

        return "\n\n".join(out)

    @staticmethod
    def _extract_argument_item(argument_text: str, index: int) -> str:
        if index < 0:
            return ""
        try:
            parts = shlex.split(argument_text)
        except Exception:
            parts = argument_text.split()
        return parts[index] if index < len(parts) else ""

    @staticmethod
    def _selection_tokens(*values: Any) -> set[str]:
        tokens: set[str] = set()
        for value in values:
            if isinstance(value, (list, tuple, set)):
                tokens.update(SkillRuntime._selection_tokens(*list(value)))
                continue
            text = str(value or "").strip().lower()
            if not text:
                continue
            for token in re.findall(r"[a-z0-9][a-z0-9_-]{1,}", text):
                tokens.add(token)
                if "_" in token:
                    tokens.update(part for part in token.split("_") if len(part) > 1)
                if "-" in token:
                    tokens.update(part for part in token.split("-") if len(part) > 1)
        return tokens

    def _skill_selection_score(self, skill: SkillManifest, ctx: SkillContext) -> int:
        score = 0
        explicit_skill_id = str(getattr(ctx, "explicit_skill_id", "")).strip().lower()
        if explicit_skill_id and skill.id.lower() == explicit_skill_id:
            score += 1000
        sticky_ids = {str(item).strip().lower() for item in getattr(ctx, "sticky_skill_ids", []) or [] if str(item).strip()}
        if skill.id.lower() in sticky_ids:
            score += 250

        user_tokens = self._selection_tokens(getattr(ctx, "user_input", ""))
        branch_tokens = self._selection_tokens(getattr(ctx, "branch_labels", []) or [])
        attachment_tokens = self._selection_tokens(*(Path(item).name for item in (getattr(ctx, "attachments", []) or [])))
        recent_tokens = self._selection_tokens(getattr(ctx, "recent_routing_hint", ""))
        skill_tokens = self._selection_tokens(
            skill.id,
            skill.name,
            skill.description,
            skill.tags,
            skill.categories,
            skill.produces,
            skill.allowed_tools,
            [entry.name for entry in self._skill_entrypoints(skill)],
        )

        score += 4 * len(user_tokens & skill_tokens)
        score += 2 * len(branch_tokens & skill_tokens)
        score += 2 * len(attachment_tokens & skill_tokens)
        score += 1 * len(recent_tokens & skill_tokens)
        return score

    def _rebase_vendor_paths(self, text: str, skill: SkillManifest) -> str:
        out = str(text or "")
        skill_root = str(skill.path or "")
        if not skill_root:
            return out
        replacements: Dict[str, str] = {}
        discovered_roots: List[Path] = []
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

    def _apply_skill_arguments(self, text: str, argument_text: str) -> str:
        out = str(text or "")
        if not argument_text:
            return out
        out = re.sub(r"\$ARGUMENTS\[(\d+)\]", lambda m: self._extract_argument_item(argument_text, int(m.group(1))), out)
        out = re.sub(r"\$(\d+)\b", lambda m: self._extract_argument_item(argument_text, int(m.group(1))), out)
        out = out.replace("$ARGUMENTS", argument_text)
        if "$ARGUMENTS" not in text and "$0" not in text and "$ARGUMENTS[" not in text:
            out = out.rstrip() + f"\n\nARGUMENTS: {argument_text}"
        return out

    def user_skill_root(self) -> Path:
        return self.skills_dir.resolve()

    def _skill_linked_files(self, skill: SkillManifest) -> Dict[str, List[str]]:
        linked: Dict[str, List[str]] = {}
        for rel in sorted(skill.bundled_files):
            parts = Path(rel).parts
            key = parts[0] if len(parts) > 1 else "root"
            linked.setdefault(key, []).append(rel)
        return linked

    def skills_list(self) -> Dict[str, Any]:
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
        skill = self.get_skill(key)
        if skill is None or not skill.enabled or not skill.available:
            raise FileNotFoundError(f"Skill '{key}' not found")
        return skill

    def skill_view(self, name: str, file_path: str, ctx: SkillContext) -> Dict[str, Any]:
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

        kept: List[str] = []
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

    def core_tool_names(self) -> List[str]:
        return sorted(name for name in self.model_exposed_tool_names() if name in _CORE_TOOL_NAMES)

    def core_tool_names_for_turn(
        self,
        selected: List[SkillManifest],
        ctx: Optional[SkillContext] = None,
    ) -> List[str]:
        return sorted(name for name in self.allowed_tool_names(selected, ctx=ctx) if name in _CORE_TOOL_NAMES)

    def optional_tool_names(self, selected: List[SkillManifest], ctx: Optional[SkillContext] = None) -> List[str]:
        selected_map = {skill.id: skill for skill in selected}
        allowed: List[str] = []
        for tool_name, reg in self._tool_registry.items():
            if tool_name == _RUN_SKILL_TOOL_NAME:
                if any(
                    (self._exposed_relevant_skill_entrypoints(skill, ctx) or self._exposed_relevant_skill_scripts(skill, ctx))
                    for skill in selected
                    if not skill.disable_model_invocation
                ):
                    allowed.append(tool_name)
                continue
            if reg.skill_id == "__runtime__":
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

    def _tool_names_for_turn(
        self,
        selected: List[SkillManifest],
        ctx: Optional[SkillContext] = None,
    ) -> List[str]:
        names = set(self.model_exposed_tool_names())
        names.update(self.optional_tool_names(selected, ctx=ctx))
        runtime_cfg = self.config.get("runtime", {}) if isinstance(self.config.get("runtime"), dict) else {}
        if not bool(runtime_cfg.get("ask_user_tool", True)):
            names.discard(_REQUEST_USER_INPUT_TOOL_NAME)
        return sorted(name for name in names if name in self._tool_registry)

    @staticmethod
    def _active_skill_ids(selected: List[SkillManifest]) -> List[str]:
        return sorted(
            dict.fromkeys(
                str(getattr(skill, "id", "")).strip()
                for skill in selected
                if str(getattr(skill, "id", "")).strip()
            )
        )

    def allowed_tool_names(
        self,
        selected: List[SkillManifest],
        ctx: Optional[SkillContext] = None,
    ) -> List[str]:
        return self._tool_names_for_turn(selected, ctx=ctx)

    def _tool_schema_cache_key(
        self,
        names: List[str],
        selected: List[SkillManifest],
    ) -> Tuple[Any, ...]:
        selected_ids = tuple(
            str(getattr(skill, "id", "")).strip()
            for skill in selected
            if str(getattr(skill, "id", "")).strip()
        )
        active_skill_ids = tuple(self._active_skill_ids(selected))
        return (
            self.generation,
            selected_ids,
            tuple(names),
            active_skill_ids,
        )

    def _dynamic_run_skill_schema(self, selected: List[SkillManifest], ctx: Optional[SkillContext]) -> Dict[str, Any]:
        executable_skills = [
            skill
            for skill in selected
            if not skill.disable_model_invocation
            and (self._exposed_relevant_skill_entrypoints(skill, ctx) or self._exposed_relevant_skill_scripts(skill, ctx))
        ]
        properties: Dict[str, Any] = {
            "skill_id": {"type": "string"},
            "entrypoint": {"type": "string"},
            "script": {"type": "string"},
            "params": {"type": "object"},
            "argv": {"type": "array", "items": {"type": "string"}},
            "stdin": {"type": "string"},
            "timeout_s": {"type": "integer"},
        }
        if len(executable_skills) > 1:
            properties["skill_id"] = {"type": "string", "enum": [skill.id for skill in executable_skills]}

        entrypoint_names = sorted(
            dict.fromkeys(
                entrypoint.name
                for skill in executable_skills
                for entrypoint in self._exposed_relevant_skill_entrypoints(skill, ctx)
            )
        )
        if entrypoint_names:
            properties["entrypoint"] = {"type": "string", "enum": entrypoint_names}

        script_names = sorted(
            dict.fromkeys(
                rel_script
                for skill in executable_skills
                for rel_script in self._exposed_relevant_skill_scripts(skill, ctx)
            )
        )
        if script_names:
            properties["script"] = {"type": "string", "enum": script_names}

        required = ["skill_id"] if len(executable_skills) > 1 else []
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def _tool_schemas(
        self,
        names: List[str],
        selected: Optional[List[SkillManifest]] = None,
        ctx: Optional[SkillContext] = None,
    ) -> List[Dict[str, Any]]:
        tools = []
        for name in names:
            reg = self._tool_registry[name]
            parameters = reg.parameters
            description = reg.description
            if reg.name == _RUN_SKILL_TOOL_NAME and selected is not None:
                parameters = self._dynamic_run_skill_schema(selected, ctx)
                available_paths: List[str] = []
                for skill in selected:
                    if skill.disable_model_invocation:
                        continue
                    for entrypoint in self._exposed_relevant_skill_entrypoints(skill, ctx):
                        available_paths.append(f"{skill.id}:{entrypoint.name}")
                    for rel_script in self._exposed_relevant_skill_scripts(skill, ctx):
                        available_paths.append(f"{skill.id}:{rel_script}")
                if available_paths:
                    description = (
                        f"{reg.description} Available executable paths: {', '.join(sorted(dict.fromkeys(available_paths))[:8])}."
                    )
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": reg.name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )
        return tools

    def tools_for_turn(
        self,
        selected: List[SkillManifest],
        ctx: Optional[SkillContext] = None,
    ) -> List[Dict[str, Any]]:
        names = self.allowed_tool_names(selected, ctx=ctx)
        cache_key = self._tool_schema_cache_key(names, selected)
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
        ctx: Optional[SkillContext] = None,
    ) -> Tuple[RegisteredTool, Optional[SkillManifest]]:
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
        args: Dict[str, Any],
        selected: List[SkillManifest],
        ctx: SkillContext,
    ) -> Dict[str, Any]:
        recovered = _recover_tool_args(args)
        validated = self._validate_tool_args(reg, recovered)
        if reg.name == _RUN_SKILL_TOOL_NAME:
            validated = self._validate_run_skill_args(validated, selected, ctx)
        return validated

    def _validate_run_skill_args(
        self,
        args: Dict[str, Any],
        selected: List[SkillManifest],
        ctx: SkillContext,
    ) -> Dict[str, Any]:
        requested_entrypoint = str(args.get("entrypoint", "")).strip()
        requested_script = str(args.get("script", "")).strip()
        if bool(requested_entrypoint) == bool(requested_script):
            raise ValueError("run_skill requires exactly one of 'entrypoint' or 'script'")
        if requested_entrypoint:
            return self._validate_skill_entrypoint_args(args, selected, ctx)
        return self._validate_skill_script_args(args, selected, ctx)

    def _validate_skill_script_args(
        self,
        args: Dict[str, Any],
        selected: List[SkillManifest],
        ctx: SkillContext,
    ) -> Dict[str, Any]:
        selected_with_scripts = [
            skill
            for skill in selected
            if self._exposed_relevant_skill_scripts(skill, ctx) and not skill.disable_model_invocation and skill.execution_allowed
        ]
        if not selected_with_scripts:
            raise PermissionError("No selected skills expose runnable bundled scripts")

        requested_skill_id = str(args.get("skill_id", "")).strip()
        if requested_skill_id:
            skill = next((item for item in selected_with_scripts if item.id == requested_skill_id), None)
            if skill is None:
                raise PermissionError(f"Skill '{requested_skill_id}' is not selected or has no runnable scripts")
        elif len(selected_with_scripts) == 1:
            skill = selected_with_scripts[0]
        else:
            skill_ids = ", ".join(skill.id for skill in selected_with_scripts[:4])
            raise ValueError(f"Multiple selected skills expose scripts; specify skill_id ({skill_ids})")

        requested_script = str(args.get("script", "")).strip()
        if not requested_script:
            raise ValueError("Missing required argument: script")
        runnable = self._exposed_relevant_skill_scripts(skill, ctx)
        chosen = ""
        if requested_script in runnable:
            chosen = requested_script
        else:
            matches = [rel for rel in runnable if Path(rel).name == requested_script or Path(rel).stem == requested_script]
            if len(matches) == 1:
                chosen = matches[0]
        if not chosen:
            raise PermissionError(f"Script '{requested_script}' is not available for skill '{skill.id}'")

        params = args.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("Invalid 'params': expected object")

        out = dict(args)
        out["skill_id"] = skill.id
        out["script"] = chosen
        out["params"] = params
        timeout_s = out.get("timeout_s")
        if timeout_s is None:
            out["timeout_s"] = 30
        return out

    def _validate_skill_entrypoint_args(
        self,
        args: Dict[str, Any],
        selected: List[SkillManifest],
        ctx: SkillContext,
    ) -> Dict[str, Any]:
        selected_with_entrypoints = [
            skill
            for skill in selected
            if self._exposed_relevant_skill_entrypoints(skill, ctx) and not skill.disable_model_invocation and skill.execution_allowed
        ]
        if not selected_with_entrypoints:
            raise PermissionError("No selected skills expose runnable entrypoints")

        requested_skill_id = str(args.get("skill_id", "")).strip()
        if requested_skill_id:
            skill = next((item for item in selected_with_entrypoints if item.id == requested_skill_id), None)
            if skill is None:
                raise PermissionError(f"Skill '{requested_skill_id}' is not selected or has no runnable entrypoints")
        elif len(selected_with_entrypoints) == 1:
            skill = selected_with_entrypoints[0]
        else:
            skill_ids = ", ".join(skill.id for skill in selected_with_entrypoints[:4])
            raise ValueError(f"Multiple selected skills expose entrypoints; specify skill_id ({skill_ids})")

        requested_entrypoint = str(args.get("entrypoint", "")).strip()
        if not requested_entrypoint:
            raise ValueError("Missing required argument: entrypoint")
        candidates = self._exposed_relevant_skill_entrypoints(skill, ctx)
        entrypoint = next((item for item in candidates if item.name == requested_entrypoint), None)
        if entrypoint is None:
            raise PermissionError(f"Entrypoint '{requested_entrypoint}' is not available for skill '{skill.id}'")

        params = args.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("Invalid 'params': expected object")
        self._validate_schema_value("params", params, entrypoint.parameters)

        out = dict(args)
        out["skill_id"] = skill.id
        out["entrypoint"] = entrypoint.name
        out["params"] = params
        if out.get("timeout_s") is None:
            out["timeout_s"] = entrypoint.timeout_s
        return out

    def _execute_registered_tool(
        self,
        reg: RegisteredTool,
        args: Dict[str, Any],
        env: ToolExecutionEnv,
        ctx: SkillContext,
    ) -> Any:
        if reg.name == _SKILLS_LIST_TOOL_NAME:
            return self.skills_list()
        if reg.name == _SKILL_VIEW_TOOL_NAME:
            return self.skill_view(str(args.get("name", "")).strip(), str(args.get("file_path", "")).strip(), ctx)
        if reg.name == _REQUEST_USER_INPUT_TOOL_NAME:
            if not env.request_user_input:
                raise PermissionError("User input runtime is unavailable")
            return env.request_user_input(args)
        if reg.name == _RUN_SKILL_TOOL_NAME:
            return self._execute_run_skill_tool(args, env)
        if reg.module is None and reg.module_path:
            reg.module = self._load_module(reg.module_path, reg.module_name or f"alphanus_tools_{reg.skill_id}")
        if reg.module is None or not hasattr(reg.module, "execute"):
            raise ToolProtocolError(f"Tool '{reg.name}' has no callable execute() handler")
        return reg.module.execute(reg.name, args, env)

    def execute_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        selected: List[SkillManifest],
        ctx: SkillContext,
        confirm_shell: Optional[ShellConfirmationFn] = None,
        request_user_input: Optional[UserInputRequestFn] = None,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            reg, _owner = self._resolve_tool_call(tool_name, selected, ctx=ctx)
            normalized_args = self._prepare_tool_args(reg, args, selected, ctx)
            env = ToolExecutionEnv(
                workspace=self.workspace,
                memory=self.memory,
                config=self.config,
                debug=self.debug,
                confirm_shell=confirm_shell,
                request_user_input=request_user_input,
            )
            result = self._execute_registered_tool(reg, normalized_args, env, ctx)
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
        except RuntimeError as exc:
            message = str(exc).strip() or "Action failed"
            return _err("E_IO", message, int((time.perf_counter() - start) * 1000))
        except Exception as exc:
            message = str(exc) if self.debug else "Action failed"
            return _err("E_IO", message, int((time.perf_counter() - start) * 1000))

    def _execute_run_skill_tool(
        self,
        args: Dict[str, Any],
        env: ToolExecutionEnv,
    ) -> Dict[str, Any]:
        if str(args.get("entrypoint", "")).strip():
            return self._execute_skill_entrypoint_tool(args, env)
        if str(args.get("script", "")).strip():
            return self._execute_skill_script_tool(args, env)
        raise ValueError("run_skill requires an entrypoint or script")

    def _execute_skill_script_tool(
        self,
        args: Dict[str, Any],
        env: ToolExecutionEnv,
    ) -> Dict[str, Any]:
        skill = self.get_skill(str(args.get("skill_id", "")).strip())
        if skill is None or not skill.path:
            raise FileNotFoundError("Selected skill root is unavailable")

        rel_script = str(args.get("script", "")).strip()
        script_path = (skill.path / rel_script).resolve()
        if not self._is_relative_to(script_path, skill.path.resolve()):
            raise PermissionError("Skill script path escapes skill root")
        if not script_path.exists():
            raise FileNotFoundError(f"Skill script not found: {rel_script}")
        ext = script_path.suffix.lower()
        interpreter = self._script_interpreter(ext)
        if not interpreter:
            raise PermissionError(f"Unsupported skill script type: {script_path.suffix}")
        if not Path(interpreter[0]).exists() and interpreter[0] != Path(sys.executable).resolve().as_posix() and shutil.which(interpreter[0]) is None:
            raise FileNotFoundError(f"Missing interpreter for skill script: {interpreter[0]}")

        argv = args.get("argv") if isinstance(args.get("argv"), list) else []
        proc_env = dict(self._proc_env_base)
        proc_env["ALPHANUS_SELECTED_SKILL_ID"] = skill.id
        proc_env["ALPHANUS_SKILL_ROOT"] = str(skill.path)
        proc_env["ALPHANUS_SKILL_SCRIPT"] = rel_script
        params_payload = args.get("params")
        if not isinstance(params_payload, dict):
            params_payload = {}
        proc_env["ALPHANUS_TOOL_ARGS_JSON"] = json.dumps(params_payload, ensure_ascii=False)
        proc = subprocess.run(
            list(interpreter) + [str(script_path)] + [str(item) for item in argv],
            cwd=str(skill.path),
            capture_output=True,
            text=True,
            input=str(args.get("stdin") or ""),
            timeout=max(1, int(args.get("timeout_s", 30))),
            env=proc_env,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            msg = stderr or stdout or f"Skill script failed with exit code {proc.returncode}"
            lowered = msg.lower()
            if "permissionerror" in lowered or "operation not permitted" in lowered:
                raise PermissionError(msg)
            if "filenotfounderror" in lowered or "no such file or directory" in lowered:
                raise FileNotFoundError(msg)
            raise RuntimeError(msg)

        out = (proc.stdout or "").strip()
        if not out:
            return {
                "skill_id": skill.id,
                "script": rel_script,
                "stdout": "",
            }
        candidate = out.splitlines()[-1].strip()
        try:
            parsed = json.loads(candidate)
        except Exception:
            return {
                "skill_id": skill.id,
                "script": rel_script,
                "stdout": out,
                "stderr": (proc.stderr or "").strip(),
            }
        if isinstance(parsed, dict):
            parsed.setdefault("skill_id", skill.id)
            parsed.setdefault("script", rel_script)
            return parsed
        return {"skill_id": skill.id, "script": rel_script, "value": parsed}

    @staticmethod
    def _resolve_entrypoint_placeholders(template: str, values: Dict[str, Any]) -> str:
        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in values:
                raise ValueError(f"Missing template value: {key}")
            return shlex.quote(str(values[key]))

        return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, template)

    def _run_shell_workflow_command(
        self,
        command: str,
        env: ToolExecutionEnv,
        timeout_s: int,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        caps = env.config.get("capabilities", {})
        dangerously_skip_permissions = bool(caps.get("dangerously_skip_permissions", False))
        shell_require_confirmation = bool(caps.get("shell_require_confirmation", True))
        if shell_require_confirmation and not dangerously_skip_permissions:
            if not env.confirm_shell:
                raise PermissionError("Shell confirmation callback is required")
            if not env.confirm_shell(command):
                raise PermissionError("Shell command rejected by user")
        allowed_cwd_roots = [cwd] if cwd else None
        result = env.workspace.run_shell_command(
            command,
            timeout_s=max(1, int(timeout_s)),
            cwd=cwd,
            allowed_cwd_roots=allowed_cwd_roots,
        )
        if not result.get("ok"):
            error = result.get("error") or {}
            message = str(error.get("message", "Shell workflow command failed"))
            code = str(error.get("code", ""))
            if code == "E_POLICY":
                raise PermissionError(message)
            if code == "E_TIMEOUT":
                raise TimeoutError(message)
            raise RuntimeError(message)
        return result["data"]

    def _execute_skill_entrypoint_tool(
        self,
        args: Dict[str, Any],
        env: ToolExecutionEnv,
    ) -> Dict[str, Any]:
        skill = self.get_skill(str(args.get("skill_id", "")).strip())
        if skill is None or not skill.path:
            raise FileNotFoundError("Selected skill root is unavailable")
        entrypoint_name = str(args.get("entrypoint", "")).strip()
        entrypoint = next((item for item in self._skill_entrypoints(skill) if item.name == entrypoint_name), None)
        if entrypoint is None:
            raise FileNotFoundError(f"Skill entrypoint not found: {entrypoint_name}")

        params = args.get("params") if isinstance(args.get("params"), dict) else {}
        template_values: Dict[str, Any] = {
            "workspace_root": str(self.workspace.workspace_root),
            "skill_root": str(skill.path),
        }
        template_values.update(params)
        timeout_s = max(1, int(args.get("timeout_s", entrypoint.timeout_s)))

        install_results: List[Dict[str, Any]] = []
        verify_results: List[Dict[str, Any]] = []
        command_cwd = str(skill.path) if entrypoint.cwd == "skill" else str(self.workspace.workspace_root)
        for template in entrypoint.install:
            command = self._resolve_entrypoint_placeholders(template, template_values)
            install_results.append(self._run_shell_workflow_command(command, env, timeout_s, cwd=command_cwd))
        for template in entrypoint.verify:
            command = self._resolve_entrypoint_placeholders(template, template_values)
            verify_results.append(self._run_shell_workflow_command(command, env, timeout_s, cwd=command_cwd))
        command = self._resolve_entrypoint_placeholders(entrypoint.command, template_values)
        run_data = self._run_shell_workflow_command(command, env, timeout_s, cwd=command_cwd)
        return {
            "skill_id": skill.id,
            "entrypoint": entrypoint.name,
            "command": command,
            "install_results": install_results,
            "verify_results": verify_results,
            "stdout": run_data.get("stdout", ""),
            "stderr": run_data.get("stderr", ""),
            "returncode": run_data.get("returncode", 0),
            "cwd": run_data.get("cwd", ""),
        }

    @staticmethod
    def _normalize_result(result: Any, duration_ms: int) -> Dict[str, Any]:
        if isinstance(result, dict) and {"ok", "data", "error"}.issubset(result.keys()):
            out = dict(result)
            meta = out.get("meta") if isinstance(out.get("meta"), dict) else {}
            meta["duration_ms"] = int(meta.get("duration_ms", duration_ms))
            out["meta"] = meta
            return out
        return _ok(result, duration_ms)
