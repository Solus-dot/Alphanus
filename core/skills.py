from __future__ import annotations

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
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.memory import VectorMemory
from core.skill_parser import SKILL_DOC, SkillEntrypointDef, SkillManifest, extract_skill_doc, parse_agentskill_manifest
from core.workspace import WorkspaceManager

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#._/-]{1,}")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "app",
    "application",
    "artifact",
    "artifacts",
    "build",
    "create",
    "for",
    "help",
    "i",
    "in",
    "interface",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "page",
    "please",
    "production",
    "project",
    "quality",
    "the",
    "this",
    "to",
    "use",
    "using",
    "with",
    "write",
}

_TIME_SENSITIVE_TOKENS = {
    "recent",
    "latest",
    "current",
    "today",
    "now",
    "breaking",
    "update",
    "updates",
    "news",
}

_PERSONAL_MEMORY_TERMS = {
    "address",
    "age",
    "birthday",
    "birthdate",
    "birthyear",
    "city",
    "editor",
    "favourite",
    "favorite",
    "home",
    "ide",
    "job",
    "like",
    "likes",
    "live",
    "location",
    "name",
    "occupation",
    "personal",
    "preference",
    "preferences",
    "prefer",
    "profile",
    "role",
    "work",
}

_SHELL_TERMS = {
    "bash",
    "cli",
    "cmd",
    "command",
    "commands",
    "console",
    "powershell",
    "shell",
    "terminal",
    "zsh",
}

_SHELL_WORKFLOW_HINTS = {
    "apt-get",
    "brew",
    "cargo",
    "go",
    "node",
    "npm",
    "pdftoppm",
    "pip",
    "pnpm",
    "poetry",
    "python",
    "python3",
    "soffice",
    "uv",
    "yarn",
}

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
        "create_files",
        "edit_file",
        "delete_path",
        "run_checks",
    }
)

_CORE_EXPOSURE_POLICIES = {
    "coding_core": _CORE_TOOL_NAMES,
}

_TEXT_LIKE_EXTENSIONS = frozenset(
    {
        "",
        ".c",
        ".cc",
        ".cfg",
        ".conf",
        ".cpp",
        ".css",
        ".csv",
        ".env",
        ".gitignore",
        ".go",
        ".graphql",
        ".h",
        ".hpp",
        ".html",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".kt",
        ".log",
        ".lua",
        ".md",
        ".mjs",
        ".pbtxt",
        ".php",
        ".py",
        ".rb",
        ".rs",
        ".scss",
        ".sh",
        ".sql",
        ".svg",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)

_GENERIC_SCRIPT_TOOL_NAME = "run_skill_script"
_GENERIC_ENTRYPOINT_TOOL_NAME = "run_skill_entrypoint"
_SCRIPT_INTERPRETER_BY_EXT = {
    ".py": [sys.executable],
    ".sh": ["bash"],
    ".js": ["node"],
    ".mjs": ["node"],
}
_ARTIFACT_SYNONYMS = {
    ".docx": ("docx", "word document", "word doc", "microsoft word"),
    ".pdf": ("pdf", "portable document format"),
    ".png": ("png", "image", "screenshot"),
    ".jpg": ("jpg", "jpeg", "image", "photo"),
    ".jpeg": ("jpeg", "jpg", "image", "photo"),
    ".csv": ("csv", "spreadsheet", "comma separated"),
    ".xlsx": ("xlsx", "excel", "spreadsheet"),
}
_CREATE_INTENT_TERMS = frozenset(
    {
        "build",
        "create",
        "generate",
        "make",
        "materialize",
        "new",
        "produce",
        "save",
        "scaffold",
        "write",
    }
)
_EDIT_INTENT_TERMS = frozenset({"edit", "format", "modify", "patch", "revise", "update"})
_REVIEW_INTENT_TERMS = frozenset({"inspect", "preview", "read", "render", "review", "view", "visual"})
_SETUP_INTENT_TERMS = frozenset({"bootstrap", "check", "dependency", "install", "setup", "verify"})
_SCRIPT_CREATE_HINTS = frozenset({"build", "convert", "create", "export", "generate", "make", "write"})
_SCRIPT_EDIT_HINTS = frozenset({"edit", "fix", "format", "modify", "patch", "rewrite", "update"})
_SCRIPT_REVIEW_HINTS = frozenset({"extract", "inspect", "preview", "read", "render", "review", "view"})
_SCRIPT_SETUP_HINTS = frozenset({"bootstrap", "dependency", "install", "setup", "verify"})


def _ok(data: Any, duration_ms: int) -> Dict[str, Any]:
    return {"ok": True, "data": data, "error": None, "meta": {"duration_ms": duration_ms}}


def _err(code: str, message: str, duration_ms: int) -> Dict[str, Any]:
    return {
        "ok": False,
        "data": None,
        "error": {"code": code, "message": message},
        "meta": {"duration_ms": duration_ms},
    }


class ToolProtocolError(RuntimeError):
    pass


def _recover_tool_args(raw_args: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw_args, dict):
        return {}
    out = dict(raw_args)
    raw = out.get("_raw")
    if not isinstance(raw, str) or not raw.strip():
        return out
    text = raw.strip()

    parsed: Optional[Dict[str, Any]] = None
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
    memory_hits: List[Dict[str, Any]]
    recent_routing_hint: str = ""
    sticky_skill_ids: List[str] = field(default_factory=list)


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
    tool_scope: str
    capability: str
    description: str
    parameters: Dict[str, Any]
    module: Optional[object] = None
    module_path: Optional[Path] = None
    module_name: str = ""
    command: str = ""
    timeout_s: int = 30
    confirm_arg: str = ""
    cwd: str = ""


@dataclass(slots=True)
class SkillCandidate:
    skill: SkillManifest
    recall_score: int
    rerank_score: int = 0


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
        self.skill_roots = [self.skills_dir]
        self.workspace = workspace
        self.memory = memory
        self.config = config or {}
        self.debug = debug
        self.skills_cfg = self.config.get("skills", {})
        self.tools_cfg = self.config.get("tools", {})
        self.selection_mode = str(self.skills_cfg.get("selection_mode", "all_enabled"))
        self.max_active_skills = int(self.skills_cfg.get("max_active_skills", 6))
        raw_core_policy = self.tools_cfg.get(
            "core_exposure_policy",
            self.skills_cfg.get("core_exposure_policy", "coding_core"),
        )
        self.core_exposure_policy = str(raw_core_policy or "coding_core").strip().lower()
        self.generation = 0

        self.skills: Dict[str, SkillManifest] = {}
        self._tool_registry: Dict[str, RegisteredTool] = {}
        self._list_skills_cache: Optional[Tuple[SkillManifest, ...]] = None
        self._enabled_skills_cache: Optional[Tuple[SkillManifest, ...]] = None
        self._skill_catalog_cache: Dict[int, str] = {}
        self._proc_env_base = self._build_proc_env_base()
        self.load_skills()

    def _build_proc_env_base(self) -> Dict[str, str]:
        env = os.environ.copy()
        env["ALPHANUS_WORKSPACE_ROOT"] = str(self.workspace.workspace_root)
        env["ALPHANUS_HOME_ROOT"] = str(self.workspace.home_root)
        env["ALPHANUS_MEMORY_PATH"] = str(self.memory.storage_path)
        env["ALPHANUS_MEMORY_MODEL"] = str(self.memory.model_name)
        env["ALPHANUS_MEMORY_BACKEND"] = str(self.memory.embedding_backend)
        env["ALPHANUS_MEMORY_EAGER_LOAD"] = "1" if bool(getattr(self.memory, "eager_load_encoder", False)) else "0"
        env["ALPHANUS_CONFIG_JSON"] = json.dumps(self.config, ensure_ascii=False)

        # Let skill scripts import project modules without per-script sys.path hacks.
        repo_root = str(self.skills_dir.parent)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = repo_root if not existing else repo_root + os.pathsep + existing
        return env

    @staticmethod
    def _prepare_command(command: str) -> Tuple[object, bool]:
        # Use shell=False for simple commands (safer + lower process overhead).
        if re.search(r"[|&;<>`$()]", command):
            return command, True
        try:
            parts = shlex.split(command)
        except ValueError:
            return command, True
        if not parts:
            return command, True
        return parts, False

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
        except Exception:
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
                    except Exception:
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

    def _ensure_skill_hooks(self, manifest: SkillManifest) -> Optional[object]:
        if manifest.hooks is not None:
            return manifest.hooks
        hooks_path = manifest.hooks_path
        if not hooks_path or not hooks_path.exists():
            return None
        manifest.hooks = self._load_module(
            hooks_path,
            f"alphanus_hooks_{manifest.id.replace('-', '_')}",
        )
        return manifest.hooks

    def _remove_skill_tools(self, skill_id: str) -> None:
        for tool_name, reg in list(self._tool_registry.items()):
            if reg.skill_id == skill_id:
                self._tool_registry.pop(tool_name, None)

    def _invalidate_skill_caches(self) -> None:
        self._list_skills_cache = None
        self._enabled_skills_cache = None
        self._skill_catalog_cache = {}

    def _register_tool(self, tool_name: str, manifest: SkillManifest, spec: Dict[str, Any], **extra: Any) -> bool:
        if tool_name in self._tool_registry:
            if self.debug:
                prev = self._tool_registry[tool_name]
                print(f"[skill] duplicate tool '{tool_name}' in {manifest.id}; already registered by {prev.skill_id}")
            return False

        capability = str(spec.get("capability", "")).strip()
        description = str(spec.get("description", "")).strip()
        parameters = spec.get("parameters")
        if not capability or not description or not isinstance(parameters, dict):
            if self.debug:
                print(f"[skill] invalid tool spec '{tool_name}' in {manifest.id}")
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

    def _core_policy_names(self) -> frozenset[str]:
        policy = self.core_exposure_policy or "coding_core"
        names = _CORE_EXPOSURE_POLICIES.get(policy)
        if names is None:
            return _CORE_EXPOSURE_POLICIES["coding_core"]
        return names

    def _register_runtime_tools(self) -> None:
        self._tool_registry[_GENERIC_ENTRYPOINT_TOOL_NAME] = RegisteredTool(
            name=_GENERIC_ENTRYPOINT_TOOL_NAME,
            skill_id="__runtime__",
            tool_scope="skill",
            capability="skill_entrypoint_runner",
            description="Run a declared execution entrypoint from a selected skill.",
            parameters={
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string"},
                    "entrypoint": {"type": "string"},
                    "params": {"type": "object"},
                    "timeout_s": {"type": "integer"},
                },
                "required": ["entrypoint"],
            },
        )
        self._tool_registry[_GENERIC_SCRIPT_TOOL_NAME] = RegisteredTool(
            name=_GENERIC_SCRIPT_TOOL_NAME,
            skill_id="__runtime__",
            tool_scope="skill",
            capability="skill_script_runner",
            description="Run a bundled script from a selected skill.",
            parameters={
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string"},
                    "script": {"type": "string"},
                    "argv": {"type": "array", "items": {"type": "string"}},
                    "stdin": {"type": "string"},
                    "args": {"type": "object"},
                    "timeout_s": {"type": "integer"},
                },
                "required": ["script"],
            },
        )

    def load_skills(self) -> None:
        previous_enabled = {skill_id: skill.enabled for skill_id, skill in self.skills.items()}
        self.generation += 1
        self.skills = {}
        self._tool_registry = {}
        self._invalidate_skill_caches()
        self._register_runtime_tools()
        if not any(root.exists() for root in self.skill_roots):
            return

        for root in self.skill_roots:
            if not root.exists():
                continue
            for child in sorted(root.iterdir(), key=lambda path: path.name):
                if not child.is_dir():
                    continue
                manifest: Optional[SkillManifest] = None
                try:
                    manifest = self._load_manifest(child)
                    if manifest is None:
                        continue

                    if manifest.id in self.skills:
                        raise ValueError(f"Duplicate skill id '{manifest.id}'")

                    if manifest.id in previous_enabled:
                        manifest.enabled = previous_enabled[manifest.id]

                    manifest.source_tier = self._skill_source_tier(manifest)
                    (
                        manifest.available,
                        manifest.availability_code,
                        manifest.availability_reason,
                    ) = self._check_skill_availability(manifest)
                    if not manifest.available:
                        self.skills[manifest.id] = manifest
                        continue

                    if not self._load_skill_tools(manifest):
                        continue

                    self.skills[manifest.id] = manifest
                except Exception as exc:
                    self._remove_skill_tools(manifest.id if manifest else child.name)
                    if self.debug:
                        print(f"[skill] failed to load {child.name}: {exc}")

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

    def _skill_source_tier(self, manifest: SkillManifest) -> str:
        path = manifest.path or manifest.doc_path
        if not path:
            return "external/local"
        try:
            if self._is_relative_to(path.resolve(), self.skills_dir.resolve()):
                return "bundled"
        except Exception:
            pass
        return "external/local"

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

        for spec in manifest.command_tools:
            if allowed_tools and spec.name not in allowed_tools:
                continue
            if self._register_tool(
                spec.name,
                manifest,
                {
                    "capability": spec.capability,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
                command=spec.command,
                timeout_s=spec.timeout_s,
                confirm_arg=spec.confirm_arg,
                cwd=str(manifest.path),
            ):
                local_registered.append(spec.name)

        tools_path = manifest.path / "tools.py"
        if tools_path.exists():
            specs = self._read_tool_specs(tools_path)
            if not isinstance(specs, dict):
                if self.debug:
                    print(f"[skill] {manifest.id} tools.py missing TOOL_SPECS dict or execute()")
                return False

            module_name = f"alphanus_tools_{manifest.id.replace('-', '_')}"

            for tool_name, spec in specs.items():
                if allowed_tools and tool_name not in allowed_tools:
                    continue
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
                if self.debug:
                    print(f"[skill] {manifest.id} missing required tools: {', '.join(missing)}")
                return False

        return True

    def list_skills(self) -> List[SkillManifest]:
        if self._list_skills_cache is None:
            self._list_skills_cache = tuple(sorted(self.skills.values(), key=lambda s: s.id))
        return list(self._list_skills_cache)

    def skill_source_label(self, skill: SkillManifest) -> str:
        path = skill.path or skill.doc_path
        if not path:
            return ""
        try:
            return str(path.resolve().relative_to(self.skills_dir.parent))
        except Exception:
            return str(path)

    @staticmethod
    def skill_provenance_label(skill: SkillManifest) -> str:
        return str(getattr(skill, "source_tier", "") or "external/local")

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
                skill for skill in self.list_skills() if skill.enabled and skill.available
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
                "enabled": bool(skill.enabled),
                "available": bool(skill.available),
                "status": self.skill_status_label(skill)[0],
                "source_tier": self.skill_provenance_label(skill),
                "source": self.skill_source_label(skill),
                "availability_code": skill.availability_code or "ready",
                "availability_reason": skill.availability_reason or "ready",
                "tools": list(skill.allowed_tools),
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
        return True

    def get_skill(self, skill_id: str) -> Optional[SkillManifest]:
        return self.skills.get(skill_id)

    @staticmethod
    def _match_tokens(text: str) -> List[str]:
        tokens: List[str] = []
        for token in _TOKEN_RE.findall(text.lower()):
            normalized = token.strip("._/-")
            if len(normalized) < 3 or normalized in _STOPWORDS:
                continue
            tokens.append(normalized)
        return tokens

    @staticmethod
    def _looks_like_memory_request(text: str, query_tokens: set[str]) -> bool:
        lowered = text.lower()
        if any(term in query_tokens for term in {"memory", "recall", "remember"}):
            return True
        if re.search(r"\b(?:my|user(?:'s)?)\s+[a-z][a-z0-9 _-]{0,40}\s+is\b", lowered):
            return True
        if re.search(r"\bi\s+(?:am|'m)\b", lowered):
            return True
        if not re.search(r"\bmy\b", lowered):
            return False
        if query_tokens & _PERSONAL_MEMORY_TERMS:
            return True
        return bool(
            re.search(r"\b(?:what(?:'s| is)|tell me|do you know|do you remember|recall)\s+my\b", lowered)
        )

    @staticmethod
    def _looks_like_shell_request(text: str, query_tokens: set[str]) -> bool:
        lowered = text.lower()
        if query_tokens & _SHELL_TERMS:
            return True
        if "`" in text:
            return True
        if re.search(r"\b(?:run|execute)\b", lowered):
            return True
        if query_tokens & {"latest", "recent", "current", "today", "news"}:
            return False
        if "version" not in query_tokens and "installed" not in query_tokens:
            return False
        if re.search(r"\bdo i have\b", lowered):
            return True
        if "installed" in query_tokens:
            return True
        if re.search(r"\bon my (?:machine|system|computer|mac|pc)\b", lowered):
            return True
        return bool(re.search(r"\bmy\b", lowered) and re.search(r"\b(?:check|show|tell|what(?:'s| is)?)\b", lowered))

    @staticmethod
    def _extract_inline_extensions(text: str) -> List[str]:
        lowered = text.lower()
        found = set()
        for match in re.finditer(r"(?<!\w)\.[a-z0-9]{2,8}\b", lowered):
            token = match.group(0).lower()
            window = lowered[max(0, match.start() - 48) : min(len(lowered), match.end() + 48)]
            if any(phrase in window for phrase in ("do not create", "don't create", "fake ", "surrogate", "instead of")):
                continue
            found.add(token)
        return sorted(found)

    @staticmethod
    def _extract_filename_extensions(text: str) -> List[str]:
        found = set()
        for match in re.finditer(r"\b[^\s/\\]+\.([a-z0-9]{2,8})\b", text.lower()):
            suffix = Path(match.group(0)).suffix.lower()
            if suffix:
                found.add(suffix)
        return sorted(found)

    def requested_artifact_extensions(self, ctx: SkillContext) -> List[str]:
        found = set(self._extract_inline_extensions(ctx.user_input))
        found.update(self._extract_filename_extensions(ctx.user_input))
        for attachment in ctx.attachments:
            suffix = Path(str(attachment).strip()).suffix.lower()
            if suffix:
                found.add(suffix)
        lowered = ctx.user_input.lower()
        for ext, phrases in _ARTIFACT_SYNONYMS.items():
            if ext in found:
                continue
            if any(phrase in lowered for phrase in phrases):
                found.add(ext)
        return sorted(found)

    def task_intents(self, ctx: SkillContext) -> set[str]:
        text = ctx.user_input.lower()
        tokens = set(self._match_tokens(ctx.user_input))
        intents: set[str] = set()
        if tokens & _CREATE_INTENT_TERMS or re.search(r"\b(?:create|generate|make|save|write|build|produce)\b", text):
            intents.add("create")
        if tokens & _EDIT_INTENT_TERMS or re.search(r"\b(?:edit|modify|update|format|revise)\b", text):
            intents.add("edit")
        if tokens & _REVIEW_INTENT_TERMS or re.search(r"\b(?:review|render|preview|inspect|read|view)\b", text):
            intents.add("review")
        if tokens & _SETUP_INTENT_TERMS or re.search(r"\b(?:install|setup|dependency|verify|check)\b", text):
            intents.add("setup")
        if not intents:
            intents.add("general")
        return intents

    @staticmethod
    def _normalize_extension_token(token: str) -> str:
        raw = str(token).strip().lower()
        if not raw:
            return ""
        if raw.startswith("."):
            return raw
        if re.fullmatch(r"[a-z0-9]{2,8}", raw):
            return f".{raw}"
        return raw

    @staticmethod
    def _script_intents(relpath: str) -> set[str]:
        stem = Path(relpath).stem.lower()
        tokens = set(re.split(r"[^a-z0-9]+", stem))
        intents: set[str] = set()
        if tokens & _SCRIPT_CREATE_HINTS:
            intents.add("create")
        if tokens & _SCRIPT_EDIT_HINTS:
            intents.add("edit")
        if tokens & _SCRIPT_REVIEW_HINTS:
            intents.add("review")
        if tokens & _SCRIPT_SETUP_HINTS:
            intents.add("setup")
        if not intents:
            intents.add("general")
        return intents

    def _skill_runnable_scripts(self, skill: SkillManifest) -> List[str]:
        runnable: List[str] = []
        for rel in skill.bundled_files:
            if not rel.startswith("scripts/"):
                continue
            ext = Path(rel).suffix.lower()
            interpreter = _SCRIPT_INTERPRETER_BY_EXT.get(ext)
            if not interpreter:
                continue
            if interpreter[0] != sys.executable and shutil.which(interpreter[0]) is None:
                continue
            runnable.append(rel)
        return sorted(dict.fromkeys(runnable))

    @staticmethod
    def _entrypoint_intents(entrypoint: SkillEntrypointDef) -> set[str]:
        intents = {str(item).strip().lower() for item in entrypoint.intents if str(item).strip()}
        return intents or {"general"}

    def _skill_entrypoints(self, skill: SkillManifest) -> List[SkillEntrypointDef]:
        return list(getattr(skill, "entrypoints", []) or [])

    @staticmethod
    def _skill_allows_generic_script_runner(skill: SkillManifest) -> bool:
        return not skill.allowed_tools or _GENERIC_SCRIPT_TOOL_NAME in skill.allowed_tools

    def _exposed_relevant_skill_scripts(self, skill: SkillManifest, ctx: Optional[SkillContext]) -> List[str]:
        if not self._skill_allows_generic_script_runner(skill):
            return []
        return self._relevant_skill_scripts(skill, ctx)

    def _entrypoint_supports_artifact(
        self,
        entrypoint: SkillEntrypointDef,
        skill: SkillManifest,
        extensions: List[str],
        intents: Optional[set[str]] = None,
    ) -> bool:
        if not extensions:
            return False
        intents = intents or {"general"}
        entry_intents = self._entrypoint_intents(entrypoint)
        if "create" in intents and not (entry_intents & {"create", "edit", "general"}):
            return False
        if "edit" in intents and not (entry_intents & {"edit", "create", "general"}):
            return False
        if "review" in intents and not (entry_intents & {"review", "edit", "create", "general"}):
            return False
        normalized_exts = {self._normalize_extension_token(item) for item in extensions if item}
        produced = {self._normalize_extension_token(item) for item in entrypoint.produces if item}
        if produced & normalized_exts:
            return True
        for ext in normalized_exts:
            naked = ext.lstrip(".")
            if naked and naked in entrypoint.command.lower():
                return True
            if naked and naked in entrypoint.description.lower():
                return True
        return False

    def _relevant_skill_entrypoints(self, skill: SkillManifest, ctx: Optional[SkillContext]) -> List[SkillEntrypointDef]:
        entrypoints = self._skill_entrypoints(skill)
        if ctx is None:
            return entrypoints
        intents = self.task_intents(ctx)
        requested_exts = self.requested_artifact_extensions(ctx)
        relevant: List[SkillEntrypointDef] = []
        for entrypoint in entrypoints:
            entry_intents = self._entrypoint_intents(entrypoint)
            if "create" in intents and not (entry_intents & {"create", "edit", "general"}):
                continue
            if "edit" in intents and not (entry_intents & {"edit", "create", "general"}):
                continue
            if "review" in intents and not (entry_intents & {"review", "edit", "create", "general"}):
                continue
            if "setup" in intents and not (entry_intents & {"setup", "create", "edit", "general"}):
                continue
            if requested_exts and not self._entrypoint_supports_artifact(entrypoint, skill, requested_exts, intents=intents):
                continue
            relevant.append(entrypoint)
        return relevant

    def _skill_shell_workflow_commands(self, skill: SkillManifest, ctx: Optional[SkillContext]) -> List[str]:
        prompt = self._ensure_skill_prompt(skill)
        if not prompt.strip():
            return []
        commands: List[str] = []
        blocks = re.findall(r"```(?:[^\n`]*)\n(.*?)```", prompt, flags=re.DOTALL)
        for block in blocks:
            for raw_line in block.splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(("-", "*")):
                    line = line[1:].strip()
                if not line:
                    continue
                lowered = line.lower()
                head = lowered.split()[0] if lowered.split() else ""
                if head in _SHELL_WORKFLOW_HINTS or any(token in lowered for token in _SHELL_WORKFLOW_HINTS):
                    commands.append(line)
        if ctx is None:
            return sorted(dict.fromkeys(commands))

        intents = self.task_intents(ctx)
        requested_exts = self.requested_artifact_extensions(ctx)
        if requested_exts and not self._skill_supports_artifact(skill, requested_exts, intents=intents):
            return []
        return sorted(dict.fromkeys(commands))

    def _skill_supports_shell_workflow(self, skill: SkillManifest, ctx: Optional[SkillContext]) -> bool:
        if ctx is None:
            return False
        if self._relevant_skill_entrypoints(skill, ctx):
            return False
        intents = self.task_intents(ctx)
        if not (intents & {"create", "edit", "review", "setup"}):
            return False
        requested_exts = self.requested_artifact_extensions(ctx)
        if requested_exts and not self._skill_supports_artifact(skill, requested_exts, intents=intents):
            return False
        return bool(self._skill_shell_workflow_commands(skill, ctx))

    def _relevant_skill_scripts(self, skill: SkillManifest, ctx: Optional[SkillContext]) -> List[str]:
        runnable = self._skill_runnable_scripts(skill)
        if ctx is None:
            return runnable

        intents = self.task_intents(ctx)
        requested_exts = self.requested_artifact_extensions(ctx)
        relevant: List[str] = []
        for rel in runnable:
            rel_lower = rel.lower()
            script_intents = self._script_intents(rel)
            if "create" in intents:
                if not (script_intents & {"create", "edit"}):
                    continue
            elif "edit" in intents:
                if not (script_intents & {"edit", "create"}):
                    continue
            elif "review" in intents:
                if not (script_intents & {"review", "edit", "create"}):
                    continue
            elif "setup" in intents:
                if not (script_intents & {"setup"}):
                    continue

            if requested_exts:
                tokens = {ext.lstrip(".") for ext in requested_exts}
                if tokens and not any(token in rel_lower for token in tokens):
                    if not self._skill_supports_artifact(skill, requested_exts, intents=intents):
                        continue
            relevant.append(rel)
        return sorted(dict.fromkeys(relevant))

    def _skill_supports_artifact(
        self,
        skill: SkillManifest,
        extensions: List[str],
        intents: Optional[set[str]] = None,
    ) -> bool:
        if not extensions:
            return False
        intents = intents or {"general"}
        normalized_exts = {self._normalize_extension_token(item) for item in extensions if item}
        for entrypoint in self._skill_entrypoints(skill):
            if self._entrypoint_supports_artifact(entrypoint, skill, extensions, intents=intents):
                return True
        produce_tokens = {self._normalize_extension_token(item) for item in skill.produces if item}
        if produce_tokens & normalized_exts:
            return True
        for ext in normalized_exts:
            if not ext:
                continue
            naked = ext.lstrip(".")
            if ext in skill.description.lower() or naked in skill.description.lower():
                return True
            if ext in skill.name.lower() or naked in skill.name.lower():
                return True
            if any(ext == self._normalize_extension_token(tag) or naked in tag.lower() for tag in skill.tags):
                return True
            for spec in skill.command_tools:
                if self._tool_supports_extension(
                    RegisteredTool(
                        name=spec.name,
                        skill_id=skill.id,
                        tool_scope="skill",
                        capability=spec.capability,
                        description=spec.description,
                        parameters=spec.parameters,
                    ),
                    skill,
                    naked,
                ):
                    return True
            for rel in self._skill_runnable_scripts(skill):
                if naked not in rel.lower():
                    continue
                rel_intents = self._script_intents(rel)
                if "create" in intents and not (rel_intents & {"create", "edit"}):
                    continue
                if "edit" in intents and not (rel_intents & {"edit", "create"}):
                    continue
                if "review" in intents and not (rel_intents & {"review", "edit", "create"}):
                    continue
                return True
        return False

    def _activation_bonus(self, skill: SkillManifest, ctx: SkillContext, requested_exts: List[str]) -> int:
        bonus = 0
        intents = self.task_intents(ctx)
        if skill.id in ctx.sticky_skill_ids:
            bonus += 12
        if ctx.recent_routing_hint:
            hint_tokens = set(self._match_tokens(ctx.recent_routing_hint))
            skill_tokens = set(
                self._match_tokens(
                    " ".join([skill.name, skill.description] + list(skill.tags) + list(skill.categories) + list(skill.produces))
                )
            )
            overlap = hint_tokens & skill_tokens
            if overlap:
                bonus += min(12, len(overlap) * 4)
        if requested_exts and self._skill_supports_artifact(skill, requested_exts, intents=intents):
            bonus += 24
        if self._relevant_skill_entrypoints(skill, ctx):
            bonus += 10
        attachment_exts = {Path(str(item).strip()).suffix.lower() for item in ctx.attachments if str(item).strip()}
        if attachment_exts and any(ext in attachment_exts for ext in skill.triggers.get("file_ext", [])):
            bonus += 8
        prompt = self._ensure_skill_prompt(skill)
        if requested_exts:
            for ext in requested_exts:
                naked = ext.lstrip(".")
                if ext in prompt.lower() or naked in prompt.lower():
                    bonus += 6
                    break
        if self._exposed_relevant_skill_scripts(skill, ctx):
            bonus += 6
        return bonus

    def score_skills(self, ctx: SkillContext) -> List[Tuple[int, SkillManifest]]:
        text = ctx.user_input.lower()
        attachments = " ".join(ctx.attachments).lower()
        query_tokens = set(self._match_tokens(ctx.user_input))
        time_sensitive = bool(query_tokens & _TIME_SENSITIVE_TOKENS)
        scored: List[Tuple[int, SkillManifest]] = []

        for skill in self.skills.values():
            if not skill.enabled or not skill.available:
                continue
            score = 0

            for kw in skill.triggers.get("keywords", []):
                kw_lower = kw.lower()
                if kw_lower not in text:
                    continue
                if skill.id == "memory-rag" and kw_lower == "my":
                    if self._looks_like_memory_request(text, query_tokens):
                        score += 30
                    continue
                score += 30

            for ext in skill.triggers.get("file_ext", []):
                ext_lower = ext.lower()
                if ext_lower in text or ext_lower in attachments:
                    score += 20

            metadata_text = " ".join(
                [skill.name, skill.description] + list(skill.tags) + list(skill.categories) + list(skill.produces)
            )
            metadata_tokens = set(self._match_tokens(metadata_text))
            overlap = query_tokens & metadata_tokens
            if overlap:
                score += min(24, len(overlap) * 6)
                if any(token in skill.name.lower() for token in overlap):
                    score += 4

            if time_sensitive:
                temporal_text = " ".join([skill.description] + list(skill.tags))
                temporal_tokens = set(self._match_tokens(temporal_text))
                if temporal_tokens & {"latest", "recent", "current", "news", "internet", "web", "online", "lookup"}:
                    score += 18

            if "memory" in skill.id and ctx.memory_hits:
                score += 10
            if skill.id == "memory-rag" and self._looks_like_memory_request(text, query_tokens):
                score += 16
            if skill.id == "shell-ops" and self._looks_like_shell_request(text, query_tokens):
                score += 28
            requested_exts = self.requested_artifact_extensions(ctx)
            if requested_exts and self._skill_supports_artifact(skill, requested_exts, intents=self.task_intents(ctx)):
                score += 18

            scored.append((score, skill))

        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return scored

    def propose_skill_candidates(self, ctx: SkillContext, top_n: int = 8) -> List[SkillCandidate]:
        ranked = [(score, skill) for score, skill in self.score_skills(ctx) if score > 0]
        if not ranked:
            return []
        limit = max(1, top_n)
        return [SkillCandidate(skill=skill, recall_score=score) for score, skill in ranked[:limit]]

    def rerank_skill_candidates(self, ctx: SkillContext, candidates: List[SkillCandidate], limit: int) -> List[SkillManifest]:
        if not candidates:
            return []
        requested_exts = self.requested_artifact_extensions(ctx)
        activated_limit = max(limit * 2, min(len(candidates), 6))
        for idx, candidate in enumerate(candidates):
            candidate.rerank_score = candidate.recall_score
            if idx < activated_limit:
                candidate.rerank_score += self._activation_bonus(candidate.skill, ctx, requested_exts)
        candidates.sort(key=lambda item: (-item.rerank_score, -item.recall_score, item.skill.id))
        return [item.skill for item in candidates[: max(1, limit)]]

    def expand_selected_skills(self, ctx: SkillContext, selected: List[SkillManifest]) -> List[SkillManifest]:
        expanded = list(selected)
        seen = {skill.id for skill in expanded}
        if any(
            skill.id != "shell-ops" and self._skill_supports_shell_workflow(skill, ctx)
            for skill in expanded
        ):
            shell_skill = self.skills.get("shell-ops")
            if shell_skill and shell_skill.enabled and shell_skill.available and shell_skill.id not in seen:
                expanded.append(shell_skill)
        return expanded

    def select_skills(self, ctx: SkillContext, top_n: int = 3) -> List[SkillManifest]:
        if self.selection_mode == "all_enabled":
            enabled = [s for s in self.skills.values() if s.enabled]
            enabled.sort(key=lambda s: s.id)
            if self.max_active_skills <= 0:
                return self.expand_selected_skills(ctx, enabled)
            return self.expand_selected_skills(ctx, enabled[: self.max_active_skills])

        limit = self.max_active_skills if self.max_active_skills > 0 else top_n
        candidates = self.propose_skill_candidates(ctx, top_n=max(limit * 3, 6))
        selected = self.rerank_skill_candidates(ctx, candidates, max(1, limit))
        return self.expand_selected_skills(ctx, selected)

    def _skill_runtime_note(self, skill: SkillManifest, ctx: SkillContext) -> str:
        requested_exts = self.requested_artifact_extensions(ctx)
        intents = self.task_intents(ctx)
        relevant_entrypoints = self._relevant_skill_entrypoints(skill, ctx)
        relevant_scripts = self._exposed_relevant_skill_scripts(skill, ctx)
        shell_commands = self._skill_shell_workflow_commands(skill, ctx)
        if not requested_exts:
            return ""
        if not self._skill_supports_artifact(skill, requested_exts, intents=intents) and not relevant_entrypoints and not relevant_scripts and not shell_commands:
            return ""
        if relevant_entrypoints:
            entry = relevant_entrypoints[0]
            install_text = ", ".join(entry.install[:3]) if entry.install else "none"
            verify_text = ", ".join(entry.verify[:3]) if entry.verify else "none"
            return (
                "Runtime note:\n"
                f"- This skill exposes structured execution entrypoints for {', '.join(requested_exts[:3])}.\n"
                f"- Preferred entrypoint: {entry.name}\n"
                f"- Install commands: {install_text}\n"
                f"- Verify commands: {verify_text}\n"
                "- Prefer run_skill_entrypoint over ad-hoc shell planning.\n"
                "- Do not invent commands or helper scripts outside the declared entrypoint contract."
            )
        if shell_commands and "create" in intents:
            return (
                "Runtime note:\n"
                f"- This skill exposes a shell/python workflow for {', '.join(requested_exts[:3])} via shell_command.\n"
                f"- Use documented commands only: {'; '.join(shell_commands[:4])}\n"
                "- Do not invent script names.\n"
                "- Prefer shell_command for dependency install and artifact creation if the packaged workflow requires it.\n"
                "- Do not use python -c import probes or run_checks before trying the documented install/create workflow."
                "\n- Each shell_command must be a single plain command with no shell control operators or fallback chaining."
            )
        ext_text = ", ".join(requested_exts[:3])
        if relevant_scripts:
            return (
                "Runtime note:\n"
                f"- Relevant bundled scripts for this request: {', '.join(relevant_scripts[:4])}\n"
                f"- Treat those as the only bundled executable script paths for {ext_text}."
            )
        if "create" in intents:
            available = self._skill_runnable_scripts(skill)
            available_text = ", ".join(available[:4]) if available else "none"
            return (
                "Runtime note:\n"
                f"- This skill has no bundled create-capable executable path for {ext_text} in this runtime.\n"
                f"- Available runnable scripts: {available_text}\n"
                "- Do not invent script names.\n"
                "- Do not probe dependencies with shell or verification tools for this local workspace file task.\n"
                "- If a real opaque artifact is required, say directly that no executable creation path is available."
            )
        return ""

    def compose_skill_block(
        self,
        selected: List[SkillManifest],
        ctx: SkillContext,
        context_limit: int,
        ratio: float = 0.15,
        hard_cap: int = 2,
    ) -> str:
        selected = selected[:hard_cap]
        if not selected:
            return ""

        sections: List[str] = []
        for skill in selected:
            extra = ""
            hooks = self._ensure_skill_hooks(skill)
            if hooks and hasattr(hooks, "pre_prompt"):
                try:
                    hook_out = hooks.pre_prompt(ctx)  # type: ignore[attr-defined]
                    if hook_out:
                        extra = str(hook_out).strip()
                except Exception:
                    pass
            body = self._ensure_skill_prompt(skill).strip()
            if extra:
                body += "\n\n" + extra
            runtime_note = self._skill_runtime_note(skill, ctx)
            if runtime_note:
                body = runtime_note + "\n\n" + body
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
        policy_names = self._core_policy_names()
        if not policy_names:
            return []
        return sorted(
            reg.name
            for reg in self._tool_registry.values()
            if reg.tool_scope == "core" and reg.name in policy_names
        )

    def selected_shell_workflow_skills(self, selected: List[SkillManifest], ctx: Optional[SkillContext]) -> List[str]:
        if ctx is None:
            return []
        return [
            skill.id
            for skill in selected
            if skill.id != "shell-ops" and self._skill_supports_shell_workflow(skill, ctx)
        ]

    def optional_tool_names(self, selected: List[SkillManifest], ctx: Optional[SkillContext] = None) -> List[str]:
        selected_map = {skill.id: skill for skill in selected}
        allowed: List[str] = []
        for tool_name, reg in self._tool_registry.items():
            if tool_name == _GENERIC_ENTRYPOINT_TOOL_NAME:
                if any(self._relevant_skill_entrypoints(skill, ctx) for skill in selected if not skill.disable_model_invocation):
                    allowed.append(tool_name)
                continue
            if tool_name == _GENERIC_SCRIPT_TOOL_NAME:
                if any(
                    self._exposed_relevant_skill_scripts(skill, ctx)
                    and not self._relevant_skill_entrypoints(skill, ctx)
                    for skill in selected
                    if not skill.disable_model_invocation
                ):
                    allowed.append(tool_name)
                continue
            if reg.tool_scope == "core":
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

    def allowed_tool_names(self, selected: List[SkillManifest], ctx: Optional[SkillContext] = None) -> List[str]:
        names = self.core_tool_names()
        names.extend(self.optional_tool_names(selected, ctx=ctx))
        if self.selected_shell_workflow_skills(selected, ctx):
            names = [name for name in names if name != "run_checks"]
        return sorted(dict.fromkeys(names))

    def selected_skill_capabilities(self, selected: List[SkillManifest]) -> Dict[str, Any]:
        executable_skill_ids: List[str] = []
        advisory_skill_ids: List[str] = []
        scripts: List[str] = []
        resources: List[str] = []
        custom_tool_names: List[str] = []

        for skill in selected:
            capability_names: List[str] = []
            bundled_files = list(skill.bundled_files)
            scripts.extend([path for path in bundled_files if path.startswith("scripts/")])
            resources.extend(
                [
                    path
                    for path in bundled_files
                    if path.startswith("references/") or path.startswith("assets/")
                ]
            )
            capability_names.extend(spec.name for spec in skill.command_tools)
            if self._skill_entrypoints(skill):
                capability_names.append(_GENERIC_ENTRYPOINT_TOOL_NAME)
            if skill.path and (skill.path / "tools.py").exists():
                capability_names.append("tools.py")

            registered_custom = [
                reg.name
                for reg in self._tool_registry.values()
                if reg.skill_id == skill.id and reg.tool_scope != "core"
            ]
            if self._skill_entrypoints(skill):
                registered_custom.append(_GENERIC_ENTRYPOINT_TOOL_NAME)
            if self._skill_runnable_scripts(skill) and self._skill_allows_generic_script_runner(skill):
                registered_custom.append(_GENERIC_SCRIPT_TOOL_NAME)
            capability_names.extend(registered_custom)
            custom_tool_names.extend(registered_custom)

            if capability_names:
                executable_skill_ids.append(skill.id)
            else:
                advisory_skill_ids.append(skill.id)

        return {
            "executable_skill_ids": sorted(dict.fromkeys(executable_skill_ids)),
            "advisory_skill_ids": sorted(dict.fromkeys(advisory_skill_ids)),
            "custom_tool_names": sorted(dict.fromkeys(custom_tool_names)),
            "scripts": sorted(dict.fromkeys(scripts)),
            "resources": sorted(dict.fromkeys(resources)),
            "has_executable_skills": bool(executable_skill_ids),
        }

    @staticmethod
    def _opaque_extension_token(path: str) -> str:
        return Path(str(path).strip()).suffix.lower().lstrip(".")

    @staticmethod
    def _tool_supports_extension(reg: RegisteredTool, skill: SkillManifest, extension: str) -> bool:
        if not extension:
            return False
        haystacks = [
            reg.name,
            reg.capability,
            reg.description,
            skill.id,
            skill.name,
            skill.description,
            " ".join(getattr(skill, "produces", []) or []),
        ]
        lowered = extension.lower()
        return any(lowered in str(value).lower() for value in haystacks if str(value).strip())

    def _selected_relevant_materializers(self, selected: List[SkillManifest], opaque_paths: List[str]) -> List[str]:
        relevant: List[str] = []
        extensions = sorted(
            {
                token
                for token in (self._opaque_extension_token(path) for path in opaque_paths)
                if token
            }
        )
        if not extensions:
            return relevant

        for skill in selected:
            for reg in self._tool_registry.values():
                if reg.skill_id != skill.id or reg.tool_scope == "core":
                    continue
                if any(self._tool_supports_extension(reg, skill, extension) for extension in extensions):
                    relevant.append(f"{skill.id}:{reg.name}")
        return sorted(dict.fromkeys(relevant))

    def selected_artifact_materializers(self, selected: List[SkillManifest], ctx: SkillContext) -> List[str]:
        requested_exts = self.requested_artifact_extensions(ctx)
        intents = self.task_intents(ctx)
        if not requested_exts:
            return []
        relevant: List[str] = []
        for skill in selected:
            if self._skill_supports_artifact(skill, requested_exts, intents=intents):
                for entrypoint in self._relevant_skill_entrypoints(skill, ctx):
                    relevant.append(f"{skill.id}:{entrypoint.name}")
                for rel in self._exposed_relevant_skill_scripts(skill, ctx):
                    relevant.append(f"{skill.id}:{rel}")
                for spec in skill.command_tools:
                    if any(
                        self._tool_supports_extension(
                            RegisteredTool(
                                name=spec.name,
                                skill_id=skill.id,
                                tool_scope="skill",
                                capability=spec.capability,
                                description=spec.description,
                                parameters=spec.parameters,
                            ),
                            skill,
                            ext.lstrip("."),
                        )
                        for ext in requested_exts
                    ):
                        relevant.append(f"{skill.id}:{spec.name}")
                if "shell-ops" in {item.id for item in selected} and self._skill_supports_shell_workflow(skill, ctx):
                    relevant.append(f"{skill.id}:shell_command")
        return sorted(dict.fromkeys(relevant))

    def _dynamic_run_skill_entrypoint_schema(self, selected: List[SkillManifest], ctx: Optional[SkillContext]) -> Dict[str, Any]:
        selected_with_entrypoints = [
            skill for skill in selected if self._relevant_skill_entrypoints(skill, ctx) and not skill.disable_model_invocation
        ]
        properties: Dict[str, Any] = {
            "entrypoint": {"type": "string"},
            "params": {"type": "object"},
            "timeout_s": {"type": "integer"},
        }
        if len(selected_with_entrypoints) > 1:
            properties["skill_id"] = {
                "type": "string",
                "enum": [skill.id for skill in selected_with_entrypoints],
            }
        else:
            properties["skill_id"] = {"type": "string"}

        entrypoint_names: List[str] = []
        for skill in selected_with_entrypoints:
            entrypoint_names.extend(entry.name for entry in self._relevant_skill_entrypoints(skill, ctx))
        entrypoint_names = sorted(dict.fromkeys(entrypoint_names))
        if entrypoint_names:
            properties["entrypoint"] = {"type": "string", "enum": entrypoint_names}

        required = ["entrypoint"]
        if len(selected_with_entrypoints) > 1:
            required.append("skill_id")
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _dynamic_run_skill_script_schema(self, selected: List[SkillManifest], ctx: Optional[SkillContext]) -> Dict[str, Any]:
        selected_with_scripts = [
            skill for skill in selected if self._exposed_relevant_skill_scripts(skill, ctx) and not skill.disable_model_invocation
        ]
        properties: Dict[str, Any] = {
            "script": {"type": "string"},
            "argv": {"type": "array", "items": {"type": "string"}},
            "stdin": {"type": "string"},
            "args": {"type": "object"},
            "timeout_s": {"type": "integer"},
        }
        if len(selected_with_scripts) > 1:
            properties["skill_id"] = {
                "type": "string",
                "enum": [skill.id for skill in selected_with_scripts],
            }
        else:
            properties["skill_id"] = {"type": "string"}

        script_names: List[str] = []
        for skill in selected_with_scripts:
            script_names.extend(self._exposed_relevant_skill_scripts(skill, ctx))
        script_names = sorted(dict.fromkeys(script_names))
        if script_names:
            properties["script"] = {
                "type": "string",
                "enum": script_names,
            }

        required = ["script"]
        if len(selected_with_scripts) > 1:
            required.append("skill_id")
        return {
            "type": "object",
            "properties": properties,
            "required": required,
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
            if reg.name == _GENERIC_ENTRYPOINT_TOOL_NAME and selected is not None:
                parameters = self._dynamic_run_skill_entrypoint_schema(selected, ctx)
                available_entrypoints: List[str] = []
                for skill in selected:
                    if skill.disable_model_invocation:
                        continue
                    for entrypoint in self._relevant_skill_entrypoints(skill, ctx):
                        available_entrypoints.append(f"{skill.id}:{entrypoint.name}")
                if available_entrypoints:
                    description = (
                        f"{reg.description} Available entrypoints: {', '.join(sorted(dict.fromkeys(available_entrypoints))[:8])}."
                    )
            if reg.name == _GENERIC_SCRIPT_TOOL_NAME and selected is not None:
                parameters = self._dynamic_run_skill_script_schema(selected, ctx)
                available_scripts: List[str] = []
                for skill in selected:
                    if skill.disable_model_invocation:
                        continue
                    for rel in self._exposed_relevant_skill_scripts(skill, ctx):
                        available_scripts.append(f"{skill.id}:{rel}")
                if available_scripts:
                    description = (
                        f"{reg.description} Available scripts: {', '.join(sorted(dict.fromkeys(available_scripts))[:8])}."
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

    def tools_for_turn(self, selected: List[SkillManifest], ctx: Optional[SkillContext] = None) -> List[Dict[str, Any]]:
        return self._tool_schemas(self.allowed_tool_names(selected, ctx=ctx), selected=selected, ctx=ctx)

    def _run_pre_action_hooks(
        self,
        selected: List[SkillManifest],
        ctx: SkillContext,
        action_name: str,
        args: Dict[str, Any],
    ) -> Tuple[bool, str]:
        for skill in selected:
            hooks = self._ensure_skill_hooks(skill)
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
            hooks = self._ensure_skill_hooks(skill)
            if hooks and hasattr(hooks, "post_response"):
                try:
                    hooks.post_response(ctx, text)  # type: ignore[attr-defined]
                except Exception:
                    continue

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
    ) -> Tuple[RegisteredTool, Optional[SkillManifest]]:
        reg = self._tool_registry.get(tool_name)
        if not reg:
            raise LookupError(f"No adapter for tool '{tool_name}'")

        if reg.name == _GENERIC_ENTRYPOINT_TOOL_NAME:
            if not any(self._skill_entrypoints(skill) for skill in selected if not skill.disable_model_invocation):
                raise PermissionError(f"Tool '{tool_name}' not allowed by active skills")
            return reg, None

        if reg.name == _GENERIC_SCRIPT_TOOL_NAME:
            if not any(
                self._skill_runnable_scripts(skill) and self._skill_allows_generic_script_runner(skill)
                for skill in selected
                if not skill.disable_model_invocation
            ):
                raise PermissionError(f"Tool '{tool_name}' not allowed by active skills")
            return reg, None

        if reg.tool_scope == "core":
            return reg, self.skills.get(reg.skill_id)

        selected_map = {skill.id: skill for skill in selected}
        owner = selected_map.get(reg.skill_id)
        if not owner:
            raise PermissionError(f"Tool '{tool_name}' not allowed by active skills")
        if owner.disable_model_invocation:
            raise PermissionError(f"Tool '{tool_name}' is disabled for model invocation")
        if owner.allowed_tools and tool_name not in owner.allowed_tools:
            raise PermissionError(f"Tool '{tool_name}' not allowed by skill policy")
        return reg, owner

    def _prepare_tool_args(
        self,
        reg: RegisteredTool,
        args: Dict[str, Any],
        selected: List[SkillManifest],
        ctx: SkillContext,
    ) -> Dict[str, Any]:
        recovered = _recover_tool_args(args)
        validated = self._validate_tool_args(reg, recovered)
        if reg.name == _GENERIC_ENTRYPOINT_TOOL_NAME:
            validated = self._validate_skill_entrypoint_args(validated, selected, ctx)
        if reg.name == _GENERIC_SCRIPT_TOOL_NAME:
            validated = self._validate_skill_script_args(validated, selected, ctx)
        self._enforce_artifact_materialization_policy(reg, validated, selected, ctx)
        allowed, reason = self._run_pre_action_hooks(selected, ctx, reg.name, validated)
        if not allowed:
            raise PermissionError(reason or "Denied by skill policy")
        return validated

    def _validate_skill_script_args(
        self,
        args: Dict[str, Any],
        selected: List[SkillManifest],
        ctx: SkillContext,
    ) -> Dict[str, Any]:
        selected_with_scripts = [
            skill for skill in selected if self._exposed_relevant_skill_scripts(skill, ctx) and not skill.disable_model_invocation
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

        out = dict(args)
        out["skill_id"] = skill.id
        out["script"] = chosen
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
            skill for skill in selected if self._relevant_skill_entrypoints(skill, ctx) and not skill.disable_model_invocation
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
        candidates = self._relevant_skill_entrypoints(skill, ctx)
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

    @staticmethod
    def _requested_filepaths(reg: RegisteredTool, args: Dict[str, Any]) -> List[str]:
        if reg.name == "create_file":
            return [str(args.get("filepath", ""))]
        if reg.name == "edit_file":
            return [str(args.get("filepath", ""))]
        if reg.name == "create_files":
            paths: List[str] = []
            for item in args.get("files") or []:
                if isinstance(item, dict):
                    paths.append(str(item.get("filepath", "")))
            return paths
        return []

    @staticmethod
    def _is_text_like_path(path: str) -> bool:
        suffix = Path(str(path).strip()).suffix.lower()
        return suffix in _TEXT_LIKE_EXTENSIONS

    def _enforce_artifact_materialization_policy(
        self,
        reg: RegisteredTool,
        args: Dict[str, Any],
        selected: List[SkillManifest],
        ctx: SkillContext,
    ) -> None:
        if reg.name not in {"create_file", "create_files", "edit_file"}:
            return

        requested_paths = [path for path in self._requested_filepaths(reg, args) if path.strip()]
        if not requested_paths:
            return

        opaque_paths = [path for path in requested_paths if not self._is_text_like_path(path)]
        if not opaque_paths:
            return

        relevant_materializers = self._selected_relevant_materializers(selected, opaque_paths)
        if relevant_materializers:
            return

        formatted = ", ".join(sorted(dict.fromkeys(opaque_paths))[:3])
        raise PermissionError(
            "Cannot materialize opaque artifact paths with prompt-only skills via plain workspace text tools: "
            f"{formatted}. Use an executable skill/tool path or decline."
        )

    def _execute_registered_tool(
        self,
        reg: RegisteredTool,
        args: Dict[str, Any],
        env: ToolExecutionEnv,
    ) -> Any:
        if reg.name == _GENERIC_ENTRYPOINT_TOOL_NAME:
            return self._execute_skill_entrypoint_tool(args, env)
        if reg.name == _GENERIC_SCRIPT_TOOL_NAME:
            return self._execute_skill_script_tool(args, env)
        if reg.command:
            return self._execute_command_tool(reg, args, env)
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
        confirm_shell: Optional[Callable[[str], bool]] = None,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            reg, _owner = self._resolve_tool_call(tool_name, selected)
            normalized_args = self._prepare_tool_args(reg, args, selected, ctx)
            env = ToolExecutionEnv(
                workspace=self.workspace,
                memory=self.memory,
                config=self.config,
                debug=self.debug,
                confirm_shell=confirm_shell,
            )
            result = self._execute_registered_tool(reg, normalized_args, env)
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

        proc_env = dict(self._proc_env_base)
        proc_env["ALPHANUS_TOOL_NAME"] = reg.name
        proc_env["ALPHANUS_TOOL_ARGS_JSON"] = json.dumps(args, ensure_ascii=False)
        prepared_command, use_shell = self._prepare_command(reg.command)

        proc = subprocess.run(
            prepared_command,
            shell=use_shell,
            cwd=reg.cwd or str(self.skills_dir),
            capture_output=True,
            text=True,
            input=json.dumps(args, ensure_ascii=False),
            timeout=max(1, int(reg.timeout_s)),
            env=proc_env,
        )

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            msg = stderr or stdout or f"Tool command failed with exit code {proc.returncode}"
            lowered = msg.lower()
            if "permissionerror" in lowered or "operation not permitted" in lowered:
                raise PermissionError(msg)
            if "filenotfounderror" in lowered or "no such file or directory" in lowered:
                raise FileNotFoundError(msg)
            if "timeouterror" in lowered or "timed out" in lowered:
                raise TimeoutError(msg)
            raise RuntimeError(msg)

        out = (proc.stdout or "").strip()
        if not out:
            return {}
        candidate = out.splitlines()[-1].strip()

        try:
            parsed = json.loads(candidate)
        except Exception as exc:
            raise ToolProtocolError(
                "Tool command output is not valid JSON"
                + (f": {exc}" if self.debug else "")
            ) from exc

        if not isinstance(parsed, dict):
            return {"value": parsed}
        return parsed

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
        interpreter = _SCRIPT_INTERPRETER_BY_EXT.get(ext)
        if not interpreter:
            raise PermissionError(f"Unsupported skill script type: {script_path.suffix}")
        if interpreter[0] != sys.executable and shutil.which(interpreter[0]) is None:
            raise FileNotFoundError(f"Missing interpreter for skill script: {interpreter[0]}")

        argv = args.get("argv") if isinstance(args.get("argv"), list) else []
        proc_env = dict(self._proc_env_base)
        proc_env["ALPHANUS_SELECTED_SKILL_ID"] = skill.id
        proc_env["ALPHANUS_SKILL_ROOT"] = str(skill.path)
        proc_env["ALPHANUS_SKILL_SCRIPT"] = rel_script
        proc_env["ALPHANUS_TOOL_ARGS_JSON"] = json.dumps(args.get("args") or {}, ensure_ascii=False)
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
    ) -> Dict[str, Any]:
        caps = env.config.get("capabilities", {})
        dangerously_skip_permissions = bool(caps.get("dangerously_skip_permissions", False))
        shell_require_confirmation = bool(caps.get("shell_require_confirmation", True))
        if shell_require_confirmation and not dangerously_skip_permissions:
            if not env.confirm_shell:
                raise PermissionError("Shell confirmation callback is required")
            if not env.confirm_shell(command):
                raise PermissionError("Shell command rejected by user")
        result = env.workspace.run_shell_command(command, timeout_s=max(1, int(timeout_s)))
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
        for template in entrypoint.install:
            command = self._resolve_entrypoint_placeholders(template, template_values)
            install_results.append(self._run_shell_workflow_command(command, env, timeout_s))
        for template in entrypoint.verify:
            command = self._resolve_entrypoint_placeholders(template, template_values)
            verify_results.append(self._run_shell_workflow_command(command, env, timeout_s))
        command = self._resolve_entrypoint_placeholders(entrypoint.command, template_values)
        run_data = self._run_shell_workflow_command(command, env, timeout_s)
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
