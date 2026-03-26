from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
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
_LOAD_SKILL_TOOL_NAME = "load_skill"
_READ_SKILL_RESOURCE_TOOL_NAME = "read_skill_resource"
_RUN_SKILL_COMMAND_TOOL_NAME = "run_skill_command"
_SPAWN_SKILL_AGENT_TOOL_NAME = "spawn_skill_agent"
_REQUEST_USER_INPUT_TOOL_NAME = "request_user_input"
_SCRIPT_INTERPRETER_BY_EXT = {
    ".py": [sys.executable],
    ".sh": ["bash"],
    ".js": ["node"],
    ".mjs": ["node"],
}
_SKILL_DIR_NAMES = ("skills", ".claude/skills", ".agents/skills", ".opencode/skills")
_USER_SKILL_DIRS = (
    ".alphanus/skills",
    ".claude/skills",
    ".agents/skills",
    ".config/opencode/skills",
)
_TRUSTED_SOURCE_TIERS = frozenset({"workspace/local", "bundled"})
_SOURCE_PRIORITY = {
    "workspace/local": 0,
    "bundled": 1,
    "user/local": 2,
    "external/local": 3,
}
_PACK_AGENT_DIR_NAMES = ("agents", "agents-codex", ".claude/agents", ".agents/agents", ".config/opencode/agents")
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
_SHELL_FENCE_LANGS = frozenset({"", "bash", "console", "shell", "sh", "zsh"})
_INLINE_COMMAND_HINT_RE = re.compile(r"\b(?:bootstrap|command|commands|convert|install|requires?|run|setup|validate|verify)\b", re.IGNORECASE)
_COMMAND_SETUP_HINT_RE = re.compile(r"\b(?:bootstrap|dependency|install|setup)\b", re.IGNORECASE)
_COMMAND_VERIFY_HINT_RE = re.compile(r"\b(?:check|preflight|validate|verify)\b", re.IGNORECASE)


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
    explicit_skill_id: str = ""
    explicit_skill_args: str = ""


@dataclass(slots=True)
class ToolExecutionEnv:
    workspace: WorkspaceManager
    memory: VectorMemory
    config: Dict[str, Any]
    debug: bool
    confirm_shell: Optional[Callable[[str], bool]] = None
    spawn_skill_agent: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    request_user_input: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


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


@dataclass(slots=True)
class SkillPackRecord:
    id: str
    root: Path
    source_tier: str
    source_label: str
    skill_ids: List[str] = field(default_factory=list)
    agent_names: List[str] = field(default_factory=list)


@dataclass(slots=True)
class AgentRecord:
    name: str
    description: str
    prompt: str
    path: Path
    pack_id: str
    source_tier: str
    model: str = ""
    reasoning_effort: str = ""
    sandbox_mode: str = ""
    web_search_mode: str = ""
    tool_policy: Dict[str, Any] = field(default_factory=dict)
    resource_roots: List[str] = field(default_factory=list)


@dataclass(slots=True)
class LoadedSkillContract:
    skill_id: str
    prompt: str
    resources: List[str]
    scripts: List[str]
    blocked_scripts: List[Dict[str, str]]
    commands: List[str]
    install_commands: List[str]
    verify_commands: List[str]
    entrypoints: List[str]
    agents: List[str]
    availability_code: str = "ready"
    availability_reason: str = ""
    argument_text: str = ""


@dataclass(slots=True)
class LoadedAgentContract:
    agent_name: str
    prompt: str
    skill_id: str = ""
    resource_roots: List[str] = field(default_factory=list)


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
        self.tools_cfg = self.config.get("tools", {})
        self.selection_mode = str(self.skills_cfg.get("selection_mode", "all_enabled")).strip().lower()
        if self.selection_mode == "hybrid_lazy":
            self.selection_mode = "hybrid_lazy"
        self.max_active_skills = int(self.skills_cfg.get("max_active_skills", 6))
        self.shortlist_size = max(1, int(self.skills_cfg.get("shortlist_size", 6)))
        load_cfg = self.skills_cfg.get("load", {}) if isinstance(self.skills_cfg.get("load"), dict) else {}
        self.upward_scan = bool(load_cfg.get("upward_scan", True))
        extra_dirs = load_cfg.get("extra_dirs", [])
        if isinstance(extra_dirs, str):
            extra_dirs = [extra_dirs]
        self.extra_skill_dirs = [Path(os.path.expanduser(str(item))).resolve() for item in extra_dirs if str(item).strip()]
        configured_python = str(load_cfg.get("python_executable") or self.skills_cfg.get("python_executable") or "").strip()
        self.python_executable = configured_python or sys.executable
        raw_core_policy = self.tools_cfg.get(
            "core_exposure_policy",
            self.skills_cfg.get("core_exposure_policy", "coding_core"),
        )
        self.core_exposure_policy = str(raw_core_policy or "coding_core").strip().lower()
        self.generation = 0

        self.skill_roots = self._discover_skill_roots()
        self.skills: Dict[str, SkillManifest] = {}
        self._all_skills: List[SkillManifest] = []
        self._shadowed_skills: List[SkillManifest] = []
        self.skill_packs: Dict[str, SkillPackRecord] = {}
        self.agents: Dict[str, AgentRecord] = {}
        self._skill_agents_by_pack: Dict[str, List[str]] = {}
        self._skill_index: Dict[str, Dict[str, Any]] = {}
        self._tool_registry: Dict[str, RegisteredTool] = {}
        self._list_skills_cache: Optional[Tuple[SkillManifest, ...]] = None
        self._enabled_skills_cache: Optional[Tuple[SkillManifest, ...]] = None
        self._skill_catalog_cache: Dict[int, str] = {}
        self._skill_cards_cache: Dict[Tuple[Tuple[str, ...], bool], str] = {}
        self._loaded_skill_contracts: Dict[tuple[str, str], LoadedSkillContract] = {}
        self._python_module_probe_cache: Dict[str, bool] = {}
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
        env["ALPHANUS_SKILL_PYTHON"] = str(self.python_executable)
        env["ALPHANUS_CONFIG_JSON"] = json.dumps(self.config, ensure_ascii=False)

        # Let skill scripts import project modules without per-script sys.path hacks.
        repo_root = str(self.skills_dir.parent)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = repo_root if not existing else repo_root + os.pathsep + existing

        npm_path = shutil.which("npm")
        if npm_path:
            try:
                proc = subprocess.run(
                    [npm_path, "root", "-g"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except Exception:
                proc = None
            node_root = (proc.stdout or "").strip() if proc and proc.returncode == 0 else ""
            if node_root:
                existing_node_path = env.get("NODE_PATH", "")
                env["NODE_PATH"] = node_root if not existing_node_path else node_root + os.pathsep + existing_node_path
        return env

    def _discover_skill_roots(self) -> List[Path]:
        roots: List[Path] = []
        seen: set[str] = set()

        def add(path: Path) -> None:
            resolved = path.resolve()
            key = str(resolved)
            if key in seen:
                return
            seen.add(key)
            roots.append(resolved)

        current = self.workspace.workspace_root.resolve()
        while True:
            for rel in _SKILL_DIR_NAMES:
                add(current / rel)
            if not self.upward_scan or current.parent == current:
                break
            current = current.parent

        home_root = self.workspace.home_root.resolve()
        for rel in _USER_SKILL_DIRS:
            add(home_root / rel)

        add(self.skills_dir)
        for path in self.extra_skill_dirs:
            add(path)
        return roots

    def _root_source_tier(self, root: Path) -> str:
        root_resolved = root.resolve()
        if self._is_relative_to(root_resolved, self.skills_dir.resolve()):
            return "bundled"
        if self._is_relative_to(root_resolved, self.workspace.workspace_root.resolve()):
            return "workspace/local"
        if self._is_relative_to(root_resolved, self.workspace.home_root.resolve()):
            return "user/local"
        return "external/local"

    def _discover_skill_dirs(self, root: Path) -> List[Path]:
        if not root.exists():
            return []
        candidates: List[Path] = []
        seen: set[str] = set()
        docs = [root / SKILL_DOC] if (root / SKILL_DOC).exists() else []
        if root.is_dir():
            try:
                docs.extend(sorted(root.rglob(SKILL_DOC)))
            except Exception:
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

    def _pack_root_for_skill(self, skill_dir: Path, root: Path) -> Path:
        skill_dir = skill_dir.resolve()
        root = root.resolve()
        root_parent = root.parent.resolve()
        if root_parent != root and any((root_parent / rel).exists() for rel in _PACK_AGENT_DIR_NAMES):
            return root_parent
        for candidate in [skill_dir] + list(skill_dir.parents):
            if not self._is_relative_to(candidate, root):
                continue
            if candidate == root:
                return candidate
            if (candidate / "skills").exists():
                return candidate
            if any((candidate / rel).exists() for rel in _PACK_AGENT_DIR_NAMES):
                return candidate
        return root

    @staticmethod
    def _pack_id_for_root(root: Path) -> str:
        text = re.sub(r"[^a-z0-9]+", "-", root.name.lower()).strip("-") or "skill-pack"
        digest = hashlib.sha1(str(root.resolve()).encode("utf-8")).hexdigest()[:10]
        return f"{text}-{digest}"

    def _discover_agents_for_pack(self, pack_root: Path, source_tier: str) -> List[AgentRecord]:
        records: List[AgentRecord] = []
        pack_id = self._pack_id_for_root(pack_root)
        for rel in _PACK_AGENT_DIR_NAMES:
            agent_root = pack_root / rel
            if not agent_root.exists():
                continue
            for path in sorted(agent_root.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in {".md", ".toml"}:
                    continue
                try:
                    record = self._load_agent_record(path.resolve(), pack_id, source_tier)
                except Exception:
                    continue
                if record.name not in self.agents:
                    records.append(record)
        return records

    def _load_agent_record(self, path: Path, pack_id: str, source_tier: str) -> AgentRecord:
        if path.suffix.lower() == ".toml":
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            prompt = str(data.get("developer_instructions") or "").strip()
            tool_policy = {key: value for key, value in data.items() if key not in {"developer_instructions", "description"}}
            return AgentRecord(
                name=path.stem.replace("_", "-"),
                description=str(data.get("description") or path.stem).strip(),
                prompt=prompt,
                path=path,
                pack_id=pack_id,
                source_tier=source_tier,
                model=str(data.get("model") or "").strip(),
                reasoning_effort=str(data.get("model_reasoning_effort") or "").strip(),
                sandbox_mode=str(data.get("sandbox_mode") or "").strip(),
                web_search_mode=str(data.get("web_search") or "").strip(),
                tool_policy=tool_policy,
                resource_roots=[str(path.parent)],
            )
        frontmatter, prompt = extract_skill_doc(path, include_prompt=True)
        return AgentRecord(
            name=str(frontmatter.get("name") or path.stem).strip(),
            description=str(frontmatter.get("description") or path.stem).strip(),
            prompt=str(prompt or "").strip(),
            path=path,
            pack_id=pack_id,
            source_tier=source_tier,
            model=str(frontmatter.get("model") or "").strip(),
            reasoning_effort=str(frontmatter.get("effort") or "").strip(),
            sandbox_mode=str(frontmatter.get("sandbox") or frontmatter.get("mode") or "").strip(),
            web_search_mode=str(frontmatter.get("web_search") or "").strip(),
            tool_policy={"tools": frontmatter.get("tools")},
            resource_roots=[str(path.parent)],
        )

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
        manifest.source_tier = self._skill_source_tier(manifest)
        manifest.trust_level = self._trust_level_for_source_tier(manifest.source_tier)
        manifest.execution_allowed = manifest.trust_level == "trusted"
        manifest.adapter = str(getattr(manifest, "vendor_flavor", "") or manifest.format or "agentskills")
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
        if not manifest.execution_allowed:
            return None
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
        self._skill_cards_cache = {}

    @staticmethod
    def _append_unique(items: List[str], value: str) -> None:
        text = str(value).strip()
        if text and text not in items:
            items.append(text)

    def _source_priority(self, source_tier: str) -> int:
        return _SOURCE_PRIORITY.get(str(source_tier).strip(), 99)

    @staticmethod
    def _trust_level_for_source_tier(source_tier: str) -> str:
        return "trusted" if source_tier in _TRUSTED_SOURCE_TIERS else "untrusted"

    def _manifest_priority(self, manifest: SkillManifest) -> Tuple[int, str]:
        path = manifest.path or manifest.doc_path
        return (self._source_priority(manifest.source_tier), str(path or manifest.id))

    def _has_executable_surface(self, manifest: SkillManifest) -> bool:
        if manifest.command_tools or manifest.entrypoints:
            return True
        if manifest.path is None:
            return False
        if (manifest.path / "tools.py").exists():
            return True
        if manifest.hooks_path and manifest.hooks_path.exists():
            return True
        if self._skill_runnable_scripts(manifest):
            return True
        return False

    def _validate_manifest_policy(self, manifest: SkillManifest) -> None:
        if manifest.command_tools:
            self._append_unique(manifest.blocked_features, "command_tools")
            self._append_unique(manifest.validation_warnings, "command_tools disabled_pending_safe_runner")

        if manifest.trust_level != "trusted":
            manifest.execution_allowed = False
            self._append_unique(manifest.blocked_features, "untrusted_root")
            if manifest.path is not None:
                if (manifest.path / "tools.py").exists():
                    self._append_unique(manifest.blocked_features, "tools.py")
                if manifest.hooks_path and manifest.hooks_path.exists():
                    self._append_unique(manifest.blocked_features, "hooks.py")
            if manifest.entrypoints:
                self._append_unique(manifest.blocked_features, "entrypoints")
            if self._skill_runnable_scripts(manifest):
                self._append_unique(manifest.blocked_features, "scripts")
            if self._has_executable_surface(manifest):
                self._append_unique(
                    manifest.validation_errors,
                    "untrusted skill roots are metadata-only; executable surfaces are blocked",
                )
            manifest.available = False
            manifest.availability_code = "untrusted"
            manifest.availability_reason = "untrusted skill roots are metadata-only"

    def _record_shadowed(self, manifest: SkillManifest, winner: SkillManifest) -> None:
        manifest.execution_allowed = False
        manifest.available = False
        manifest.availability_code = "shadowed"
        manifest.shadowed_by = winner.id
        source = self.skill_source_label(winner) or winner.id
        manifest.availability_reason = f"shadowed by {winner.id} ({source})"
        self._append_unique(winner.shadowing, manifest.id)
        if manifest in self._all_skills:
            self._all_skills.remove(manifest)
        self._shadowed_skills.append(manifest)
        self._all_skills.append(manifest)

    def _rebuild_skill_index(self) -> None:
        self._skill_index = {}
        for skill in self.enabled_skills():
            self._skill_index[skill.id] = {
                "id": skill.id,
                "name": skill.name,
                "description": skill.description,
                "tags": list(skill.tags),
                "categories": list(skill.categories),
                "keywords": list(skill.triggers.get("keywords", [])),
                "file_ext": list(skill.triggers.get("file_ext", [])),
                "tools": list(skill.allowed_tools),
                "produces": list(skill.produces),
                "entrypoints": [entry.name for entry in self._skill_entrypoints(skill)],
                "scripts": self._skill_runnable_scripts(skill),
                "source_tier": skill.source_tier,
                "trust_level": skill.trust_level,
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

    def _core_policy_names(self) -> frozenset[str]:
        policy = self.core_exposure_policy or "coding_core"
        names = _CORE_EXPOSURE_POLICIES.get(policy)
        if names is None:
            return _CORE_EXPOSURE_POLICIES["coding_core"]
        return names

    def _register_runtime_tools(self) -> None:
        self._tool_registry[_LOAD_SKILL_TOOL_NAME] = RegisteredTool(
            name=_LOAD_SKILL_TOOL_NAME,
            skill_id="__runtime__",
            tool_scope="skill",
            capability="skill_loader",
            description="Load the full content and packaged resources for a selected skill.",
            parameters={
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string"},
                    "arguments": {"type": "string"},
                },
                "required": ["skill_id"],
            },
        )
        self._tool_registry[_READ_SKILL_RESOURCE_TOOL_NAME] = RegisteredTool(
            name=_READ_SKILL_RESOURCE_TOOL_NAME,
            skill_id="__runtime__",
            tool_scope="skill",
            capability="skill_resource_reader",
            description="Read a supporting file bundled with a loaded skill.",
            parameters={
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        )
        self._tool_registry[_RUN_SKILL_COMMAND_TOOL_NAME] = RegisteredTool(
            name=_RUN_SKILL_COMMAND_TOOL_NAME,
            skill_id="__runtime__",
            tool_scope="skill",
            capability="skill_command_runner",
            description="Run a packaged skill workflow command relative to the skill or workspace root.",
            parameters={
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string"},
                    "command": {"type": "string"},
                    "cwd": {"type": "string", "enum": ["skill", "workspace"]},
                    "timeout_s": {"type": "integer"},
                },
                "required": ["command"],
            },
        )
        self._tool_registry[_SPAWN_SKILL_AGENT_TOOL_NAME] = RegisteredTool(
            name=_SPAWN_SKILL_AGENT_TOOL_NAME,
            skill_id="__runtime__",
            tool_scope="skill",
            capability="skill_agent_runner",
            description="Start, inspect, or wait for a companion agent provided by the active skill pack.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["start", "status", "wait"]},
                    "agent_name": {"type": "string"},
                    "prompt": {"type": "string"},
                    "task_id": {"type": "string"},
                    "background": {"type": "boolean"},
                    "timeout_s": {"type": "integer"},
                },
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
        self.skill_roots = self._discover_skill_roots()
        self.skills = {}
        self._all_skills = []
        self._shadowed_skills = []
        self.skill_packs = {}
        self.agents = {}
        self._skill_agents_by_pack = {}
        self._skill_index = {}
        self._tool_registry = {}
        self._loaded_skill_contracts = {}
        self._invalidate_skill_caches()
        self._register_runtime_tools()
        if not any(root.exists() for root in self.skill_roots):
            return

        for root in self.skill_roots:
            if not root.exists():
                continue
            source_tier = self._root_source_tier(root)
            pack_agents_registered: set[str] = set()
            for child in self._discover_skill_dirs(root):
                manifest: Optional[SkillManifest] = None
                try:
                    manifest = self._load_manifest(child)
                    if manifest is None:
                        continue

                    if manifest.id in previous_enabled:
                        manifest.enabled = previous_enabled[manifest.id]

                    pack_root = self._pack_root_for_skill(child, root)
                    pack_id = self._pack_id_for_root(pack_root)
                    pack = self.skill_packs.get(pack_id)
                    if pack is None:
                        pack = SkillPackRecord(
                            id=pack_id,
                            root=pack_root,
                            source_tier=source_tier,
                            source_label=str(pack_root),
                        )
                        self.skill_packs[pack_id] = pack
                    manifest.metadata.setdefault("_pack_id", pack_id)
                    manifest.source_tier = source_tier
                    manifest.trust_level = self._trust_level_for_source_tier(source_tier)
                    manifest.execution_allowed = manifest.trust_level == "trusted"
                    (
                        manifest.available,
                        manifest.availability_code,
                        manifest.availability_reason,
                    ) = self._check_skill_availability(manifest)
                    self._validate_manifest_policy(manifest)

                    existing = self.skills.get(manifest.id)
                    if existing is not None:
                        if self._manifest_priority(manifest) < self._manifest_priority(existing):
                            self._remove_skill_tools(existing.id)
                            self.skills.pop(existing.id, None)
                            self._record_shadowed(existing, manifest)
                        else:
                            self._record_shadowed(manifest, existing)
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
                    pack.skill_ids.append(manifest.id)
                    if manifest.execution_allowed and pack_id not in pack_agents_registered:
                        for agent in self._discover_agents_for_pack(pack_root, source_tier):
                            if agent.name in self.agents:
                                continue
                            self.agents[agent.name] = agent
                            pack.agent_names.append(agent.name)
                        pack_agents_registered.add(pack_id)
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

    def _skill_source_tier(self, manifest: SkillManifest) -> str:
        path = manifest.path or manifest.doc_path
        if not path:
            return "external/local"
        try:
            if self._is_relative_to(path.resolve(), self.skills_dir.resolve()):
                return "bundled"
            if self._is_relative_to(path.resolve(), self.workspace.workspace_root.resolve()):
                return "workspace/local"
            if self._is_relative_to(path.resolve(), self.workspace.home_root.resolve()):
                return "user/local"
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

        if manifest.command_tools:
            self._append_unique(manifest.validation_errors, "command_tools are disabled_pending_safe_runner")

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
                    key=lambda s: (s.id, self._source_priority(s.source_tier), self.skill_source_label(s)),
                )
            )
        return list(self._list_skills_cache)

    def list_agents(self) -> List[AgentRecord]:
        return [self.agents[name] for name in sorted(self.agents)]

    def get_agent(self, agent_name: str) -> Optional[AgentRecord]:
        return self.agents.get(str(agent_name).strip())

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

    @staticmethod
    def skill_provenance_label(skill: SkillManifest) -> str:
        return str(getattr(skill, "source_tier", "") or "external/local")

    @staticmethod
    def skill_status_label(skill: SkillManifest) -> Tuple[str, str]:
        if skill.shadowed_by:
            return "shadowed", "yellow"
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
                if skill.enabled and skill.available and skill.execution_allowed and not skill.shadowed_by
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
                "trust_level": skill.trust_level,
                "execution_allowed": bool(skill.execution_allowed),
                "adapter": skill.adapter,
                "status": self.skill_status_label(skill)[0],
                "source_tier": self.skill_provenance_label(skill),
                "source": self.skill_source_label(skill),
                "pack_id": self._skill_pack_id(skill),
                "availability_code": skill.availability_code or "ready",
                "availability_reason": skill.availability_reason or "ready",
                "tools": self._reported_skill_tools(skill),
                "scripts": self._reported_skill_scripts(skill),
                "entrypoints": [entry.name for entry in self._reported_skill_entrypoints(skill)],
                "agents": self._reported_skill_agents(skill),
                "user_invocable": bool(skill.user_invocable),
                "model_invocable": not bool(skill.disable_model_invocation),
                "validation_errors": list(skill.validation_errors),
                "validation_warnings": list(skill.validation_warnings),
                "blocked_features": list(skill.blocked_features),
                "shadowed_by": skill.shadowed_by,
                "shadowing": list(skill.shadowing),
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
        return sorted(dict.fromkeys(runnable))

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
            )
            available = proc.returncode == 0
        except Exception:
            available = False
        self._python_module_probe_cache[module_name] = available
        return available

    @staticmethod
    def _entrypoint_intents(entrypoint: SkillEntrypointDef) -> set[str]:
        intents = {str(item).strip().lower() for item in entrypoint.intents if str(item).strip()}
        return intents or {"general"}

    def _skill_entrypoints(self, skill: SkillManifest) -> List[SkillEntrypointDef]:
        return list(getattr(skill, "entrypoints", []) or [])

    def _reported_skill_tools(self, skill: SkillManifest) -> List[str]:
        if not getattr(skill, "execution_allowed", True):
            return []
        return list(getattr(skill, "allowed_tools", []) or [])

    def _reported_skill_scripts(self, skill: SkillManifest) -> List[str]:
        if not getattr(skill, "execution_allowed", True):
            return []
        return self._skill_runnable_scripts(skill)

    def _reported_skill_entrypoints(self, skill: SkillManifest) -> List[SkillEntrypointDef]:
        if not getattr(skill, "execution_allowed", True):
            return []
        return self._skill_entrypoints(skill)

    def _reported_skill_agents(self, skill: SkillManifest) -> List[str]:
        if not getattr(skill, "execution_allowed", True):
            return []
        return self._agents_for_skill(skill)

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
        blocks = re.findall(r"```([^\n`]*)\n(.*?)```", prompt, flags=re.DOTALL)
        for lang_raw, block in blocks:
            lang = str(lang_raw or "").strip().lower().split(None, 1)[0] if str(lang_raw or "").strip() else ""
            if lang not in _SHELL_FENCE_LANGS:
                continue
            for raw_line in block.splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(("-", "*")):
                    line = line[1:].strip()
                if not line:
                    continue
                if self._looks_like_shell_workflow_command(line):
                    commands.append(line)
        for raw_line in prompt.splitlines():
            line = raw_line.strip()
            if "`" not in line:
                continue
            if not _INLINE_COMMAND_HINT_RE.search(line) and not any(token in line.lower() for token in _SHELL_WORKFLOW_HINTS):
                continue
            for candidate in re.findall(r"`([^`\n]+)`", line):
                command = candidate.strip()
                if self._looks_like_shell_workflow_command(command):
                    commands.append(command)
        if ctx is None:
            return sorted(dict.fromkeys(commands))

        intents = self.task_intents(ctx)
        requested_exts = self.requested_artifact_extensions(ctx)
        if requested_exts and not self._skill_supports_artifact(skill, requested_exts, intents=intents):
            return []
        return sorted(dict.fromkeys(commands))

    def _runtime_visible_skill_commands(self, skill: SkillManifest, ctx: Optional[SkillContext]) -> List[str]:
        if not getattr(skill, "execution_allowed", True):
            return []
        return sorted(
            dict.fromkeys(self._rebase_vendor_paths(command, skill) for command in self._skill_shell_workflow_commands(skill, ctx))
        )

    @staticmethod
    def _looks_like_shell_workflow_command(text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        try:
            parts = shlex.split(raw)
        except ValueError:
            parts = raw.split()
        if not parts:
            return False
        head = Path(parts[0]).name.lower()
        if head == "sudo" and len(parts) > 1:
            head = Path(parts[1]).name.lower()
            parts = parts[1:]
        if head not in _SHELL_WORKFLOW_HINTS:
            return False
        if head in {"python", "python3"} and len(parts) >= 2:
            target = str(parts[1]).strip()
            return target.endswith(".py") or "/" in target
        return True

    def _skill_install_commands(self, skill: SkillManifest) -> List[str]:
        commands = list(self._runtime_visible_skill_commands(skill, None))
        commands.extend(self._rebase_vendor_paths(command, skill) for entrypoint in self._skill_entrypoints(skill) for command in entrypoint.install)
        return sorted(dict.fromkeys(command for command in commands if _COMMAND_SETUP_HINT_RE.search(command)))

    def _skill_verify_commands(self, skill: SkillManifest) -> List[str]:
        commands = list(self._runtime_visible_skill_commands(skill, None))
        commands.extend(self._rebase_vendor_paths(command, skill) for entrypoint in self._skill_entrypoints(skill) for command in entrypoint.verify)
        return sorted(dict.fromkeys(command for command in commands if _COMMAND_VERIFY_HINT_RE.search(command)))

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

    @staticmethod
    def should_use_deterministic_selection(scored: List[Tuple[int, SkillManifest]]) -> bool:
        if len(scored) <= 1:
            return True
        top_score = scored[0][0]
        second_score = scored[1][0]
        if top_score >= 60 and top_score - second_score >= 18:
            return True
        if top_score >= 90 and second_score <= 24:
            return True
        return False

    def score_skills(self, ctx: SkillContext) -> List[Tuple[int, SkillManifest]]:
        text = ctx.user_input.lower()
        attachments = " ".join(ctx.attachments).lower()
        query_tokens = set(self._match_tokens(ctx.user_input))
        time_sensitive = bool(query_tokens & _TIME_SENSITIVE_TOKENS)
        scored: List[Tuple[int, SkillManifest]] = []
        requested_exts = self.requested_artifact_extensions(ctx)
        intents = self.task_intents(ctx)

        for skill in self.enabled_skills():
            if skill.disable_model_invocation and skill.id != ctx.explicit_skill_id:
                continue
            index = self._skill_index.get(skill.id, {})
            score = 0

            if ctx.explicit_skill_id and skill.id == ctx.explicit_skill_id:
                score += 1000

            for kw in index.get("keywords", skill.triggers.get("keywords", [])):
                kw_lower = kw.lower()
                if kw_lower not in text:
                    continue
                if skill.id == "memory-rag" and kw_lower == "my":
                    if self._looks_like_memory_request(text, query_tokens):
                        score += 30
                    continue
                score += 30

            for ext in index.get("file_ext", skill.triggers.get("file_ext", [])):
                ext_lower = ext.lower()
                if ext_lower in text or ext_lower in attachments:
                    score += 20

            metadata_text = " ".join(
                [index.get("name", skill.name), index.get("description", skill.description)]
                + list(index.get("tags", skill.tags))
                + list(index.get("categories", skill.categories))
                + list(index.get("produces", skill.produces))
            )
            metadata_tokens = set(self._match_tokens(metadata_text))
            overlap = query_tokens & metadata_tokens
            if overlap:
                score += min(24, len(overlap) * 6)
                if any(token in skill.name.lower() for token in overlap):
                    score += 4

            if time_sensitive:
                temporal_text = " ".join([index.get("description", skill.description)] + list(index.get("tags", skill.tags)))
                temporal_tokens = set(self._match_tokens(temporal_text))
                if temporal_tokens & {"latest", "recent", "current", "news", "internet", "web", "online", "lookup"}:
                    score += 18

            if "memory" in skill.id and ctx.memory_hits:
                score += 10
            if skill.id == "memory-rag" and self._looks_like_memory_request(text, query_tokens):
                score += 16
            if skill.id == "shell-ops" and self._looks_like_shell_request(text, query_tokens):
                score += 28
            if requested_exts and self._skill_supports_artifact(skill, requested_exts, intents=intents):
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
        if ctx.explicit_skill_id:
            explicit = self.resolve_skill_reference(ctx.explicit_skill_id)
            if explicit and explicit.enabled and explicit.available and explicit.execution_allowed:
                return self.expand_selected_skills(ctx, [explicit])
        if self.selection_mode == "all_enabled":
            enabled = self.enabled_skills()
            enabled.sort(key=lambda s: s.id)
            if self.max_active_skills <= 0:
                return self.expand_selected_skills(ctx, enabled)
            return self.expand_selected_skills(ctx, enabled[: self.max_active_skills])

        limit = self.max_active_skills if self.max_active_skills > 0 else top_n
        shortlist = self.shortlist_size if self.shortlist_size > 0 else max(limit * 3, 6)
        candidates = self.propose_skill_candidates(ctx, top_n=max(limit * 3, shortlist))
        selected = self.rerank_skill_candidates(ctx, candidates, max(1, limit))
        return self.expand_selected_skills(ctx, selected)

    @staticmethod
    def _skill_has_runtime_surface(skill: SkillManifest) -> bool:
        return bool(
            getattr(skill, "allowed_tools", None)
            or getattr(skill, "command_tools", None)
            or getattr(skill, "entrypoints", None)
            or getattr(skill, "bundled_files", None)
        )

    def default_loaded_skill_ids(self, selected: List[SkillManifest], ctx: SkillContext) -> List[str]:
        loaded: List[str] = []
        for skill in selected:
            skill_id = str(getattr(skill, "id", "")).strip()
            if ctx.explicit_skill_id and skill_id == ctx.explicit_skill_id:
                loaded.append(skill_id)
                continue
            if bool(getattr(skill, "disable_model_invocation", False)):
                continue
            if not self._skill_has_runtime_surface(skill):
                loaded.append(skill_id)
        return sorted(dict.fromkeys(loaded))

    def loaded_skills(self, selected: List[SkillManifest], loaded_skill_ids: List[str]) -> List[SkillManifest]:
        selected_map = {skill.id: skill for skill in selected}
        out: List[SkillManifest] = []
        for skill_id in loaded_skill_ids:
            skill = selected_map.get(skill_id) or self.get_skill(skill_id)
            if skill:
                out.append(skill)
        return out

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
                f"trust: {getattr(skill, 'trust_level', 'trusted')} | execution: {'yes' if getattr(skill, 'execution_allowed', True) else 'no'} | "
                f"adapter: {getattr(skill, 'adapter', 'agentskills')} | user_invocable: {'yes' if getattr(skill, 'user_invocable', True) else 'no'} | "
                f"model_invocable: {'no' if getattr(skill, 'disable_model_invocation', False) else 'yes'}"
            )
        rendered = "\n".join(lines)
        self._skill_cards_cache[key] = rendered
        return rendered

    def requested_opaque_artifact_extensions(self, ctx: SkillContext) -> List[str]:
        return [ext for ext in self.requested_artifact_extensions(ctx) if ext not in _TEXT_LIKE_EXTENSIONS]

    def _skill_runtime_note(self, skill: SkillManifest, ctx: SkillContext) -> str:
        requested_exts = self.requested_artifact_extensions(ctx)
        intents = self.task_intents(ctx)
        relevant_entrypoints = self._relevant_skill_entrypoints(skill, ctx)
        relevant_scripts = self._exposed_relevant_skill_scripts(skill, ctx)
        shell_commands = self._runtime_visible_skill_commands(skill, ctx)
        install_commands = self._skill_install_commands(skill)
        verify_commands = self._skill_verify_commands(skill)
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
            install_text = "; ".join(install_commands[:3]) if install_commands else "none"
            verify_text = "; ".join(verify_commands[:3]) if verify_commands else "none"
            return (
                "Runtime note:\n"
                f"- This skill exposes a shell/python workflow for {', '.join(requested_exts[:3])} via shell_command.\n"
                f"- Use documented commands only: {'; '.join(shell_commands[:4])}\n"
                f"- Documented install commands: {install_text}\n"
                f"- Documented verify commands: {verify_text}\n"
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
            hooks = self._ensure_skill_hooks(skill) if skill.execution_allowed else None
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

    def _skill_pack_id(self, skill: SkillManifest) -> str:
        return str(skill.metadata.get("_pack_id", "")).strip()

    def _agents_for_skill(self, skill: SkillManifest) -> List[str]:
        if not skill.execution_allowed:
            return []
        pack_id = self._skill_pack_id(skill)
        pack = self.skill_packs.get(pack_id)
        if not pack:
            return []
        return list(pack.agent_names)

    @staticmethod
    def _extract_argument_item(argument_text: str, index: int) -> str:
        if index < 0:
            return ""
        try:
            parts = shlex.split(argument_text)
        except Exception:
            parts = argument_text.split()
        return parts[index] if index < len(parts) else ""

    def _rebase_vendor_paths(self, text: str, skill: SkillManifest) -> str:
        out = str(text or "")
        skill_root = str(skill.path or "")
        if not skill_root:
            return out
        replacements: Dict[str, str] = {}
        for candidate in self.list_skills():
            if not candidate.path:
                continue
            for prefix in ("~/.codex/skills", "~/.claude/skills", "~/.config/opencode/skills", "~/.agents/skills"):
                replacements[f"{prefix}/{candidate.id}"] = str(candidate.path)
            for prefix in ("/home/weizhena/.codex/skills", "/home/weizhena/.claude/skills"):
                replacements[f"{prefix}/{candidate.id}"] = str(candidate.path)
        pack_id = self._skill_pack_id(skill)
        pack = self.skill_packs.get(pack_id)
        if pack:
            for agent_name in pack.agent_names:
                agent = self.agents.get(agent_name)
                if not agent:
                    continue
                agent_root = str(agent.path.parent)
                for prefix in ("~/.codex/agents", "~/.claude/agents", "~/.config/opencode/agents", "~/.agents/agents"):
                    replacements[f"{prefix}/{agent_name}"] = str(agent.path)
                    replacements[f"{prefix}/web-search-modules"] = str(Path(agent_root) / "web-search-modules")
                for prefix in ("/home/weizhena/.codex/agents", "/home/weizhena/.claude/agents"):
                    replacements[f"{prefix}/web-search-modules"] = str(Path(agent_root) / "web-search-modules")
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

    def load_skill_contract(self, skill_id: str, argument_text: str = "") -> LoadedSkillContract:
        skill = self.resolve_skill_reference(skill_id)
        resolved_skill_id = skill.id if skill else str(skill_id).strip()
        key = (resolved_skill_id, argument_text.strip())
        cached = self._loaded_skill_contracts.get(key)
        if cached is not None:
            return cached
        if skill is None or not skill.path:
            raise FileNotFoundError(f"Skill not found: {skill_id}")
        prompt = self._ensure_skill_prompt(skill).strip()
        prompt = prompt.replace("${CLAUDE_SKILL_DIR}", str(skill.path)).replace("${OPENCODE_SKILL_DIR}", str(skill.path))
        prompt = self._apply_skill_arguments(prompt, argument_text.strip())
        prompt = self._rebase_vendor_paths(prompt, skill)
        contract = LoadedSkillContract(
            skill_id=skill.id,
            prompt=prompt,
            resources=sorted(
                rel
                for rel in skill.bundled_files
                if not self._is_skill_script_candidate(rel) and rel != "tools.py" and rel != "hooks.py"
            ),
            scripts=self._reported_skill_scripts(skill),
            blocked_scripts=self._blocked_skill_scripts(skill) if skill.execution_allowed else [],
            commands=self._runtime_visible_skill_commands(skill, None),
            install_commands=self._skill_install_commands(skill),
            verify_commands=self._skill_verify_commands(skill),
            entrypoints=[entry.name for entry in self._reported_skill_entrypoints(skill)],
            agents=self._reported_skill_agents(skill),
            availability_code=str(skill.availability_code or "ready"),
            availability_reason=str(skill.availability_reason or ""),
            argument_text=argument_text.strip(),
        )
        self._loaded_skill_contracts[key] = contract
        return contract

    def load_agent_contract(self, agent_name: str, skill_id: str = "") -> LoadedAgentContract:
        agent = self.get_agent(agent_name)
        if agent is None:
            raise FileNotFoundError(f"Agent not found: {agent_name}")
        skill = self.resolve_skill_reference(skill_id) if skill_id else None
        if skill is not None and not skill.execution_allowed:
            raise PermissionError(f"Skill '{skill.id}' is blocked by trust policy")
        if skill is not None and self._skill_pack_id(skill) != agent.pack_id:
            raise PermissionError(
                f"Agent '{agent_name}' does not belong to the active skill pack for '{skill.id}'"
            )
        prompt = str(agent.prompt or "").strip()
        if skill is not None:
            prompt = self._rebase_vendor_paths(prompt, skill)
            prompt = (
                prompt.replace("${SKILL_DIR}", str(skill.path))
                .replace("${PACK_ROOT}", str(self.skill_packs.get(agent.pack_id).root if self.skill_packs.get(agent.pack_id) else ""))
                .replace("${WORKSPACE_ROOT}", str(self.workspace.workspace_root))
            )
        return LoadedAgentContract(
            agent_name=agent.name,
            prompt=prompt,
            skill_id=skill.id if skill is not None else "",
            resource_roots=list(agent.resource_roots),
        )

    def read_skill_resource(self, skill_id: str, relpath: str) -> Dict[str, Any]:
        skill = self.get_skill(skill_id)
        if skill is None or not skill.path:
            raise FileNotFoundError(f"Skill not found: {skill_id}")
        target = (skill.path / relpath).resolve()
        if not self._is_relative_to(target, skill.path.resolve()):
            raise PermissionError("Skill resource path escapes skill root")
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"Skill resource not found: {relpath}")
        return {
            "skill_id": skill.id,
            "path": relpath,
            "content": target.read_text(encoding="utf-8"),
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

    def allowed_tool_names(
        self,
        selected: List[SkillManifest],
        ctx: Optional[SkillContext] = None,
        loaded_skill_ids: Optional[List[str]] = None,
    ) -> List[str]:
        loaded_skill_ids = loaded_skill_ids or []
        names = self.core_tool_names()
        names.extend(self.optional_tool_names(selected, ctx=ctx))
        if any(getattr(skill, "id", "") not in set(loaded_skill_ids) for skill in selected):
            names.append(_LOAD_SKILL_TOOL_NAME)
        if loaded_skill_ids:
            names.extend(
                [
                    _READ_SKILL_RESOURCE_TOOL_NAME,
                    _RUN_SKILL_COMMAND_TOOL_NAME,
                    _SPAWN_SKILL_AGENT_TOOL_NAME,
                ]
            )
            runtime_cfg = self.config.get("runtime", {}) if isinstance(self.config.get("runtime"), dict) else {}
            if bool(runtime_cfg.get("ask_user_tool", True)):
                names.append(_REQUEST_USER_INPUT_TOOL_NAME)
        if self.selected_shell_workflow_skills(selected, ctx):
            names = [name for name in names if name != "run_checks"]
        return sorted(dict.fromkeys(names))

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
        loaded_skill_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        tools = []
        loaded_skill_ids = loaded_skill_ids or []
        for name in names:
            reg = self._tool_registry[name]
            parameters = reg.parameters
            description = reg.description
            if reg.name == _LOAD_SKILL_TOOL_NAME and selected is not None:
                candidates = [skill.id for skill in selected if not skill.disable_model_invocation or (ctx and ctx.explicit_skill_id == skill.id)]
                if candidates:
                    parameters = {
                        "type": "object",
                        "properties": {
                            "skill_id": {"type": "string", "enum": sorted(dict.fromkeys(candidates))},
                            "arguments": {"type": "string"},
                        },
                        "required": ["skill_id"],
                    }
                    description = f"{reg.description} Available skills: {', '.join(sorted(dict.fromkeys(candidates))[:8])}."
            if reg.name in {_READ_SKILL_RESOURCE_TOOL_NAME, _RUN_SKILL_COMMAND_TOOL_NAME, _SPAWN_SKILL_AGENT_TOOL_NAME} and loaded_skill_ids:
                parameters = dict(parameters) if isinstance(parameters, dict) else {}
                properties = dict(parameters.get("properties", {}))
                properties["skill_id"] = {"type": "string", "enum": sorted(dict.fromkeys(loaded_skill_ids))}
                parameters["type"] = "object"
                parameters["properties"] = properties
                if reg.name == _RUN_SKILL_COMMAND_TOOL_NAME:
                    command_values: List[str] = []
                    for skill_id in loaded_skill_ids:
                        skill = self.get_skill(skill_id)
                        if skill:
                            command_values.extend(self._runtime_visible_skill_commands(skill, None))
                    command_values = sorted(dict.fromkeys(command_values))
                    if command_values:
                        properties["command"] = {"type": "string", "enum": command_values}
                        description = (
                            "Run one documented shell workflow command declared by the loaded skill. "
                            "Do not invent commands. "
                            f"Available commands: {', '.join(command_values[:6])}."
                        )
                if reg.name == _SPAWN_SKILL_AGENT_TOOL_NAME:
                    agent_names: List[str] = []
                    for skill_id in loaded_skill_ids:
                        skill = self.get_skill(skill_id)
                        if skill:
                            agent_names.extend(self._agents_for_skill(skill))
                    if agent_names:
                        properties["agent_name"] = {"type": "string", "enum": sorted(dict.fromkeys(agent_names))}
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

    def tools_for_turn(
        self,
        selected: List[SkillManifest],
        ctx: Optional[SkillContext] = None,
        loaded_skill_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return self._tool_schemas(
            self.allowed_tool_names(selected, ctx=ctx, loaded_skill_ids=loaded_skill_ids),
            selected=selected,
            ctx=ctx,
            loaded_skill_ids=loaded_skill_ids,
        )

    def _run_pre_action_hooks(
        self,
        selected: List[SkillManifest],
        ctx: SkillContext,
        action_name: str,
        args: Dict[str, Any],
    ) -> Tuple[bool, str]:
        for skill in selected:
            if not skill.execution_allowed:
                continue
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
            if not skill.execution_allowed:
                continue
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

        if reg.name in {
            _LOAD_SKILL_TOOL_NAME,
            _READ_SKILL_RESOURCE_TOOL_NAME,
            _RUN_SKILL_COMMAND_TOOL_NAME,
            _SPAWN_SKILL_AGENT_TOOL_NAME,
            _REQUEST_USER_INPUT_TOOL_NAME,
        }:
            return reg, None

        if reg.name == _GENERIC_ENTRYPOINT_TOOL_NAME:
            if not any(
                self._skill_entrypoints(skill)
                for skill in selected
                if not skill.disable_model_invocation and skill.execution_allowed
            ):
                raise PermissionError(f"Tool '{tool_name}' not allowed by active skills")
            return reg, None

        if reg.name == _GENERIC_SCRIPT_TOOL_NAME:
            if not any(
                self._skill_runnable_scripts(skill) and self._skill_allows_generic_script_runner(skill)
                for skill in selected
                if not skill.disable_model_invocation and skill.execution_allowed
            ):
                raise PermissionError(f"Tool '{tool_name}' not allowed by active skills")
            return reg, None

        if reg.tool_scope == "core":
            return reg, self.skills.get(reg.skill_id)

        selected_map = {skill.id: skill for skill in selected}
        owner = selected_map.get(reg.skill_id)
        if not owner:
            raise PermissionError(f"Tool '{tool_name}' not allowed by active skills")
        if not owner.execution_allowed:
            raise PermissionError(f"Tool '{tool_name}' is blocked by skill trust policy")
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
        if reg.name == _RUN_SKILL_COMMAND_TOOL_NAME:
            validated = self._validate_skill_command_args(validated, selected)
        self._enforce_artifact_materialization_policy(reg, validated, selected, ctx)
        allowed, reason = self._run_pre_action_hooks(selected, ctx, reg.name, validated)
        if not allowed:
            raise PermissionError(reason or "Denied by skill policy")
        return validated

    def _validate_skill_command_args(
        self,
        args: Dict[str, Any],
        selected: List[SkillManifest],
    ) -> Dict[str, Any]:
        requested_skill_id = str(args.get("skill_id", "")).strip()
        if requested_skill_id:
            skill = self.resolve_skill_reference(requested_skill_id)
        elif len(selected) == 1:
            skill = selected[0]
        else:
            raise ValueError("run_skill_command requires skill_id when multiple skills are selected")
        if skill is None or not skill.path:
            raise FileNotFoundError(f"Skill not found: {requested_skill_id or 'selected skill'}")
        if not skill.execution_allowed:
            raise PermissionError(f"Skill '{skill.id}' is blocked by trust policy")

        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("Missing required argument: command")
        declared = self._runtime_visible_skill_commands(skill, None)
        if command not in declared:
            raise PermissionError(
                f"Command is not declared by skill '{skill.id}'. Use one of the documented commands exposed by load_skill."
            )
        out = dict(args)
        out["skill_id"] = skill.id
        out["command"] = command
        return out

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
            skill
            for skill in selected
            if self._relevant_skill_entrypoints(skill, ctx) and not skill.disable_model_invocation and skill.execution_allowed
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
        if reg.name == _LOAD_SKILL_TOOL_NAME:
            contract = self.load_skill_contract(str(args.get("skill_id", "")).strip(), str(args.get("arguments", "")).strip())
            return {
                "skill_id": contract.skill_id,
                "prompt": contract.prompt,
                "resources": contract.resources,
                "scripts": contract.scripts,
                "blocked_scripts": contract.blocked_scripts,
                "commands": contract.commands,
                "install_commands": contract.install_commands,
                "verify_commands": contract.verify_commands,
                "entrypoints": contract.entrypoints,
                "agents": contract.agents,
                "availability_code": contract.availability_code,
                "availability_reason": contract.availability_reason,
                "arguments": contract.argument_text,
            }
        if reg.name == _READ_SKILL_RESOURCE_TOOL_NAME:
            return self.read_skill_resource(str(args.get("skill_id", "")).strip(), str(args.get("path", "")).strip())
        if reg.name == _RUN_SKILL_COMMAND_TOOL_NAME:
            return self._execute_skill_command_tool(args, env)
        if reg.name == _SPAWN_SKILL_AGENT_TOOL_NAME:
            if not env.spawn_skill_agent:
                raise PermissionError("Skill agent runtime is unavailable")
            return env.spawn_skill_agent(args)
        if reg.name == _REQUEST_USER_INPUT_TOOL_NAME:
            if not env.request_user_input:
                raise PermissionError("User input runtime is unavailable")
            return env.request_user_input(args)
        if reg.name == _GENERIC_ENTRYPOINT_TOOL_NAME:
            return self._execute_skill_entrypoint_tool(args, env)
        if reg.name == _GENERIC_SCRIPT_TOOL_NAME:
            return self._execute_skill_script_tool(args, env)
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
        spawn_skill_agent: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        request_user_input: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
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
                spawn_skill_agent=spawn_skill_agent,
                request_user_input=request_user_input,
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
        result = env.workspace.run_shell_command(command, timeout_s=max(1, int(timeout_s)), cwd=cwd)
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

    def _execute_skill_command_tool(
        self,
        args: Dict[str, Any],
        env: ToolExecutionEnv,
    ) -> Dict[str, Any]:
        skill = self.get_skill(str(args.get("skill_id", "")).strip())
        if skill is None or not skill.path:
            raise FileNotFoundError("Selected skill root is unavailable")
        command = self._rebase_vendor_paths(str(args.get("command", "")).strip(), skill)
        if not command:
            raise ValueError("Missing required argument: command")
        cwd_mode = str(args.get("cwd", "skill")).strip().lower() or "skill"
        cwd = str(skill.path) if cwd_mode == "skill" else str(self.workspace.workspace_root)
        timeout_s = max(1, int(args.get("timeout_s", 30)))
        run_data = self._run_shell_workflow_command(command, env, timeout_s, cwd=cwd)
        return {
            "skill_id": skill.id,
            "command": command,
            "stdout": run_data.get("stdout", ""),
            "stderr": run_data.get("stderr", ""),
            "returncode": run_data.get("returncode", 0),
            "cwd": run_data.get("cwd", ""),
        }

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
