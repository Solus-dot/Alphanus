import argparse
import copy
import getpass
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from agent.core import Agent
from agent.telemetry import configure_logging
from alphanus_paths import get_app_paths
from core.backend_profiles import BACKEND_PROFILE_LABELS, VALID_BACKEND_PROFILES
from core.configuration import (
    DEFAULT_CONFIG,
    config_for_editor_view,
    deep_merge,
    load_dotenv,
    load_global_config,
    normalize_config,
    validate_endpoint_policy,
)
from core.endpoint_modes import ENDPOINT_MODE_AUTO, ENDPOINT_MODE_CHAT, ENDPOINT_MODE_RESPONSES, ENDPOINT_MODES
from core.memory import LexicalMemory
from core.retrieval import SQLiteRetrievalStore, configured_store_path
from core.search_providers import (
    DEFAULT_TAVILY_API_KEY_ENV,
    SEARCH_FALLBACK_PROVIDERS,
    SEARCH_PROVIDER_SEARXNG,
    SEARCH_PROVIDER_TAVILY,
    SEARCH_PROVIDERS,
)
from core.theme_catalog import DEFAULT_THEME_ID, normalize_theme_id
from core.project import ProjectRuntime
from skills.runtime import SkillRuntime
from tui.interface import AlphanusTUI
from tui.themes import available_theme_ids, theme_spec

INIT_SECTIONS = ("all", "model", "search", "theme", "permissions")
_VALID_CLI_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _CliTheme:
    def __init__(self) -> None:
        term = os.environ.get("TERM", "").lower()
        self.enabled = bool(getattr(sys.stdout, "isatty", lambda: False)()) and os.environ.get("NO_COLOR") is None and term != "dumb"
        self.indigo = (99, 102, 241)  # #6366f1
        self.cyan = (34, 211, 238)  # #22d3ee
        self.violet = (167, 139, 250)  # #a78bfa
        self.muted_gray = (161, 161, 170)  # #a1a1aa
        self.foreground = (228, 228, 231)  # #e4e4e7
        self.ok_green = (16, 185, 129)  # #10b981
        self.warn_amber = (245, 158, 11)  # #f59e0b
        self.error_rose = (244, 63, 94)  # #f43f5e

    def _fmt(self, text: str, code: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def _fg(self, text: str, rgb: tuple[int, int, int], *, bold: bool = False, dim: bool = False) -> str:
        parts = []
        if bold:
            parts.append("1")
        if dim:
            parts.append("2")
        parts.append(f"38;2;{rgb[0]};{rgb[1]};{rgb[2]}")
        return self._fmt(text, ";".join(parts))

    def brand(self, text: str) -> str:
        if not self.enabled:
            return text
        return self._fmt(text, "1;38;2;199;210;254;48;2;49;46;129")

    def accent(self, text: str) -> str:
        return self._fg(text, self.indigo, bold=True)

    def cyan_text(self, text: str) -> str:
        return self._fg(text, self.cyan, bold=True)

    def value(self, text: str) -> str:
        return self._fg(text, self.violet, bold=True)

    def muted(self, text: str) -> str:
        return self._fg(text, self.muted_gray, dim=True)

    def ok(self, text: str) -> str:
        return self._fg(text, self.ok_green, bold=True)

    def warn(self, text: str) -> str:
        return self._fg(text, self.warn_amber, bold=True)

    def error(self, text: str) -> str:
        return self._fg(text, self.error_rose, bold=True)

    def label(self, text: str) -> str:
        return self._fg(text, self.foreground, bold=True)

    def step(self, text: str) -> str:
        return f"{self.accent('>')} {self.label(text)}"

    def path(self, text: str) -> str:
        return self._fg(text, (199, 210, 254))

    def rule(self, title: str = "") -> str:
        line = "-" * 52
        if not title:
            return self.muted(line)
        tail = "-" * max(10, 48 - len(title))
        return f"{self.muted('--')} {self.accent(title)} {self.muted(tail)}"


def _as_object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--debug", action="store_true", help="Enable verbose debug logs")
    common.add_argument(
        "--project-root",
        type=str,
        default="",
        help="Project root for this run (defaults to enclosing git root, then cwd)",
    )
    parser = argparse.ArgumentParser(description="Alphanus", parents=[common])
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("run", help="Launch the TUI", parents=[common])

    init_parser = subparsers.add_parser("init", help="Initialize ~/.alphanus config and env", parents=[common])
    init_parser.add_argument(
        "section",
        nargs="?",
        choices=INIT_SECTIONS,
        default="all",
        help="Initialize all settings or a specific section",
    )
    init_parser.add_argument("--non-interactive", action="store_true", help="Use defaults/flags without prompts")
    init_parser.add_argument("--reset", action="store_true", help="Reset selected section(s) to defaults before applying")
    init_parser.add_argument("--base-url", type=str, default="", help="OpenAI-compatible base URL")
    init_parser.add_argument("--responses-endpoint", type=str, default="", help="Responses API endpoint override")
    init_parser.add_argument("--model-endpoint", type=str, default="", help="Model chat completions endpoint")
    init_parser.add_argument("--models-endpoint", type=str, default="", help="Model listing endpoint")
    init_parser.add_argument(
        "--endpoint-mode",
        type=str,
        choices=sorted(ENDPOINT_MODES),
        default="",
        help="Preferred endpoint mode",
    )
    init_parser.add_argument(
        "--backend-profile",
        type=str,
        choices=sorted(VALID_BACKEND_PROFILES),
        default="",
        help="Backend compatibility profile",
    )
    init_parser.add_argument("--api-key", type=str, default="", help="Model API key (writes to ~/.alphanus/.env)")
    init_parser.add_argument("--api-key-env", type=str, default="", help="Environment variable name for model API key")
    init_parser.add_argument(
        "--backend-api-key-env",
        type=str,
        default="",
        help="Optional extra environment variable to mirror the model API key for a local backend process",
    )
    init_parser.add_argument("--search-provider", type=str, choices=list(SEARCH_PROVIDERS), default="", help="Search provider")
    init_parser.add_argument("--search-fallback-provider", type=str, choices=list(SEARCH_FALLBACK_PROVIDERS), default="", help="Search fallback provider")
    init_parser.add_argument("--searxng-base-url", type=str, default="", help="SearXNG base URL")
    init_parser.add_argument("--tavily-api-key", type=str, default="", help="Tavily API key (writes to ~/.alphanus/.env)")
    init_parser.add_argument("--tavily-api-key-env", type=str, default="", help="Environment variable name for Tavily API key")
    init_parser.add_argument(
        "--theme",
        type=str,
        default="",
        help="Theme id",
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Validate config, env, endpoint, and project readiness",
        parents=[common],
    )
    doctor_parser.add_argument("--json", action="store_true", help="Emit doctor report as JSON")

    retrieval_parser = subparsers.add_parser("retrieval", help="Inspect or reset the local retrieval store", parents=[common])
    retrieval_subparsers = retrieval_parser.add_subparsers(dest="retrieval_command")
    retrieval_subparsers.add_parser("stats", help="Show retrieval store statistics")
    reset_parser = retrieval_subparsers.add_parser("reset", help="Delete the retrieval SQLite database")
    reset_parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    return parser


def _load_runtime_config(app_paths: Any, args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    load_dotenv(app_paths.dotenv_path)
    config_path = app_paths.config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Global config not found at {config_path}. Run `alphanus init` first.")
    config_warnings: list[str] = []
    config = load_global_config(config_path, warnings=config_warnings)
    if args.debug:
        debug_dir = app_paths.state_root / "logs"
        debug_dir.mkdir(parents=True, exist_ok=True)
        agent_cfg = config.setdefault("agent", {})
        agent_cfg["debug_log_path"] = str(debug_dir / "http-debug.jsonl")
    config, normalization_warnings = normalize_config(config)
    config_warnings.extend(normalization_warnings)
    config["_project_root_override"] = str(getattr(args, "project_root", "") or "").strip()
    validate_endpoint_policy(config)
    runtime_cfg = config.setdefault("runtime", {})
    if isinstance(runtime_cfg, dict):
        runtime_cfg["state_root"] = str(Path(app_paths.state_root).resolve())
    return config, config_warnings


def _git_root_for_cwd(cwd: Path) -> Path | None:
    git_path = shutil.which("git")
    if not git_path:
        return None
    try:
        proc = subprocess.run(
            [git_path, "-C", str(cwd), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    root_text = proc.stdout.strip()
    return Path(root_text).expanduser().resolve() if root_text else None


def resolve_project_root(config: dict[str, Any], *, override: str = "", cwd: Path | None = None) -> Path:
    if override.strip():
        return Path(os.path.expanduser(override.strip())).resolve()
    start = (cwd or Path.cwd()).resolve()
    strategy = str(_as_object(config.get("project")).get("root_strategy", "git-or-cwd")).strip().lower()
    if strategy == "git-or-cwd":
        return _git_root_for_cwd(start) or start
    return start


def _build_agent_runtime(
    app_paths: Any, config: dict[str, Any], *, debug: bool
) -> tuple[ProjectRuntime, LexicalMemory, SkillRuntime, Agent]:
    project_root = resolve_project_root(config, override=str(config.get("_project_root_override", "")))
    permissions_cfg = _as_object(config.get("permissions"))
    sandbox_cfg = _as_object(config.get("sandbox"))
    project = ProjectRuntime(
        project_root=str(project_root),
        permission_mode=str(permissions_cfg.get("mode", "project-write")),
        network_access=bool(permissions_cfg.get("network", False)),
        sandbox_backend=str(sandbox_cfg.get("backend", "auto")),
        sandbox_fail_closed=bool(sandbox_cfg.get("fail_closed", True)),
    )
    memory_path = str((Path(app_paths.state_root).resolve() / "memory" / "events.jsonl").resolve())
    memory_cfg = config.get("memory", {})
    memory = LexicalMemory(
        storage_path=memory_path,
        min_score=float(memory_cfg.get("min_score_default", 0.3)),
        backup_revisions=int(memory_cfg.get("backup_revisions", 2)),
    )
    user_skills_dir = getattr(app_paths, "user_skills_dir", None)
    runtime_skills_dir = Path(user_skills_dir).resolve() if user_skills_dir is not None else (Path(app_paths.state_root).resolve() / "skills").resolve()
    runtime_skills_dir.mkdir(parents=True, exist_ok=True)
    runtime = SkillRuntime(
        skills_dir=str(runtime_skills_dir),
        bundled_skills_dir=str(app_paths.bundled_skills_dir),
        project=project,
        memory=memory,
        config=config,
        debug=debug,
    )
    agent = Agent(config=config, skill_runtime=runtime, debug=debug)
    return project, memory, runtime, agent


def _prompt_with_default(label: str, default: str, *, hint: str = "", theme: _CliTheme | None = None) -> str:
    default_text = theme.value(default) if theme is not None else default
    prompt = f"{label} [{default_text}]"
    if hint:
        prompt = f"{prompt}\n  {hint}"
    value = input(f"{prompt}: ").strip()
    return value or default


def _prompt_env_name(theme: _CliTheme, label: str, default: str, *, hint: str = "") -> str:
    while True:
        value = _prompt_with_default(label, default, hint=hint, theme=theme).strip()
        if not value:
            return ""
        lowered = value.lower()
        looks_like_secret_value = lowered.startswith(("sk-", "tvly-", "bearer ", "key-")) or (
            len(value) >= 24 and any(char.isdigit() for char in value) and any(char.isalpha() for char in value)
        )
        if looks_like_secret_value:
            print(theme.warn("Enter an environment variable name here, not the API key value."))
            continue
        if not _VALID_CLI_ENV_NAME_RE.match(value):
            print(theme.warn("Environment variable names must use letters, numbers, and underscores, and cannot start with a number."))
            continue
        if value != value.upper():
            print(theme.warn("Use an uppercase environment variable name here, such as ALPHANUS_API_KEY or OPENAI_API_KEY."))
            continue
        return value


def _prompt_yes_no(label: str, *, default: bool = True, theme: _CliTheme | None = None) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    if theme is not None:
        suffix = theme.value(suffix)
    raw = input(f"{label} {suffix}: ").strip().lower()
    if not raw:
        return default
    if raw in {"y", "yes"}:
        return True
    if raw in {"n", "no"}:
        return False
    return default


def _prompt_choice(theme: _CliTheme, label: str, options: list[tuple[str, str]], *, default: str) -> str:
    print(theme.label(label))
    value_width = max((len(value) for value, _desc in options), default=7)
    for idx, (value, description) in enumerate(options, start=1):
        marker = theme.ok("*") if value == default else " "
        print(f"  {marker} {theme.accent(str(idx).rjust(2) + '.')} {theme.value(f'{value:<{value_width}}')}  {theme.muted(description)}")
    default_index = next((idx for idx, (value, _desc) in enumerate(options, start=1) if value == default), 1)
    while True:
        raw = input(f"{theme.cyan_text('Choose option')} {theme.value('[' + str(default_index) + ']')}: ").strip().lower()
        if not raw:
            return default
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        for value, _desc in options:
            if raw == value:
                return value
        print(theme.warn("Invalid choice. Enter option number or name."))


def _section_selected(section: str, name: str) -> bool:
    return section == "all" or section == name


def _upsert_env_var(dotenv_path: Path, key: str, value: str) -> None:
    key = key.strip()
    if not key or not _VALID_CLI_ENV_NAME_RE.match(key):
        return
    dotenv_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if dotenv_path.exists():
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    prefix = f"{key}="
    replaced = False
    updated: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix) or stripped.startswith(f"export {prefix}"):
            updated.append(f"{key}={value}")
            replaced = True
            continue
        updated.append(line)
    if not replaced:
        if updated and updated[-1].strip():
            updated.append("")
        updated.append(f"{key}={value}")
    dotenv_path.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")


def _screen_capture_setup_lines(*, open_settings: bool = False) -> list[tuple[str, str]]:
    system = platform.system().lower()
    if system == "darwin":
        if open_settings:
            subprocess.run(
                ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        return [
            ("macOS Screen Recording", "required for screenshot capture"),
            ("Setup", "System Settings > Privacy & Security > Screen Recording"),
            ("Allow", "the terminal app or launcher used to run Alphanus, then restart it"),
        ]
    if system == "linux":
        binary = shutil.which("gnome-screenshot") or shutil.which("scrot")
        session = os.environ.get("XDG_SESSION_TYPE", "").strip().lower() or "unknown"
        if binary:
            detail = f"found {Path(binary).name}; session={session}"
        else:
            detail = "install gnome-screenshot or scrot"
        if session == "wayland":
            detail += "; compositor/portal permission prompts may still be required"
        return [("Linux screenshot helper", detail)]
    if system == "windows":
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        detail = f"found {Path(powershell).name}" if powershell else "PowerShell is required for screenshot capture"
        return [("Windows screenshot helper", detail)]
    return [(f"{platform.system() or 'Unknown'} screenshot helper", "unsupported by screenshot-ocr")]


def _print_screen_capture_setup(theme: _CliTheme, *, interactive: bool) -> None:
    open_settings = False
    if interactive and platform.system().lower() == "darwin":
        open_settings = _prompt_yes_no("Open macOS Screen Recording settings now?", default=True, theme=theme)
    print("")
    print(theme.accent("Screen Capture Permissions"))
    for label, detail in _screen_capture_setup_lines(open_settings=open_settings):
        print(f"  {theme.label(label + ':')} {detail}")


def _print_init_step(theme: _CliTheme, index: int, total: int, title: str) -> None:
    print(theme.rule(f"Step {index}/{total}"))
    print(theme.cyan_text(title))
    print("")


def _print_review_group(theme: _CliTheme, title: str, rows: list[tuple[str, str]]) -> None:
    print(f"  {theme.accent(title)}")
    label_width = max((len(label) for label, _value in rows), default=8)
    for label, value in rows:
        print(f"    {theme.label((label + ':').ljust(label_width + 1))} {theme.value(value)}")
    print("")


def _doctor_state(theme: _CliTheme, state: str) -> str:
    normalized = state.strip().lower()
    if normalized == "ok":
        return theme.ok("[OK]")
    if normalized == "wait":
        return theme.warn("[WAIT]")
    return theme.error("[FAIL]")


def _print_doctor_group(theme: _CliTheme, title: str, rows: list[tuple[str, str, str]]) -> None:
    print(f"  {theme.accent(title)}")
    label_width = max((len(label) for label, _state, _detail in rows), default=8)
    for label, state, detail in rows:
        line = f"    {theme.label((label + ':').ljust(label_width + 1))} {_doctor_state(theme, state)}"
        if detail:
            line = f"{line}  {theme.muted(detail)}"
        print(line)
    print("")


def _run_init(args: Any) -> int:
    theme = _CliTheme()
    section = str(getattr(args, "section", "all") or "all").strip().lower()
    if section not in INIT_SECTIONS:
        section = "all"
    reset_requested = bool(getattr(args, "reset", False))
    app_paths = get_app_paths()
    state_root = Path(app_paths.state_root)
    state_root.mkdir(parents=True, exist_ok=True)

    base: dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
    existing_warnings: list[str] = []
    if app_paths.config_path.exists():
        try:
            existing = load_global_config(app_paths.config_path, warnings=existing_warnings)
            base = deep_merge(base, existing)
        except (OSError, ValueError):
            base = copy.deepcopy(DEFAULT_CONFIG)
    if reset_requested:
        if section == "all":
            base = copy.deepcopy(DEFAULT_CONFIG)
        else:
            default_cfg = copy.deepcopy(DEFAULT_CONFIG)
            if _section_selected(section, "model"):
                current_agent = base.get("agent", {}) if isinstance(base.get("agent"), dict) else {}
                default_agent = default_cfg.get("agent", {}) if isinstance(default_cfg.get("agent"), dict) else {}
                for key in (
                    "base_url",
                    "responses_endpoint",
                    "model_endpoint",
                    "models_endpoint",
                    "endpoint_mode",
                    "backend_profile",
                    "api_key",
                    "api_key_env",
                    "auth_header_template",
                ):
                    if key in default_agent:
                        current_agent[key] = copy.deepcopy(default_agent[key])
                base["agent"] = current_agent
            if _section_selected(section, "search"):
                base["search"] = copy.deepcopy(default_cfg.get("search", {}))
            if _section_selected(section, "theme"):
                current_tui = base.get("tui", {}) if isinstance(base.get("tui"), dict) else {}
                default_tui = default_cfg.get("tui", {}) if isinstance(default_cfg.get("tui"), dict) else {}
                if "theme" in default_tui:
                    current_tui["theme"] = copy.deepcopy(default_tui["theme"])
                base["tui"] = current_tui

    base_url_default = str(base.get("agent", {}).get("base_url", DEFAULT_CONFIG["agent"]["base_url"]))
    model_endpoint_default = str(base.get("agent", {}).get("model_endpoint", DEFAULT_CONFIG["agent"]["model_endpoint"]))
    responses_endpoint_default = str(base.get("agent", {}).get("responses_endpoint", DEFAULT_CONFIG["agent"]["responses_endpoint"]))
    models_endpoint_default = str(base.get("agent", {}).get("models_endpoint", DEFAULT_CONFIG["agent"]["models_endpoint"]))
    endpoint_mode_default = str(base.get("agent", {}).get("endpoint_mode", DEFAULT_CONFIG["agent"]["endpoint_mode"]))
    backend_profile_default = str(base.get("agent", {}).get("backend_profile", DEFAULT_CONFIG["agent"].get("backend_profile", "auto")))
    api_key_ref_default = str(base.get("agent", {}).get("api_key", DEFAULT_CONFIG["agent"]["api_key"]))
    api_key_env_default = str(base.get("agent", {}).get("api_key_env", DEFAULT_CONFIG["agent"]["api_key_env"]))
    search_provider_default = str(base.get("search", {}).get("provider", DEFAULT_CONFIG["search"]["provider"]))
    search_fallback_default = str(base.get("search", {}).get("fallback_provider", DEFAULT_CONFIG["search"]["fallback_provider"])) or "none"
    tavily_api_key_env_default = str(
        base.get("search", {}).get(
            "tavily_api_key_env",
            DEFAULT_CONFIG["search"].get("tavily_api_key_env", DEFAULT_TAVILY_API_KEY_ENV),
        )
    )
    theme_default = str(base.get("tui", {}).get("theme", DEFAULT_THEME_ID))
    theme_ids = available_theme_ids()
    theme_default, _ = normalize_theme_id(theme_default, default=DEFAULT_THEME_ID, available=theme_ids)

    base_url = base_url_default
    model_endpoint = model_endpoint_default
    responses_endpoint = responses_endpoint_default
    models_endpoint = models_endpoint_default
    endpoint_mode = endpoint_mode_default
    backend_profile = backend_profile_default
    api_key_env = api_key_env_default
    api_key_ref = api_key_ref_default
    api_key_value = ""
    backend_api_key_env = ""
    search_provider = search_provider_default
    search_fallback_provider = search_fallback_default
    searxng_base_url_default = str(base.get("search", {}).get("searxng_base_url", DEFAULT_CONFIG["search"]["searxng_base_url"]))
    searxng_base_url = searxng_base_url_default
    tavily_api_key_env = tavily_api_key_env_default
    tavily_api_key_value = ""
    ui_theme = theme_default

    if args.non_interactive:
        print(theme.brand(" ALPHANUS INIT "))
        print(theme.muted(f"Applying non-interactive setup profile ({section})."))
        if _section_selected(section, "model"):
            base_url = str(getattr(args, "base_url", "") or "").strip() or base_url_default
            model_endpoint = str(getattr(args, "model_endpoint", "") or "").strip() or model_endpoint_default
            responses_endpoint = str(getattr(args, "responses_endpoint", "") or "").strip() or responses_endpoint_default
            models_endpoint = str(getattr(args, "models_endpoint", "") or "").strip() or models_endpoint_default
            endpoint_mode = str(getattr(args, "endpoint_mode", "") or "").strip() or endpoint_mode_default
            backend_profile = str(getattr(args, "backend_profile", "") or "").strip() or backend_profile_default
            api_key_env = str(getattr(args, "api_key_env", "") or "").strip() or api_key_env_default
            api_key_ref = f"env:{api_key_env}"
            api_key_value = str(getattr(args, "api_key", "") or "").strip()
            backend_api_key_env = str(getattr(args, "backend_api_key_env", "") or "").strip()
        if _section_selected(section, "search"):
            search_provider = str(getattr(args, "search_provider", "") or "").strip() or search_provider_default
            search_fallback_provider = (
                str(getattr(args, "search_fallback_provider", "") or "").strip() or search_fallback_default
            )
            searxng_base_url = str(getattr(args, "searxng_base_url", "") or "").strip() or searxng_base_url_default
            tavily_api_key_env = str(getattr(args, "tavily_api_key_env", "") or "").strip() or tavily_api_key_env_default
            tavily_api_key_value = str(getattr(args, "tavily_api_key", "") or "").strip()
            if search_provider == SEARCH_PROVIDER_TAVILY:
                search_fallback_provider = "none"
                searxng_base_url = ""
        if _section_selected(section, "theme"):
            requested_theme = str(getattr(args, "theme", "") or "").strip() or theme_default
            ui_theme, _ = normalize_theme_id(requested_theme, default=theme_default, available=theme_ids)
    else:
        steps = [name for name in ("model", "search", "theme", "permissions") if _section_selected(section, name)]
        total_steps = max(len(steps), 1)
        step_index = 1
        print(theme.brand(" ALPHANUS SETUP "))
        print(theme.rule())
        print(theme.muted(f"Wizard scope: {section}. Press Enter to keep defaults."))
        print(theme.muted("We'll configure runtime state, model connectivity, search, and display theme defaults."))
        print(f"{theme.label('State root:')} {theme.path(str(state_root))}")
        print("")
        if _section_selected(section, "model"):
            _print_init_step(theme, step_index, total_steps, "Model endpoint")
            use_local = _prompt_yes_no(
                "Use the local OpenAI-compatible endpoint preset (127.0.0.1:8080)?",
                default=base_url_default == str(DEFAULT_CONFIG["agent"]["base_url"]),
                theme=theme,
            )
            if use_local:
                base_url = str(DEFAULT_CONFIG["agent"]["base_url"])
                model_endpoint = str(DEFAULT_CONFIG["agent"]["model_endpoint"])
                responses_endpoint = str(DEFAULT_CONFIG["agent"]["responses_endpoint"])
                models_endpoint = str(DEFAULT_CONFIG["agent"]["models_endpoint"])
                endpoint_mode = str(DEFAULT_CONFIG["agent"]["endpoint_mode"])
                backend_profile = str(DEFAULT_CONFIG["agent"].get("backend_profile", "auto"))
                print(theme.muted("Applied local preset for /v1/responses, /v1/chat/completions, and /v1/models."))
            else:
                base_url = _prompt_with_default(
                    "Base URL",
                    base_url_default,
                    hint=theme.muted("provider root, e.g. https://api.openai.com"),
                    theme=theme,
                )
                model_endpoint = _prompt_with_default(
                    "Chat endpoint",
                    model_endpoint_default,
                    hint=theme.muted("chat completions endpoint"),
                    theme=theme,
                )
                responses_endpoint = _prompt_with_default(
                    "Responses endpoint",
                    responses_endpoint_default,
                    hint=theme.muted("responses endpoint"),
                    theme=theme,
                )
                models_endpoint = _prompt_with_default(
                    "Models endpoint",
                    models_endpoint_default,
                    hint=theme.muted("model catalog endpoint"),
                    theme=theme,
                )
                endpoint_mode = _prompt_choice(
                    theme,
                    "Endpoint mode:",
                    [
                        (ENDPOINT_MODE_AUTO, "responses first with fallback to chat"),
                        (ENDPOINT_MODE_RESPONSES, "force /v1/responses"),
                        (ENDPOINT_MODE_CHAT, "force /v1/chat/completions"),
                    ],
                    default=endpoint_mode_default if endpoint_mode_default in ENDPOINT_MODES else ENDPOINT_MODE_AUTO,
                )
                backend_profile = _prompt_choice(
                    theme,
                    "Backend profile:",
                    [(profile, BACKEND_PROFILE_LABELS.get(profile, profile)) for profile in sorted(VALID_BACKEND_PROFILES)],
                    default=(
                        backend_profile_default
                        if backend_profile_default in VALID_BACKEND_PROFILES
                        else "auto"
                    ),
                )
            api_key_env = _prompt_env_name(
                theme,
                "Alphanus API key env var name",
                api_key_env_default,
                hint=theme.muted("name only, not the key value; where Alphanus reads the key"),
            )
            api_key_ref = f"env:{api_key_env.strip() or 'ALPHANUS_API_KEY'}"
            api_key_value = getpass.getpass("API key (stored in ~/.alphanus/.env; leave blank to keep current): ").strip()
            if api_key_value:
                backend_api_key_env = _prompt_env_name(
                    theme,
                    "Also write key to backend env var",
                    "",
                    hint=theme.muted("optional name only; use OPENAI_API_KEY if your local backend requires it"),
                )
            step_index += 1
            print("")
        if _section_selected(section, "search"):
            _print_init_step(theme, step_index, total_steps, "Search")
            search_provider = _prompt_choice(
                theme,
                "Primary provider:",
                [
                    ("searxng", "local/private search when a SearXNG instance is running"),
                    ("tavily", "hosted fallback search using TAVILY_API_KEY"),
                ],
                default=search_provider_default if search_provider_default in SEARCH_PROVIDERS else SEARCH_PROVIDER_SEARXNG,
            )
            if search_provider == SEARCH_PROVIDER_SEARXNG:
                searxng_base_url = _prompt_with_default(
                    "SearXNG base URL",
                    searxng_base_url_default,
                    hint=theme.muted("used by SearXNG, e.g. http://127.0.0.1:8888"),
                    theme=theme,
                )
                search_fallback_provider = _prompt_choice(
                    theme,
                    "Fallback provider:",
                    [
                        ("tavily", "use TAVILY_API_KEY if SearXNG is unavailable"),
                        ("none", "do not use a hosted fallback"),
                    ],
                    default=search_fallback_default if search_fallback_default in SEARCH_FALLBACK_PROVIDERS else SEARCH_PROVIDER_TAVILY,
                )
            else:
                searxng_base_url = ""
                search_fallback_provider = "none"
            if search_provider == SEARCH_PROVIDER_TAVILY or search_fallback_provider == SEARCH_PROVIDER_TAVILY:
                tavily_api_key_env = _prompt_env_name(
                    theme,
                    "Tavily API key env var name",
                    tavily_api_key_env_default or DEFAULT_TAVILY_API_KEY_ENV,
                    hint=theme.muted("name only, not the key value; where Alphanus reads the Tavily key"),
                )
                tavily_api_key_value = getpass.getpass(
                    "Tavily API key (stored in ~/.alphanus/.env; leave blank to keep current): "
                ).strip()
            step_index += 1
            print("")
        if _section_selected(section, "theme"):
            _print_init_step(theme, step_index, total_steps, "Theme")
            theme_options = [(name, theme_spec(name).description) for name in theme_ids]
            selected_theme = _prompt_choice(
                theme,
                "Choose a UI theme:",
                theme_options,
                default=theme_default,
            )
            ui_theme, _ = normalize_theme_id(selected_theme, default=theme_default, available=theme_ids)
            step_index += 1
            print("")
        if _section_selected(section, "permissions"):
            _print_init_step(theme, step_index, total_steps, "Permissions")
            _print_screen_capture_setup(theme, interactive=True)
            print("")

    updates: dict[str, Any] = {}
    if _section_selected(section, "model"):
        updates["agent"] = {
            "base_url": base_url,
            "model_endpoint": model_endpoint,
            "responses_endpoint": responses_endpoint,
            "models_endpoint": models_endpoint,
            "endpoint_mode": endpoint_mode,
            "backend_profile": backend_profile,
            "api_key": api_key_ref,
            "api_key_env": api_key_env,
        }
    if _section_selected(section, "search"):
        updates["search"] = {
            "provider": search_provider,
            "fallback_provider": search_fallback_provider,
            "searxng_base_url": searxng_base_url,
            "tavily_api_key_env": tavily_api_key_env,
        }
    if _section_selected(section, "theme"):
        updates["tui"] = {"theme": ui_theme}
    merged = deep_merge(base, updates)
    try:
        normalized, warnings = normalize_config(merged)
        validate_endpoint_policy(normalized)
    except ValueError as exc:
        print(f"{theme.error('init failed:')} {exc}")
        return 2

    if not args.non_interactive:
        print(theme.rule("Review"))
        _print_review_group(
            theme,
            "Model",
            [
                ("Base URL", str(normalized["agent"]["base_url"])),
                ("Responses", str(normalized["agent"]["responses_endpoint"])),
                ("Chat", str(normalized["agent"]["model_endpoint"])),
                ("Models", str(normalized["agent"]["models_endpoint"])),
                ("Mode", str(normalized["agent"]["endpoint_mode"])),
                ("Backend", str(normalized["agent"]["backend_profile"])),
                ("API key", str(normalized["agent"]["api_key"])),
            ],
        )
        _print_review_group(
            theme,
            "Search",
            [
                ("Provider", str(normalized["search"]["provider"])),
                ("Fallback", str(normalized["search"]["fallback_provider"] or "none")),
                ("SearXNG", str(normalized["search"]["searxng_base_url"] or "(not set)")),
                ("Tavily env", str(normalized["search"]["tavily_api_key_env"])),
            ],
        )
        _print_review_group(
            theme,
            "Interface",
            [
                ("Theme", str(normalized["tui"]["theme"])),
                ("Secrets", str(app_paths.dotenv_path)),
            ],
        )
        if not _prompt_yes_no("Write these settings now?", default=True, theme=theme):
            print(theme.warn("Setup cancelled. No files were written."))
            return 1

    cleaned = config_for_editor_view(normalized)
    app_paths.config_path.parent.mkdir(parents=True, exist_ok=True)
    app_paths.config_path.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
    app_paths.dotenv_path.parent.mkdir(parents=True, exist_ok=True)
    if not app_paths.dotenv_path.exists():
        app_paths.dotenv_path.write_text(
            "\n".join(
                [
                    "# Alphanus environment variables",
                    "# Uncomment and set values as needed.",
                    "# ALPHANUS_API_KEY=",
                    "# OPENAI_API_KEY=",
                    "# TAVILY_API_KEY=",
                    "# ALPHANUS_EMBEDDINGS_API_KEY=",
                    "# ALPHANUS_AUTH_HEADER=Authorization: Bearer <token>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if _section_selected(section, "model") and api_key_value:
        _upsert_env_var(app_paths.dotenv_path, normalized["agent"]["api_key_env"], api_key_value)
        if backend_api_key_env:
            _upsert_env_var(app_paths.dotenv_path, backend_api_key_env, api_key_value)
    if _section_selected(section, "search") and tavily_api_key_value:
        _upsert_env_var(app_paths.dotenv_path, normalized["search"]["tavily_api_key_env"], tavily_api_key_value)

    for warning in existing_warnings + warnings:
        print(f"{theme.warn('config warning:')} {warning}")
    if args.non_interactive and _section_selected(section, "permissions"):
        _print_screen_capture_setup(theme, interactive=False)
    print("")
    print(theme.ok("Initialization complete."))
    print(f"  {theme.label('Config:')} {theme.path(str(app_paths.config_path))}")
    print(f"  {theme.label('Env:')} {theme.path(str(app_paths.dotenv_path))}")
    print(f"  {theme.label('Sessions:')} {theme.path(str((Path(app_paths.state_root) / 'sessions').resolve()))}")
    print(f"  {theme.label('Memory:')} {theme.path(str((Path(app_paths.state_root) / 'memory').resolve()))}")
    print("")
    print(theme.accent("Next steps"))
    print(f"  {theme.step('uv run alphanus doctor')}  {theme.muted('validate configuration and dependencies')}")
    print(f"  {theme.step('uv run alphanus')}         {theme.muted('launch the interface')}")
    return 0


def _run_doctor(args: Any) -> int:
    theme = _CliTheme()
    app_paths = get_app_paths()
    try:
        config, warnings = _load_runtime_config(app_paths, args)
    except (FileNotFoundError, OSError, ValueError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc), "config_warnings": []}, indent=2))
            return 2
        print(f"{theme.error('doctor config error:')} {exc}")
        return 2

    try:
        project, memory, _runtime, agent = _build_agent_runtime(app_paths, config, debug=args.debug)
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc), "config_warnings": warnings}, indent=2))
            return 2
        print(f"{theme.error('doctor runtime error:')} {exc}")
        return 2
    report = agent.doctor_report()
    report_obj = report if isinstance(report, dict) else {}
    if args.json:
        payload: dict[str, Any] = dict(report_obj) if report_obj else {"doctor": report}
        payload["ok"] = True
        payload["config_warnings"] = warnings
        print(json.dumps(payload, indent=2))
    else:
        agent_status = _as_object(report_obj.get("agent"))
        project_status = _as_object(report_obj.get("project"))
        search_status = _as_object(report_obj.get("search"))
        retrieval_status = _as_object(report_obj.get("retrieval"))
        endpoint_error = str(agent_status.get("endpoint_policy_error") or "").strip()
        endpoint_ok = not endpoint_error
        model_ok = bool(agent_status.get("ready"))
        project_ok = bool(project_status.get("exists")) and bool(project_status.get("writable"))
        search_ok = bool(search_status.get("ready"))
        retrieval_ok = bool(retrieval_status.get("ready"))

        print("")
        print(theme.brand(" ALPHANUS DOCTOR "))
        print(theme.rule())
        print(theme.muted("Validating model connectivity, project access, search, and local state."))
        print("")

        if warnings:
            print(theme.rule("Config Warnings"))
            for warning in warnings:
                print(f"  {theme.warn('!')} {warning}")
            print("")

        _print_review_group(
            theme,
            "Runtime Paths",
            [
                ("Config", str(app_paths.config_path)),
                ("Env", str(app_paths.dotenv_path)),
                ("Sessions", str((Path(app_paths.state_root) / "sessions").resolve())),
                ("Memory", str(memory.storage_path)),
                ("Project", str(project.project_root)),
            ],
        )

        search_detail = str(search_status.get("reason") or "").strip() or "ok"
        retrieval_detail = str(retrieval_status.get("reason") or "").strip() or "ok"
        _print_doctor_group(
            theme,
            "Checks",
            [
                ("Endpoint policy", "ok" if endpoint_ok else "fail", "" if endpoint_ok else endpoint_error),
                ("Model readiness", "ok" if model_ok else "wait", ""),
                ("Project access", "ok" if project_ok else "fail", ""),
                ("Search provider", "ok" if search_ok else "wait", "" if search_detail == "ok" else search_detail),
                ("Retrieval store", "ok" if retrieval_ok else "fail", "" if retrieval_detail == "ok" else retrieval_detail),
            ],
        )

    failures = []
    agent_status = _as_object(report_obj.get("agent"))
    project_status = _as_object(report_obj.get("project"))
    search_status = _as_object(report_obj.get("search"))
    retrieval_status = _as_object(report_obj.get("retrieval"))
    if str(agent_status.get("endpoint_policy_error") or "").strip():
        failures.append("endpoint-policy")
    if not bool(agent_status.get("ready")):
        failures.append("model-readiness")
    if not bool(project_status.get("exists")) or not bool(project_status.get("writable")):
        failures.append("project")
    if not bool(search_status.get("ready")):
        failures.append("search")
    if not bool(retrieval_status.get("ready")):
        failures.append("retrieval")
    if not args.json:
        print(theme.rule("Result"))
        if failures:
            print(f"  {_doctor_state(theme, 'fail')} {theme.label('Attention required')}  {theme.muted(', '.join(failures))}")
            print(f"  {theme.muted('Fix failing checks, then rerun `uv run alphanus doctor`.')}")
        else:
            print(f"  {_doctor_state(theme, 'ok')} {theme.label('Ready')}")
        print("")
    return 1 if failures else 0


def _run_retrieval(args: Any) -> int:
    app_paths = get_app_paths()
    theme = _CliTheme()
    try:
        config, warnings = _load_runtime_config(app_paths, args)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"retrieval failed: {exc}")
        return 2
    for warning in warnings:
        print(f"{theme.warn('config warning:')} {warning}")

    db_path = configured_store_path(config)
    command = str(getattr(args, "retrieval_command", "") or "stats")
    if command == "reset":
        if not bool(getattr(args, "yes", False)) and not _prompt_yes_no(f"Delete retrieval store at {db_path}?", default=False):
            print(theme.warn("Reset cancelled."))
            return 1
        deleted = []
        for suffix in ("", "-wal", "-shm"):
            target = Path(f"{db_path}{suffix}")
            if target.exists():
                target.unlink()
                deleted.append(str(target))
        SQLiteRetrievalStore(db_path)
        print(theme.ok("Retrieval store reset."))
        if deleted:
            for path in deleted:
                print(f"  {theme.path(path)}")
        return 0

    stats = SQLiteRetrievalStore(db_path).stats()
    print(theme.brand(" ALPHANUS RETRIEVAL "))
    print(f"  {theme.label('Store:')} {theme.path(str(db_path))}")
    print(f"  {theme.label('Records:')} {stats['records']}")
    print(f"  {theme.label('Chunks:')} {stats['chunks']}")
    print(f"  {theme.label('Stale web records:')} {stats['stale_records']}")
    by_type = stats.get("by_type", {})
    if isinstance(by_type, dict) and by_type:
        print(f"  {theme.label('By type:')} " + ", ".join(f"{key}={value}" for key, value in sorted(by_type.items())))
    return 0


def _run_tui(args: Any) -> int:
    app_paths = get_app_paths()
    try:
        config, config_warnings = _load_runtime_config(app_paths, args)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"startup failed: {exc}")
        return 2

    logger = configure_logging(config)
    for warning in config_warnings:
        logger.warning(f"config: {warning}")

    try:
        _project, memory, _runtime, agent = _build_agent_runtime(app_paths, config, debug=args.debug)
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
        print(f"startup failed: {exc}")
        return 2
    if args.debug:
        logger.info(f"debug HTTP log: {config['agent']['debug_log_path']}")

    agent_cfg = _as_object(config.get("agent"))
    if not bool(agent_cfg.get("tls_verify", True)):
        logger.warning("TLS verification is disabled (agent.tls_verify=false)")
    memory_stats = memory.stats()
    logger.info(f"[info] memory mode: {memory_stats.get('mode_label', 'lexical')}")
    logger.info(f"[info] memory min_score_default: {memory_stats.get('min_score_default', 0.3)}")
    logger.info("use /doctor inside the TUI for readiness and health diagnostics.")

    logger.info("startup skips blocking model handshake; TUI status will refresh asynchronously.")

    app = AlphanusTUI(agent=agent, debug=args.debug)
    app.run()
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    command = str(getattr(args, "command", "") or "")
    if command == "init":
        return _run_init(args)
    if command == "doctor":
        return _run_doctor(args)
    if command == "retrieval":
        return _run_retrieval(args)
    return _run_tui(args)
