import argparse
import copy
import getpass
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

from alphanus_paths import get_app_paths
from agent.core import Agent
from agent.telemetry import configure_logging
from core.configuration import (
    DEFAULT_CONFIG,
    config_for_editor_view,
    deep_merge,
    load_dotenv,
    load_global_config,
    normalize_config,
    resolve_path,
    validate_endpoint_policy,
)
from core.theme_catalog import BUILTIN_THEME_IDS, DEFAULT_THEME_ID, THEME_ALIASES, normalize_theme_id
from core.memory import LexicalMemory
from core.skills import SkillRuntime
from core.workspace import WorkspaceManager
from tui.interface import AlphanusTUI
from tui.themes import theme_spec


INIT_SECTIONS = ("all", "workspace", "model", "search", "theme")


class _CliTheme:
    def __init__(self) -> None:
        term = os.environ.get("TERM", "").lower()
        self.enabled = bool(getattr(sys.stdout, "isatty", lambda: False)()) and os.environ.get("NO_COLOR") is None and term != "dumb"
        self.indigo = (99, 102, 241)  # #6366f1
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
        return self._fmt(text, "1;38;2;99;102;241;48;2;26;23;48")

    def heading(self, text: str) -> str:
        return self._fg(text, self.indigo, bold=True)

    def accent(self, text: str) -> str:
        return self._fg(text, self.indigo, bold=True)

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


def _resolve_runtime_skills_dir(app_paths) -> Path:
    repo_root = getattr(app_paths, "repo_root", None)
    if repo_root is not None:
        return (Path(repo_root).resolve() / "skills").resolve()
    return (Path(app_paths.app_root).resolve() / "skills").resolve()


def _seed_runtime_skills(runtime_skills_dir: Path, bundled_skills_dir: Path) -> None:
    runtime_skills_dir.mkdir(parents=True, exist_ok=True)
    try:
        if runtime_skills_dir.resolve() == bundled_skills_dir.resolve():
            return
    except Exception as exc:
        logging.debug("failed to compare skills directories during seeding: %s", exc)
        return
    if not bundled_skills_dir.exists():
        return
    for child in sorted(bundled_skills_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "SKILL.md").exists():
            continue
        target = runtime_skills_dir / child.name
        if target.exists():
            continue
        shutil.copytree(child, target)


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--debug", action="store_true", help="Enable verbose debug logs")
    common.add_argument(
        "--dangerously-skip-permissions",
        action="store_true",
        help="Disable interactive shell permission prompts (unsafe).",
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
    init_parser.add_argument("--workspace-path", type=str, default="", help="Workspace root path")
    init_parser.add_argument("--base-url", type=str, default="", help="OpenAI-compatible base URL")
    init_parser.add_argument("--responses-endpoint", type=str, default="", help="Responses API endpoint override")
    init_parser.add_argument("--model-endpoint", type=str, default="", help="Model chat completions endpoint")
    init_parser.add_argument("--models-endpoint", type=str, default="", help="Model listing endpoint")
    init_parser.add_argument(
        "--endpoint-mode",
        type=str,
        choices=["auto", "responses", "chat"],
        default="",
        help="Preferred endpoint mode",
    )
    init_parser.add_argument("--api-key", type=str, default="", help="Model API key (writes to ~/.alphanus/.env)")
    init_parser.add_argument("--api-key-env", type=str, default="", help="Environment variable name for model API key")
    init_parser.add_argument("--search-provider", type=str, choices=["tavily", "brave"], default="", help="Search provider")
    init_parser.add_argument(
        "--theme",
        type=str,
        choices=list(BUILTIN_THEME_IDS) + sorted(THEME_ALIASES.keys()),
        default="",
        help="Theme id",
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Validate config, env, endpoint, and workspace readiness",
        parents=[common],
    )
    doctor_parser.add_argument("--json", action="store_true", help="Emit doctor report as JSON")
    return parser


def _load_runtime_config(app_paths: Any, args: argparse.Namespace) -> tuple[Dict[str, Any], list[str]]:
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
    if args.dangerously_skip_permissions:
        caps = config.setdefault("capabilities", {})
        caps["dangerously_skip_permissions"] = True
        caps["shell_require_confirmation"] = False
    config, normalization_warnings = normalize_config(config)
    config_warnings.extend(normalization_warnings)
    validate_endpoint_policy(config)
    runtime_cfg = config.setdefault("runtime", {})
    if isinstance(runtime_cfg, dict):
        runtime_cfg["state_root"] = str(Path(app_paths.state_root).resolve())
    return config, config_warnings


def _build_agent_runtime(app_paths: Any, config: Dict[str, Any], *, debug: bool) -> tuple[WorkspaceManager, LexicalMemory, SkillRuntime, Agent]:
    workspace_root = resolve_path(config["workspace"]["path"], app_paths.state_root)
    workspace = WorkspaceManager(workspace_root=workspace_root)
    memory_path = str((Path(app_paths.state_root).resolve() / "memory" / "events.jsonl").resolve())
    memory_cfg = config.get("memory", {})
    memory = LexicalMemory(
        storage_path=memory_path,
        min_score=float(memory_cfg.get("min_score_default", 0.3)),
        backup_revisions=int(memory_cfg.get("backup_revisions", 2)),
    )
    runtime_skills_dir = _resolve_runtime_skills_dir(app_paths)
    _seed_runtime_skills(runtime_skills_dir, Path(app_paths.bundled_skills_dir))
    runtime = SkillRuntime(
        skills_dir=str(runtime_skills_dir),
        workspace=workspace,
        memory=memory,
        config=config,
        debug=debug,
    )
    agent = Agent(config=config, skill_runtime=runtime, debug=debug)
    return workspace, memory, runtime, agent


def _prompt_with_default(label: str, default: str, *, hint: str = "") -> str:
    prompt = f"{label} [{default}]"
    if hint:
        prompt = f"{prompt}  {hint}"
    value = input(f"{prompt}: ").strip()
    return value or default


def _prompt_yes_no(label: str, *, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
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
        print(f"  {theme.accent(str(idx) + '.')} {value:<{value_width}} {theme.muted(description)}")
    default_index = next((idx for idx, (value, _desc) in enumerate(options, start=1) if value == default), 1)
    while True:
        raw = input(f"Choose option [{default_index}]: ").strip().lower()
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


def _apply_reset_scope(base: Dict[str, Any], *, section: str) -> Dict[str, Any]:
    if section == "all":
        return copy.deepcopy(DEFAULT_CONFIG)

    updated = copy.deepcopy(base)
    default_cfg = copy.deepcopy(DEFAULT_CONFIG)

    if _section_selected(section, "workspace"):
        updated["workspace"] = copy.deepcopy(default_cfg.get("workspace", {}))

    if _section_selected(section, "model"):
        current_agent = updated.get("agent", {}) if isinstance(updated.get("agent"), dict) else {}
        default_agent = default_cfg.get("agent", {}) if isinstance(default_cfg.get("agent"), dict) else {}
        for key in (
            "base_url",
            "responses_endpoint",
            "model_endpoint",
            "models_endpoint",
            "endpoint_mode",
            "api_key",
            "api_key_env",
            "auth_header_template",
        ):
            if key in default_agent:
                current_agent[key] = copy.deepcopy(default_agent[key])
        updated["agent"] = current_agent

    if _section_selected(section, "search"):
        updated["search"] = copy.deepcopy(default_cfg.get("search", {}))

    if _section_selected(section, "theme"):
        current_tui = updated.get("tui", {}) if isinstance(updated.get("tui"), dict) else {}
        default_tui = default_cfg.get("tui", {}) if isinstance(default_cfg.get("tui"), dict) else {}
        if "theme" in default_tui:
            current_tui["theme"] = copy.deepcopy(default_tui["theme"])
        updated["tui"] = current_tui

    return updated


def _ensure_dotenv_template(dotenv_path: Path) -> None:
    dotenv_path.parent.mkdir(parents=True, exist_ok=True)
    if dotenv_path.exists():
        return
    template = "\n".join(
        [
            "# Alphanus environment variables",
            "# Uncomment and set values as needed.",
            "# ALPHANUS_API_KEY=",
            "# TAVILY_API_KEY=",
            "# BRAVE_SEARCH_API_KEY=",
            "# ALPHANUS_AUTH_HEADER=Authorization: Bearer <token>",
            "",
        ]
    )
    dotenv_path.write_text(template, encoding="utf-8")


def _upsert_env_var(dotenv_path: Path, key: str, value: str) -> None:
    key = key.strip()
    if not key:
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


def _run_init(args: argparse.Namespace) -> int:
    theme = _CliTheme()
    section = str(getattr(args, "section", "all") or "all").strip().lower()
    if section not in INIT_SECTIONS:
        section = "all"
    reset_requested = bool(getattr(args, "reset", False))
    app_paths = get_app_paths()
    state_root = Path(app_paths.state_root)
    state_root.mkdir(parents=True, exist_ok=True)

    base: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
    existing_warnings: list[str] = []
    if app_paths.config_path.exists():
        try:
            existing = load_global_config(app_paths.config_path, warnings=existing_warnings)
            base = deep_merge(base, existing)
        except (OSError, ValueError):
            base = copy.deepcopy(DEFAULT_CONFIG)
    if reset_requested:
        base = _apply_reset_scope(base, section=section)

    workspace_default = str(base.get("workspace", {}).get("path", DEFAULT_CONFIG["workspace"]["path"]))
    base_url_default = str(base.get("agent", {}).get("base_url", DEFAULT_CONFIG["agent"]["base_url"]))
    model_endpoint_default = str(base.get("agent", {}).get("model_endpoint", DEFAULT_CONFIG["agent"]["model_endpoint"]))
    responses_endpoint_default = str(
        base.get("agent", {}).get("responses_endpoint", DEFAULT_CONFIG["agent"]["responses_endpoint"])
    )
    models_endpoint_default = str(base.get("agent", {}).get("models_endpoint", DEFAULT_CONFIG["agent"]["models_endpoint"]))
    endpoint_mode_default = str(base.get("agent", {}).get("endpoint_mode", DEFAULT_CONFIG["agent"]["endpoint_mode"]))
    api_key_ref_default = str(base.get("agent", {}).get("api_key", DEFAULT_CONFIG["agent"]["api_key"]))
    api_key_env_default = str(base.get("agent", {}).get("api_key_env", DEFAULT_CONFIG["agent"]["api_key_env"]))
    search_provider_default = str(base.get("search", {}).get("provider", DEFAULT_CONFIG["search"]["provider"]))
    theme_default = str(base.get("tui", {}).get("theme", DEFAULT_THEME_ID))
    theme_default, _ = normalize_theme_id(theme_default, default=DEFAULT_THEME_ID)

    workspace_path = workspace_default
    base_url = base_url_default
    model_endpoint = model_endpoint_default
    responses_endpoint = responses_endpoint_default
    models_endpoint = models_endpoint_default
    endpoint_mode = endpoint_mode_default
    api_key_env = api_key_env_default
    api_key_ref = api_key_ref_default
    api_key_value = ""
    search_provider = search_provider_default
    ui_theme = theme_default

    if args.non_interactive:
        print(theme.brand(" ALPHANUS INIT "))
        print(theme.muted(f"Applying non-interactive setup profile ({section})."))
        if _section_selected(section, "workspace"):
            workspace_path = str(getattr(args, "workspace_path", "") or "").strip() or workspace_default
        if _section_selected(section, "model"):
            base_url = str(getattr(args, "base_url", "") or "").strip() or base_url_default
            model_endpoint = str(getattr(args, "model_endpoint", "") or "").strip() or model_endpoint_default
            responses_endpoint = (
                str(getattr(args, "responses_endpoint", "") or "").strip() or responses_endpoint_default
            )
            models_endpoint = str(getattr(args, "models_endpoint", "") or "").strip() or models_endpoint_default
            endpoint_mode = str(getattr(args, "endpoint_mode", "") or "").strip() or endpoint_mode_default
            api_key_env = str(getattr(args, "api_key_env", "") or "").strip() or api_key_env_default
            api_key_ref = f"env:{api_key_env}"
            api_key_value = str(getattr(args, "api_key", "") or "").strip()
        if _section_selected(section, "search"):
            search_provider = str(getattr(args, "search_provider", "") or "").strip() or search_provider_default
        if _section_selected(section, "theme"):
            requested_theme = str(getattr(args, "theme", "") or "").strip() or theme_default
            ui_theme, _ = normalize_theme_id(requested_theme, default=theme_default)
    else:
        steps = [name for name in ("workspace", "model", "search", "theme") if _section_selected(section, name)]
        total_steps = max(len(steps), 1)
        step_index = 1
        print(theme.brand(" ALPHANUS SETUP "))
        print(theme.muted(f"Wizard scope: {section}. Press Enter to keep defaults."))
        print(theme.muted("We'll configure runtime state, model connectivity, search, and display theme defaults."))
        print(f"{theme.label('State root:')} {state_root}")
        print("")
        if _section_selected(section, "workspace"):
            print(theme.accent(f"Step {step_index}/{total_steps}: Workspace"))
            workspace_path = _prompt_with_default(
                "Workspace path",
                workspace_default,
                hint=theme.muted("where project files live"),
            )
            step_index += 1
            print("")
        if _section_selected(section, "model"):
            print(theme.accent(f"Step {step_index}/{total_steps}: Model endpoint"))
            use_local = _prompt_yes_no(
                "Use the local OpenAI-compatible endpoint preset (127.0.0.1:8080)?",
                default=base_url_default == str(DEFAULT_CONFIG["agent"]["base_url"]),
            )
            if use_local:
                base_url = str(DEFAULT_CONFIG["agent"]["base_url"])
                model_endpoint = str(DEFAULT_CONFIG["agent"]["model_endpoint"])
                responses_endpoint = str(DEFAULT_CONFIG["agent"]["responses_endpoint"])
                models_endpoint = str(DEFAULT_CONFIG["agent"]["models_endpoint"])
                endpoint_mode = str(DEFAULT_CONFIG["agent"]["endpoint_mode"])
                print(theme.muted("Applied local preset for /v1/responses, /v1/chat/completions, and /v1/models."))
            else:
                base_url = _prompt_with_default(
                    "Base URL",
                    base_url_default,
                    hint=theme.muted("provider root, e.g. https://api.openai.com"),
                )
                model_endpoint = _prompt_with_default(
                    "Chat endpoint",
                    model_endpoint_default,
                    hint=theme.muted("chat completions endpoint"),
                )
                responses_endpoint = _prompt_with_default(
                    "Responses endpoint",
                    responses_endpoint_default,
                    hint=theme.muted("responses endpoint"),
                )
                models_endpoint = _prompt_with_default(
                    "Models endpoint",
                    models_endpoint_default,
                    hint=theme.muted("model catalog endpoint"),
                )
                endpoint_mode = _prompt_choice(
                    theme,
                    "Endpoint mode:",
                    [
                        ("auto", "responses first with fallback to chat"),
                        ("responses", "force /v1/responses"),
                        ("chat", "force /v1/chat/completions"),
                    ],
                    default=endpoint_mode_default if endpoint_mode_default in {"auto", "responses", "chat"} else "auto",
                )
            api_key_env = _prompt_with_default(
                "API key env var",
                api_key_env_default,
                hint=theme.muted("where init stores the provider key"),
            )
            api_key_ref = f"env:{api_key_env.strip() or 'ALPHANUS_API_KEY'}"
            api_key_value = getpass.getpass(
                "API key (stored in ~/.alphanus/.env; leave blank to keep current): "
            ).strip()
            step_index += 1
            print("")
        if _section_selected(section, "search"):
            print(theme.accent(f"Step {step_index}/{total_steps}: Search provider"))
            search_provider = _prompt_choice(
                theme,
                "Choose a provider:",
                [
                    ("tavily", "recommended default, requires TAVILY_API_KEY"),
                    ("brave", "alternative, requires BRAVE_SEARCH_API_KEY"),
                ],
                default=search_provider_default,
            )
            step_index += 1
            print("")
        if _section_selected(section, "theme"):
            print(theme.accent(f"Step {step_index}/{total_steps}: Theme"))
            theme_options = [(name, theme_spec(name).description) for name in BUILTIN_THEME_IDS]
            selected_theme = _prompt_choice(
                theme,
                "Choose a UI theme:",
                theme_options,
                default=theme_default,
            )
            ui_theme, _ = normalize_theme_id(selected_theme, default=theme_default)
            print("")

    updates: Dict[str, Any] = {}
    if _section_selected(section, "workspace"):
        updates["workspace"] = {"path": workspace_path}
    if _section_selected(section, "model"):
        updates["agent"] = {
            "base_url": base_url,
            "model_endpoint": model_endpoint,
            "responses_endpoint": responses_endpoint,
            "models_endpoint": models_endpoint,
            "endpoint_mode": endpoint_mode,
            "api_key": api_key_ref,
            "api_key_env": api_key_env,
        }
    if _section_selected(section, "search"):
        updates["search"] = {"provider": search_provider}
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
        print(theme.accent("Review"))
        print(f"  {theme.label('Workspace:')} {normalized['workspace']['path']}")
        print(f"  {theme.label('Base URL:')} {normalized['agent']['base_url']}")
        print(f"  {theme.label('Responses endpoint:')} {normalized['agent']['responses_endpoint']}")
        print(f"  {theme.label('Model endpoint:')} {normalized['agent']['model_endpoint']}")
        print(f"  {theme.label('Models endpoint:')} {normalized['agent']['models_endpoint']}")
        print(f"  {theme.label('Endpoint mode:')} {normalized['agent']['endpoint_mode']}")
        print(f"  {theme.label('API key ref:')} {normalized['agent']['api_key']}")
        print(f"  {theme.label('Search provider:')} {normalized['search']['provider']}")
        print(f"  {theme.label('Theme:')} {normalized['tui']['theme']}")
        print(f"  {theme.label('Secrets file:')} {app_paths.dotenv_path}")
        if not _prompt_yes_no("Write these settings now?", default=True):
            print(theme.warn("Setup cancelled. No files were written."))
            return 1

    cleaned = config_for_editor_view(normalized)
    app_paths.config_path.parent.mkdir(parents=True, exist_ok=True)
    app_paths.config_path.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
    _ensure_dotenv_template(app_paths.dotenv_path)
    if _section_selected(section, "model") and api_key_value:
        _upsert_env_var(app_paths.dotenv_path, normalized["agent"]["api_key_env"], api_key_value)

    for warning in existing_warnings + warnings:
        print(f"{theme.warn('config warning:')} {warning}")
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


def _run_doctor(args: argparse.Namespace) -> int:
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

    workspace, memory, _runtime, agent = _build_agent_runtime(app_paths, config, debug=args.debug)
    report = agent.doctor_report()
    if args.json:
        payload = dict(report) if isinstance(report, dict) else {"doctor": report}
        payload["ok"] = True
        payload["config_warnings"] = warnings
        print(json.dumps(payload, indent=2))
    else:
        for warning in warnings:
            print(f"{theme.warn('config warning:')} {warning}")
        agent_status = report.get("agent", {}) if isinstance(report, dict) else {}
        workspace_status = report.get("workspace", {}) if isinstance(report, dict) else {}
        search_status = report.get("search", {}) if isinstance(report, dict) else {}
        endpoint_error = str(agent_status.get("endpoint_policy_error") or "").strip()
        endpoint_ok = not endpoint_error
        model_ok = bool(agent_status.get("ready"))
        workspace_ok = bool(workspace_status.get("exists")) and bool(workspace_status.get("writable"))
        search_ok = bool(search_status.get("ready"))

        print("")
        print(theme.brand(" ALPHANUS DOCTOR "))
        print(theme.muted("Running health checks for config, model endpoint, workspace, and search integration."))
        print("")
        print(f"  {theme.label('Config:')} {theme.path(str(app_paths.config_path))}")
        print(f"  {theme.label('Env:')} {theme.path(str(app_paths.dotenv_path))}")
        print(f"  {theme.label('Sessions:')} {theme.path(str((Path(app_paths.state_root) / 'sessions').resolve()))}")
        print(f"  {theme.label('Memory:')} {theme.path(str(memory.storage_path))}")
        print(f"  {theme.label('Workspace:')} {theme.path(str(workspace.workspace_root))}")
        print("\n")
        print(theme.accent("Checks"))
        endpoint_detail = "ok" if endpoint_ok else endpoint_error
        endpoint_state = theme.ok("[OK]") if endpoint_ok else theme.error("[FAIL]")
        model_state = theme.ok("[OK]") if model_ok else theme.warn("[WAIT]")
        workspace_state = theme.ok("[OK]") if workspace_ok else theme.error("[FAIL]")
        search_state = theme.ok("[OK]") if search_ok else theme.warn("[WAIT]")
        search_detail = str(search_status.get("reason") or "").strip() or "ok"
        endpoint_suffix = f"  {endpoint_detail}" if endpoint_detail != "ok" else ""
        search_suffix = f"  {search_detail}" if search_detail != "ok" else ""
        print(f"  endpoint policy:   {endpoint_state}{endpoint_suffix}")
        print(f"  model readiness:   {model_state}")
        print(f"  workspace access:  {workspace_state}")
        print(f"  search provider:   {search_state}{search_suffix}")

    failures = []
    agent_status = report.get("agent", {}) if isinstance(report, dict) else {}
    workspace_status = report.get("workspace", {}) if isinstance(report, dict) else {}
    search_status = report.get("search", {}) if isinstance(report, dict) else {}
    if str(agent_status.get("endpoint_policy_error") or "").strip():
        failures.append("endpoint-policy")
    if not bool(agent_status.get("ready")):
        failures.append("model-readiness")
    if not bool(workspace_status.get("exists")) or not bool(workspace_status.get("writable")):
        failures.append("workspace")
    if not bool(search_status.get("ready")):
        failures.append("search")
    if not args.json:
        if failures:
            print("")
            print(f"{theme.error('Doctor result: FAIL')} ({', '.join(failures)})")
            print(theme.muted("Fix failing checks, then rerun `uv run alphanus doctor`."))
        else:
            print("")
            print(theme.ok("Doctor result: PASS"))
        print("")
    return 1 if failures else 0


def _run_tui(args: argparse.Namespace) -> int:
    app_paths = get_app_paths()
    try:
        config, config_warnings = _load_runtime_config(app_paths, args)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"startup failed: {exc}")
        return 2

    logger = configure_logging(config)
    for warning in config_warnings:
        logger.warning(f"config: {warning}")

    _workspace, memory, _runtime, agent = _build_agent_runtime(app_paths, config, debug=args.debug)
    if args.debug:
        logger.info(f"debug HTTP log: {config['agent']['debug_log_path']}")

    if not config.get("agent", {}).get("tls_verify", True):
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
    return _run_tui(args)
