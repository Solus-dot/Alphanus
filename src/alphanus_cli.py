import argparse
import copy
import json
import logging
import shutil
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
from core.memory import VectorMemory
from core.skills import SkillRuntime
from core.workspace import WorkspaceManager
from tui.interface import AlphanusTUI


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
    init_parser.add_argument("--non-interactive", action="store_true", help="Use defaults/flags without prompts")
    init_parser.add_argument("--workspace-path", type=str, default="", help="Workspace root path")
    init_parser.add_argument("--model-endpoint", type=str, default="", help="Model chat completions endpoint")
    init_parser.add_argument("--models-endpoint", type=str, default="", help="Model listing endpoint")
    init_parser.add_argument("--search-provider", type=str, choices=["tavily", "brave"], default="", help="Search provider")

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


def _build_agent_runtime(app_paths: Any, config: Dict[str, Any], *, debug: bool) -> tuple[WorkspaceManager, VectorMemory, SkillRuntime, Agent]:
    workspace_root = resolve_path(config["workspace"]["path"], app_paths.state_root)
    workspace = WorkspaceManager(workspace_root=workspace_root)
    memory_path = str((Path(app_paths.state_root).resolve() / "memory" / "events.jsonl").resolve())
    memory_cfg = config.get("memory", {})
    memory = VectorMemory(
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


def _prompt_with_default(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def _ensure_dotenv_template(dotenv_path: Path) -> None:
    dotenv_path.parent.mkdir(parents=True, exist_ok=True)
    if dotenv_path.exists():
        return
    template = "\n".join(
        [
            "# Alphanus environment variables",
            "# Uncomment and set values as needed.",
            "# TAVILY_API_KEY=",
            "# BRAVE_SEARCH_API_KEY=",
            "# ALPHANUS_AUTH_HEADER=Authorization: Bearer <token>",
            "",
        ]
    )
    dotenv_path.write_text(template, encoding="utf-8")


def _run_init(args: argparse.Namespace) -> int:
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

    if args.non_interactive:
        workspace_path = args.workspace_path.strip() or str(base.get("workspace", {}).get("path", DEFAULT_CONFIG["workspace"]["path"]))
        model_endpoint = args.model_endpoint.strip() or str(base.get("agent", {}).get("model_endpoint", DEFAULT_CONFIG["agent"]["model_endpoint"]))
        models_endpoint = args.models_endpoint.strip() or str(base.get("agent", {}).get("models_endpoint", DEFAULT_CONFIG["agent"]["models_endpoint"]))
        search_provider = args.search_provider.strip() or str(base.get("search", {}).get("provider", DEFAULT_CONFIG["search"]["provider"]))
    else:
        print("Alphanus setup")
        print(f"State root: {state_root}")
        workspace_path = _prompt_with_default(
            "Workspace path",
            str(base.get("workspace", {}).get("path", DEFAULT_CONFIG["workspace"]["path"])),
        )
        model_endpoint = _prompt_with_default(
            "Model endpoint",
            str(base.get("agent", {}).get("model_endpoint", DEFAULT_CONFIG["agent"]["model_endpoint"])),
        )
        models_endpoint = _prompt_with_default(
            "Models endpoint",
            str(base.get("agent", {}).get("models_endpoint", DEFAULT_CONFIG["agent"]["models_endpoint"])),
        )
        search_provider = _prompt_with_default(
            "Search provider (tavily/brave)",
            str(base.get("search", {}).get("provider", DEFAULT_CONFIG["search"]["provider"])),
        ).lower()

    updates = {
        "workspace": {"path": workspace_path},
        "agent": {"model_endpoint": model_endpoint, "models_endpoint": models_endpoint},
        "search": {"provider": search_provider},
    }
    merged = deep_merge(base, updates)
    try:
        normalized, warnings = normalize_config(merged)
        validate_endpoint_policy(normalized)
    except ValueError as exc:
        print(f"init failed: {exc}")
        return 2

    cleaned = config_for_editor_view(normalized)
    app_paths.config_path.parent.mkdir(parents=True, exist_ok=True)
    app_paths.config_path.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
    _ensure_dotenv_template(app_paths.dotenv_path)

    for warning in existing_warnings + warnings:
        print(f"config warning: {warning}")
    print(f"Initialized config: {app_paths.config_path}")
    print(f"Initialized env template: {app_paths.dotenv_path}")
    print("Next steps: `uv run alphanus doctor` then `uv run alphanus`")
    return 0


def _run_doctor(args: argparse.Namespace) -> int:
    app_paths = get_app_paths()
    try:
        config, warnings = _load_runtime_config(app_paths, args)
    except (FileNotFoundError, OSError, ValueError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc), "config_warnings": []}, indent=2))
            return 2
        print(f"doctor: config error: {exc}")
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
            print(f"config warning: {warning}")
        print("Alphanus doctor")
        print(f"  config: {app_paths.config_path}")
        print(f"  env: {app_paths.dotenv_path}")
        print(f"  sessions: {(Path(app_paths.state_root) / 'sessions').resolve()}")
        print(f"  memory: {memory.storage_path}")
        print(f"  workspace: {workspace.workspace_root}")
        agent_status = report.get("agent", {}) if isinstance(report, dict) else {}
        workspace_status = report.get("workspace", {}) if isinstance(report, dict) else {}
        search_status = report.get("search", {}) if isinstance(report, dict) else {}
        print(f"  endpoint policy: {agent_status.get('endpoint_policy_error') or 'ok'}")
        print(f"  model ready: {bool(agent_status.get('ready'))}")
        print(f"  workspace writable: {bool(workspace_status.get('writable'))}")
        print(f"  search ready: {bool(search_status.get('ready'))} {search_status.get('reason') or ''}".rstrip())

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
    memory_stats = memory.stats(probe_encoder=False)
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
