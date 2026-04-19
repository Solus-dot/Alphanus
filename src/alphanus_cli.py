import argparse
import logging
import shutil
from pathlib import Path

from alphanus_paths import get_app_paths
from agent.core import Agent
from agent.telemetry import configure_logging
from core.configuration import (
    load_dotenv,
    load_or_create_global_config,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Alphanus")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logs")
    parser.add_argument(
        "--dangerously-skip-permissions",
        action="store_true",
        help="Disable interactive shell permission prompts (unsafe).",
    )
    args = parser.parse_args()

    app_paths = get_app_paths()
    load_dotenv(app_paths.dotenv_path)
    config_path = app_paths.config_path
    config_warnings: list[str] = []
    config = load_or_create_global_config(config_path, warnings=config_warnings)
    if args.debug:
        debug_dir = app_paths.app_root / "logs"
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
    logger = configure_logging(config)
    for warning in config_warnings:
        logger.warning(f"config: {warning}")

    workspace_root = resolve_path(config["workspace"]["path"], app_paths.app_root)

    workspace = WorkspaceManager(workspace_root=workspace_root)
    memory_path = str((Path(workspace.workspace_root).resolve() / ".alphanus" / "memory" / "events.jsonl").resolve())
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
        debug=args.debug,
    )

    agent = Agent(config=config, skill_runtime=runtime, debug=args.debug)
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
