import argparse
import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.config_model import ConfigSchema, config_schema
from core.configuration import load_global_config, normalize_config, validate_endpoint_policy


def _as_object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _load_runtime_config(app_paths: Any, args: argparse.Namespace) -> tuple[ConfigSchema, list[str]]:
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
    logging_cfg = config.setdefault("logging", {})
    if isinstance(logging_cfg, dict):
        configured_log = Path(str(logging_cfg.get("path") or "logs/runtime.jsonl")).expanduser()
        if not configured_log.is_absolute():
            configured_log = Path(app_paths.state_root).resolve() / configured_log
        logging_cfg["path"] = str(configured_log.resolve())
    return config_schema(config), config_warnings


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


def resolve_project_root(config: ConfigSchema | Mapping[str, Any], *, override: str = "", cwd: Path | None = None) -> Path:
    config = config_schema(config)
    if override.strip():
        return Path(os.path.expanduser(override.strip())).resolve()
    start = (cwd or Path.cwd()).resolve()
    strategy = config.project.root_strategy.strip().lower()
    if strategy == "git-or-cwd":
        return _git_root_for_cwd(start) or start
    return start


def _build_agent_runtime(app_paths: Any, config: ConfigSchema, *, debug: bool) -> tuple[Any, Any, Any, Any]:
    from agent.core import Agent
    from core.memory import LexicalMemory
    from core.project import ProjectRuntime
    from skills.runtime import SkillRuntime

    project_root = resolve_project_root(config, override=str(getattr(config, "_project_root_override", "")))
    project = ProjectRuntime(
        project_root=str(project_root),
        permission_mode=config.permissions.mode,
        network_access=config.permissions.network,
        sandbox_backend=config.sandbox.backend,
        sandbox_fail_closed=config.sandbox.fail_closed,
    )
    memory_path = str((Path(app_paths.state_root).resolve() / "memory" / "events.jsonl").resolve())
    memory = LexicalMemory(
        storage_path=memory_path,
        min_score=config.memory.min_score_default,
        backup_revisions=config.memory.backup_revisions,
    )
    user_skills_dir = getattr(app_paths, "user_skills_dir", None)
    runtime_skills_dir = (
        Path(user_skills_dir).resolve() if user_skills_dir is not None else (Path(app_paths.state_root).resolve() / "skills").resolve()
    )
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


def _close_memory(memory: Any) -> None:
    if memory is None:
        return
    try:
        memory.flush()
        memory.close()
    except Exception:
        pass
