from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

from dotenv import load_dotenv

from agent.core import Agent
from core.memory import VectorMemory
from core.skills import SkillRuntime
from core.workspace import WorkspaceManager
from tui.interface import AlphanusTUI

SCHEMA_VERSION = "1.0.0"


DEFAULT_CONFIG: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "agent": {
        "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
        "models_endpoint": "http://127.0.0.1:8080/v1/models",
        "request_timeout_s": 180,
        "readiness_timeout_s": 30,
        "readiness_poll_s": 0.5,
        "enable_thinking": True,
        "auth_header": "",
        "tls_verify": True,
        "ca_bundle_path": "",
        "allow_cross_host_endpoints": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": None,
        "context_budget_max_tokens": 1024,
        "max_action_depth": 10,
    },
    "workspace": {
        "path": "~/Desktop/Alphanus-Workspace",
    },
    "memory": {
        "path": "./memories/memory.pkl",
    },
    "context": {
        "context_limit": 8192,
        "keep_last_n": 10,
        "safety_margin": 500,
    },
    "capabilities": {
        "shell_require_confirmation": True,
        "email_enabled": False,
        "email_imap_server": "imap.gmail.com",
    },
    "whatsapp": {
        "enabled": False,
    },
}


def deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_or_create_global_config(path: Path) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        return dict(DEFAULT_CONFIG)

    raw = json.loads(path.read_text(encoding="utf-8"))
    schema = raw.get("schema_version", "1.0.0")
    if int(schema.split(".", 1)[0]) != int(SCHEMA_VERSION.split(".", 1)[0]):
        raise ValueError(f"Unsupported config schema_version: {schema}")

    merged = deep_merge(DEFAULT_CONFIG, raw)
    merged["schema_version"] = schema
    return merged


def resolve_path(path_str: str, root: Path) -> str:
    path = Path(os.path.expanduser(path_str))
    if not path.is_absolute():
        path = (root / path).resolve()
    return str(path)


def validate_endpoint_policy(config: Dict[str, Any]) -> None:
    agent_cfg = config.get("agent", {})
    model_endpoint = agent_cfg.get("model_endpoint", "")
    models_endpoint = agent_cfg.get("models_endpoint", "")
    allow_cross = bool(agent_cfg.get("allow_cross_host_endpoints", False))

    host_a = urlparse(model_endpoint).netloc
    host_b = urlparse(models_endpoint).netloc
    if host_a != host_b and not allow_cross:
        raise ValueError("agent.model_endpoint and agent.models_endpoint must share host")


def main() -> int:
    parser = argparse.ArgumentParser(description="Alphanus")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logs")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")

    config_path = project_root / "config" / "global_config.json"
    config = load_or_create_global_config(config_path)

    validate_endpoint_policy(config)

    workspace_root = resolve_path(config["workspace"]["path"], project_root)
    memory_path = resolve_path(config["memory"]["path"], project_root)

    workspace = WorkspaceManager(workspace_root=workspace_root)
    memory = VectorMemory(storage_path=memory_path)
    runtime = SkillRuntime(
        skills_dir=str(project_root / "skills"),
        workspace=workspace,
        memory=memory,
        config=config,
        debug=args.debug,
    )

    agent = Agent(config=config, skill_runtime=runtime, debug=args.debug)

    if not config.get("agent", {}).get("tls_verify", True):
        print("[warning] TLS verification is disabled (agent.tls_verify=false)")

    # Readiness is validated before first generation too; this startup check
    # keeps failure visible early while still letting TUI boot.
    if not agent.ensure_ready():
        print(f"[warning] Model endpoint not ready at {agent.models_endpoint}; you can still open the TUI.")

    app = AlphanusTUI(agent=agent, debug=args.debug)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
