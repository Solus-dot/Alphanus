from __future__ import annotations

import argparse
import copy
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
        "max_tokens": None,
        "context_budget_max_tokens": 1024,
        "max_action_depth": 10,
        "max_tool_result_chars": 12000,
        "max_reasoning_chars": 20000,
        "compact_tool_results_in_history": False,
        "compact_tool_result_tools": [],
    },
    "workspace": {
        "path": "~/Desktop/Alphanus-Workspace",
    },
    "memory": {
        "path": "./memories/memory.pkl",
        "embedding_backend": "hash",
        "model_name": "all-MiniLM-L6-v2",
        "eager_load_encoder": False,
    },
    "context": {
        "context_limit": 8192,
        "keep_last_n": 10,
        "safety_margin": 500,
    },
    "capabilities": {
        "shell_require_confirmation": True,
        "dangerously_skip_permissions": False,
        "email_enabled": False,
        "email_imap_server": "imap.gmail.com",
    },
    "skills": {
        "selection_mode": "all_enabled",
        "max_active_skills": 6,
        "strict_capability_policy": False
    },
    "tui": {
        "chat_log_max_lines": 5000,
        "tree_compaction": {
            "enabled": True,
            "inactive_assistant_char_limit": 12000,
            "inactive_tool_argument_char_limit": 5000,
            "inactive_tool_content_char_limit": 8000,
        },
    },
}


def deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_or_create_global_config(path: Path) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        return copy.deepcopy(DEFAULT_CONFIG)

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
    parser.add_argument(
        "--dangerously-skip-permissions",
        action="store_true",
        help="Disable interactive shell permission prompts (unsafe).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")

    config_path = project_root / "config" / "global_config.json"
    config = load_or_create_global_config(config_path)
    if args.dangerously_skip_permissions:
        caps = config.setdefault("capabilities", {})
        caps["dangerously_skip_permissions"] = True
        caps["shell_require_confirmation"] = False

    validate_endpoint_policy(config)

    workspace_root = resolve_path(config["workspace"]["path"], project_root)
    memory_path = resolve_path(config["memory"]["path"], project_root)

    workspace = WorkspaceManager(workspace_root=workspace_root)
    memory_cfg = config.get("memory", {})
    memory = VectorMemory(
        storage_path=memory_path,
        model_name=str(memory_cfg.get("model_name", "all-MiniLM-L6-v2")),
        embedding_backend=str(memory_cfg.get("embedding_backend", "hash")),
        eager_load_encoder=bool(memory_cfg.get("eager_load_encoder", False)),
    )
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
    print(f"[info] waiting for endpoint {agent.models_endpoint} handshake...")
    if not agent.ensure_ready():
        print(f"[warning] Model endpoint not ready at {agent.models_endpoint}; you can still open the TUI.")
    else:
        print(f"[info] endpoint {agent.models_endpoint} handshake complete.")

    app = AlphanusTUI(agent=agent, debug=args.debug)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
