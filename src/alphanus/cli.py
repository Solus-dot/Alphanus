import argparse
import importlib
import importlib.machinery
import importlib.util
import platform
import sys
from pathlib import Path
from typing import Any

from alphanus.commands.backend import _run_runtime
from alphanus.commands.doctor import _run_doctor
from alphanus.commands.exec import _run_exec
from alphanus.commands.init import _run_init
from alphanus.commands.retrieval import _run_retrieval
from alphanus.console import INIT_SECTIONS
from core.backend_profiles import VALID_BACKEND_PROFILES
from core.endpoint_modes import ENDPOINT_MODES
from core.search_providers import SEARCH_FALLBACK_PROVIDERS, SEARCH_PROVIDERS


def _build_parser() -> argparse.ArgumentParser:
    project_root_help = "Project root for this run (defaults to enclosing git root, then cwd)"
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--debug", action="store_true", help="Enable verbose debug logs")
    common.add_argument("--project-root", type=str, default="", help=project_root_help)
    command_common = argparse.ArgumentParser(add_help=False)
    command_common.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help="Enable verbose debug logs")
    command_common.add_argument("--project-root", type=str, default=argparse.SUPPRESS, help=project_root_help)
    parser = argparse.ArgumentParser(description="Alphanus", parents=[common])
    subparsers = parser.add_subparsers(dest="command", metavar="{run,exec,init,doctor,retrieval}")

    subparsers.add_parser("run", help="Launch the TUI", parents=[command_common])

    runtime_parser = subparsers.add_parser("_runtime", help=argparse.SUPPRESS, parents=[command_common])
    runtime_parser.add_argument("--stdio", action="store_true", default=True, help=argparse.SUPPRESS)
    choices = getattr(subparsers, "_choices_actions", [])
    choices[:] = [choice for choice in choices if getattr(choice, "dest", "") != "_runtime"]

    exec_parser = subparsers.add_parser("exec", help="Run one turn using the versioned JSONL protocol", parents=[command_common])
    exec_parser.add_argument("prompt", nargs="?", default="", help="Prompt text; reads stdin when omitted")
    exec_parser.add_argument("--input", choices=("text", "jsonl"), default="text", help="Input framing")
    exec_parser.add_argument(
        "--approval-policy",
        choices=("deny", "allow-boundary"),
        default="deny",
        help="How non-interactive boundary approvals are handled",
    )
    exec_parser.add_argument("--no-thinking", action="store_true", help="Disable model reasoning mode")

    init_parser = subparsers.add_parser("init", help="Initialize ~/.alphanus config and env", parents=[command_common])
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
    init_parser.add_argument("--api-key", type=str, default="", help=argparse.SUPPRESS)
    init_parser.add_argument("--api-key-env", type=str, default="", help="Environment variable name for model API key")
    init_parser.add_argument(
        "--backend-api-key-env",
        type=str,
        default="",
        help="Optional extra environment variable to mirror the model API key for a local backend process",
    )
    init_parser.add_argument("--search-provider", type=str, choices=list(SEARCH_PROVIDERS), default="", help="Search provider")
    init_parser.add_argument(
        "--search-fallback-provider", type=str, choices=list(SEARCH_FALLBACK_PROVIDERS), default="", help="Search fallback provider"
    )
    init_parser.add_argument("--searxng-base-url", type=str, default="", help="SearXNG base URL")
    init_parser.add_argument("--tavily-api-key", type=str, default="", help=argparse.SUPPRESS)
    init_parser.add_argument("--tavily-api-key-env", type=str, default="", help="Environment variable name for Tavily API key")
    init_parser.add_argument("--theme", type=str, default="", help="Theme id")

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Validate config, env, endpoint, and project readiness",
        parents=[command_common],
    )
    doctor_parser.add_argument("--json", action="store_true", help="Emit doctor report as JSON")

    retrieval_parser = subparsers.add_parser("retrieval", help="Inspect or reset the local retrieval store", parents=[command_common])
    retrieval_subparsers = retrieval_parser.add_subparsers(dest="retrieval_command")
    retrieval_subparsers.add_parser("stats", help="Show retrieval store statistics")
    reset_parser = retrieval_subparsers.add_parser("reset", help="Delete the retrieval SQLite database")
    reset_parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    return parser


def _run_tui(args: Any) -> int:
    try:
        _alphanus_tui = importlib.import_module("_alphanus_tui")
    except ImportError as exc:
        spec = importlib.machinery.PathFinder.find_spec("_alphanus_tui", [str(Path(__file__).resolve().parents[1])])
        if spec is not None and spec.loader is not None:
            _alphanus_tui = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_alphanus_tui)
        else:
            print(
                "Ratatui frontend is unavailable for this installation. "
                "Install a supported Alphanus wheel or rebuild from source with Rust/Cargo.",
                file=sys.stderr,
            )
            if bool(getattr(args, "debug", False)):
                print(f"frontend import failed: {exc}", file=sys.stderr)
            return 2
    project_root = str(getattr(args, "project_root", "") or "").strip() or None
    try:
        return int(_alphanus_tui.run(sys.executable, project_root, bool(getattr(args, "debug", False))))
    except Exception as exc:
        print(f"Ratatui frontend failed: {exc}", file=sys.stderr)
        return 2


def main() -> int:
    if platform.system().lower() not in {"darwin", "linux"}:
        print("Alphanus v1 supports macOS and Linux only.", file=sys.stderr)
        return 2
    parser = _build_parser()
    args = parser.parse_args()

    command = str(getattr(args, "command", "") or "")
    if command == "init":
        return _run_init(args)
    if command == "doctor":
        return _run_doctor(args)
    if command == "retrieval":
        return _run_retrieval(args)
    if command == "exec":
        return _run_exec(args)
    if command == "_runtime":
        return _run_runtime(args)
    return _run_tui(args)


if __name__ == "__main__":
    raise SystemExit(main())
