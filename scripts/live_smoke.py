from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.core import Agent
from core.memory import VectorMemory
from core.skills import SkillContext, SkillRuntime
from core.workspace import WorkspaceManager
from main import DEFAULT_CONFIG, deep_merge


@dataclass
class Scenario:
    name: str
    prompt: str


def _build_agent(root: Path, workspace_root: Path, memory_path: Path) -> tuple[Agent, SkillRuntime]:
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "workspace": {"path": str(workspace_root)},
            "memory": {
                "path": str(memory_path),
                "embedding_backend": "transformer",
                "model_name": "BAAI/bge-small-en-v1.5",
                "allow_model_download": True,
            },
            "capabilities": {
                "dangerously_skip_permissions": True,
                "shell_require_confirmation": False,
            },
            "skills": {
                "selection_mode": "all_enabled",
                "max_active_skills": 6,
            },
            "agent": {
                "request_timeout_s": 120,
                "readiness_timeout_s": 5,
                "readiness_poll_s": 0.5,
                "max_tokens": None,
                "context_budget_max_tokens": 2048,
            },
        },
    )
    workspace = WorkspaceManager(workspace_root=str(workspace_root), home_root=str(Path.home()))
    memory = VectorMemory(
        storage_path=str(memory_path),
        embedding_backend="transformer",
        model_name="BAAI/bge-small-en-v1.5",
        allow_model_download=True,
    )
    runtime = SkillRuntime(
        skills_dir=str(root / "skills"),
        workspace=workspace,
        memory=memory,
        config=config,
        debug=False,
    )
    return Agent(config=config, skill_runtime=runtime, debug=False), runtime


def _seed_state(runtime: SkillRuntime, workspace_root: Path) -> None:
    ctx = SkillContext(
        user_input="seed",
        branch_labels=[],
        attachments=[],
        workspace_root=str(workspace_root),
        memory_hits=[],
    )
    ws_skill = runtime.get_skill("workspace-ops")
    mem_skill = runtime.get_skill("memory-rag")
    if ws_skill:
        runtime.execute_tool_call(
            "create_file",
            {"filepath": "notes.txt", "content": "alpha\nbeta"},
            selected=[ws_skill],
            ctx=ctx,
        )
    if mem_skill:
        runtime.execute_tool_call(
            "store_memory",
            {"text": "User's favorite editor is Neovim."},
            selected=[mem_skill],
            ctx=ctx,
        )


def _base_scenarios() -> List[Scenario]:
    return [
        Scenario("normal", "What is 7 multiplied by 8? Answer directly."),
        Scenario("create_file", "Create a file named notes.txt in the workspace containing exactly: alpha"),
        Scenario("read_file", "Read notes.txt and tell me the contents."),
        Scenario("edit_file", "Edit notes.txt so it contains two lines: alpha and beta."),
        Scenario("list_files", "List the files in the workspace root."),
        Scenario("workspace_tree", "Show me the workspace tree."),
        Scenario("search_home_files", "Search my home files for notes.txt."),
        Scenario("store_memory", "Remember that my favorite editor is Neovim."),
        Scenario("recall_memory", "What is my favorite editor?"),
        Scenario("list_memories", "List my recent memories."),
        Scenario("get_memory_stats", "Show my memory stats."),
        Scenario("export_memories", "Export my memories to exported_memories.txt in the workspace."),
        Scenario("forget_memory", "Forget memory id 1."),
        Scenario("shell_command", "Run the shell command pwd and tell me the output."),
        Scenario("get_weather", "What is the weather in London right now?"),
    ]


def _browser_scenarios() -> List[Scenario]:
    return [
        Scenario("open_url", "Open https://example.com in the browser."),
        Scenario("play_youtube", "Play lofi hip hop on YouTube."),
    ]


def _summarize_events(events: List[Dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tool_calls = [
        {"name": evt.get("name"), "arguments": evt.get("arguments")}
        for evt in events
        if evt.get("type") == "tool_call"
    ]
    tool_results = [
        {
            "name": evt.get("name"),
            "ok": evt.get("result", {}).get("ok"),
            "error": evt.get("result", {}).get("error"),
            "data_keys": sorted(list((evt.get("result", {}).get("data") or {}).keys()))
            if isinstance(evt.get("result", {}).get("data"), dict)
            else None,
        }
        for evt in events
        if evt.get("type") == "tool_result"
    ]
    return tool_calls, tool_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live Alphanus smoke checks against a real model endpoint.")
    parser.add_argument("--include-browser", action="store_true", help="Include open_url and play_youtube scenarios.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full scenario report as JSON instead of a compact human summary.",
    )
    args = parser.parse_args()

    root = ROOT
    workspace_root = root / "tmp-live-smoke-workspace"
    memory_path = root / "tmp-live-smoke-memory.pkl"
    corrupt = root / "tmp-live-smoke-memory.pkl.corrupted"

    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    workspace_root.mkdir(parents=True)
    if memory_path.exists():
        memory_path.unlink()
    if corrupt.exists():
        corrupt.unlink()

    agent, runtime = _build_agent(root, workspace_root, memory_path)
    if not agent.ensure_ready():
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"Model endpoint not ready: {agent.models_endpoint}",
                },
                ensure_ascii=False,
            )
        )
        return 1

    _seed_state(runtime, workspace_root)

    scenarios = _base_scenarios()
    if args.include_browser:
        scenarios.extend(_browser_scenarios())

    results = []
    for scenario in scenarios:
        events: List[Dict[str, Any]] = []
        result = agent.run_turn(
            history_messages=[{"role": "user", "content": scenario.prompt}],
            user_input=scenario.prompt,
            thinking=True,
            on_event=events.append,
            confirm_shell=lambda _command: True,
        )
        tool_calls, tool_results = _summarize_events(events)
        results.append(
            {
                "scenario": scenario.name,
                "status": result.status,
                "error": result.error,
                "tool_calls": tool_calls,
                "tool_results": tool_results,
                "assistant_excerpt": result.content[:300],
            }
        )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        failures = 0
        for item in results:
            status = item["status"]
            if status != "done":
                failures += 1
            tool_names = ", ".join(call["name"] for call in item["tool_calls"]) or "-"
            print(f"[{status}] {item['scenario']}: tools={tool_names}")
            if item["error"]:
                print(f"  error: {item['error']}")
            for result in item["tool_results"]:
                if result["ok"] is False:
                    print(f"  tool {result['name']} failed: {result['error']}")
        print(f"\nCompleted {len(results)} scenarios, failures={failures}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
