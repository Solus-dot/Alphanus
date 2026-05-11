from __future__ import annotations

from pathlib import Path

from agent.llm_client import LLMClient
from core.memory import LexicalMemory
from core.message_types import ChatMessage
from core.skills import SkillRuntime
from core.workspace import WorkspaceManager


def test_benchmark_memory_search(benchmark, tmp_path: Path) -> None:
    memory = LexicalMemory(str(tmp_path / "memory" / "events.jsonl"), autosave_every=10000, autosave_interval_s=0)
    for idx in range(300):
        memory.add_memory(f"project alpha note {idx} with provider and workspace details", memory_type="conversation")

    result = benchmark(lambda: memory.search("alpha provider workspace", top_k=5, min_score=0.1))

    assert result


def test_benchmark_workspace_search_code(benchmark, tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    for idx in range(80):
        (workspace / f"module_{idx}.py").write_text(f"def function_{idx}():\n    return 'needle {idx}'\n", encoding="utf-8")
    manager = WorkspaceManager(str(workspace), home_root=str(tmp_path))

    result = benchmark(lambda: manager.search_code("needle", max_results=25))

    assert result["count"] == 25


def test_benchmark_workspace_tree(benchmark, tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    for group in range(10):
        folder = workspace / f"pkg_{group}"
        folder.mkdir(parents=True)
        for idx in range(10):
            (folder / f"file_{idx}.txt").write_text("content\n", encoding="utf-8")
    manager = WorkspaceManager(str(workspace), home_root=str(tmp_path))

    tree = benchmark(lambda: manager.workspace_tree(max_depth=3))

    assert "pkg_0/" in tree


def test_benchmark_tool_schema_generation(benchmark, tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    for idx in range(10):
        skill_dir = skills_dir / f"skill-{idx}"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: skill-{idx}\ndescription: Skill {idx}\ntags: [alpha]\nallowed-tools: [run_skill]\n---\nUse this skill.\n",
            encoding="utf-8",
        )
    workspace = WorkspaceManager(str(tmp_path / "ws"), home_root=str(tmp_path))
    memory = LexicalMemory(str(tmp_path / "memory" / "events.jsonl"), autosave_every=10000, autosave_interval_s=0)
    runtime = SkillRuntime(str(skills_dir), workspace, memory)
    selected = runtime.enabled_skills()[:3]

    tools = benchmark(lambda: runtime.tools_for_turn(selected))

    assert tools


def test_benchmark_provider_payload_build(benchmark) -> None:
    client = LLMClient({"agent": {"endpoint_mode": "chat"}})
    messages: list[ChatMessage] = [{"role": "user", "content": "hello"}]

    payload = benchmark(lambda: client.build_payload(messages, thinking=True, tools=[]))

    assert payload["stream"] is True
