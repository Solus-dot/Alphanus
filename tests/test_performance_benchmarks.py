from __future__ import annotations

from pathlib import Path

from agent.llm_client import LLMClient
from core.memory import LexicalMemory
from core.message_types import ChatMessage
from core.project import ProjectRuntime
from skills.runtime import SkillRuntime


def test_benchmark_memory_search(benchmark, tmp_path: Path) -> None:
    memory = LexicalMemory(str(tmp_path / "memory" / "events.jsonl"), autosave_every=10000, autosave_interval_s=0)
    for idx in range(300):
        memory.add_memory(f"project alpha note {idx} with provider and project details", memory_type="conversation")

    result = benchmark(lambda: memory.search("alpha provider project", top_k=5, min_score=0.1))

    assert result


def test_benchmark_project_search_code(benchmark, tmp_path: Path) -> None:
    project = tmp_path / "ws"
    project.mkdir()
    for idx in range(80):
        (project / f"module_{idx}.py").write_text(f"def function_{idx}():\n    return 'needle {idx}'\n", encoding="utf-8")
    manager = ProjectRuntime(str(project))

    result = benchmark(lambda: manager.search_code("needle", max_results=25))

    assert result["count"] == 25


def test_benchmark_project_tree(benchmark, tmp_path: Path) -> None:
    project = tmp_path / "ws"
    for group in range(10):
        folder = project / f"pkg_{group}"
        folder.mkdir(parents=True)
        for idx in range(10):
            (folder / f"file_{idx}.txt").write_text("content\n", encoding="utf-8")
    manager = ProjectRuntime(str(project))

    tree = benchmark(lambda: manager.project_tree(max_depth=3))

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
    project_root = tmp_path / "ws"
    project_root.mkdir()
    project = ProjectRuntime(str(project_root))
    memory = LexicalMemory(str(tmp_path / "memory" / "events.jsonl"), autosave_every=10000, autosave_interval_s=0)
    runtime = SkillRuntime(str(skills_dir), project, memory)
    selected = runtime.enabled_skills()[:3]

    tools = benchmark(lambda: runtime.tools_for_turn(selected))

    assert tools


def test_benchmark_provider_payload_build(benchmark) -> None:
    client = LLMClient({"agent": {"endpoint_mode": "chat"}})
    messages: list[ChatMessage] = [{"role": "user", "content": "hello"}]

    payload = benchmark(lambda: client.build_payload(messages, thinking=True, tools=[]))

    assert payload["stream"] is True
