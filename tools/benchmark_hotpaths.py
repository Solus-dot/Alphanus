from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

from agent.classifier import TurnClassifier
from core.memory import LexicalMemory
from core.skills import SkillRuntime
from core.workspace import WorkspaceManager


class _NoopLLMClient:
    enable_structured_classification = True

    def call_with_retry(self, payload, stop_event, on_event, pass_id):
        raise RuntimeError("not used by benchmark")


def _run_timed(label: str, fn, *, loops: int) -> None:
    durations_ms: list[float] = []
    for _ in range(max(1, loops)):
        started = time.perf_counter()
        fn()
        durations_ms.append((time.perf_counter() - started) * 1000.0)
    p95 = max(durations_ms)
    if len(durations_ms) >= 2:
        p95 = statistics.quantiles(durations_ms, n=20)[18]
    print(
        f"{label:34} mean={statistics.mean(durations_ms):8.2f}ms "
        f"p95={p95:8.2f}ms "
        f"min={min(durations_ms):8.2f}ms max={max(durations_ms):8.2f}ms"
    )


def _benchmark_workspace_search(tmp_root: Path, loops: int) -> None:
    home = tmp_root / "home"
    ws = home / "ws"
    src = ws / "src"
    src.mkdir(parents=True, exist_ok=True)
    for idx in range(12):
        lines = []
        for line_no in range(250):
            if line_no % 7 == 0:
                lines.append(f"def greet_{idx}_{line_no}(name):")
            else:
                lines.append("    return name")
        (src / f"module_{idx}.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manager = WorkspaceManager(str(ws), home_root=str(home))

    def _search() -> None:
        result = manager.search_code(
            "greet_",
            path="src",
            glob="*.py",
            max_results=120,
            case_sensitive=True,
            fixed_strings=True,
            before_context=2,
            after_context=2,
        )
        assert result["count"] > 0

    _run_timed("workspace.search_code (rg+context)", _search, loops=loops)


def _benchmark_skill_resolution(tmp_root: Path, loops: int) -> None:
    home = tmp_root / "home2"
    ws = home / "ws"
    skills_dir = tmp_root / "skills"
    home.mkdir(parents=True, exist_ok=True)
    ws.mkdir(parents=True, exist_ok=True)
    skills_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(140):
        skill_id = f"build-helper-{idx}"
        root = skills_dir / skill_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "SKILL.md").write_text(
            "\n".join(
                [
                    "---",
                    f"name: build-helper-{idx}",
                    "description: benchmark skill",
                    "version: 1.0.0",
                    "aliases:",
                    f"  - helper-{idx}",
                    f"  - bh{idx}",
                    "---",
                    "Body",
                ]
            ),
            encoding="utf-8",
        )
    runtime = SkillRuntime(
        skills_dir=str(skills_dir),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_root / "mem.events.jsonl")),
        config={},
    )

    def _resolve() -> None:
        assert runtime.resolve_skill_reference("build-helper-3") is not None
        assert runtime.resolve_skill_reference("build-helper-11") is not None
        assert runtime.resolve_skill_reference("build-helper-11".upper()) is not None
        assert runtime.resolve_skill_reference("buildhelper42") is not None
        assert runtime.resolve_skill_reference("build-he") is None

    _run_timed("skills.resolve_skill_reference", _resolve, loops=loops * 4)


def _benchmark_classifier_path_scan(tmp_root: Path, loops: int) -> None:
    home = tmp_root / "home3"
    ws = home / "ws"
    skills_dir = tmp_root / "skills2"
    home.mkdir(parents=True, exist_ok=True)
    ws.mkdir(parents=True, exist_ok=True)
    skills_dir.mkdir(parents=True, exist_ok=True)
    runtime = SkillRuntime(
        skills_dir=str(skills_dir),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_root / "mem2.events.jsonl")),
        config={},
    )
    classifier = TurnClassifier(config={}, skill_runtime=runtime, llm_client=_NoopLLMClient())
    external = str(home / "other project" / "repo")
    text = (
        f'Please update "{external}" and then inspect /tmp/logs/runtime.txt '
        "while ignoring https://example.com/docs and /Users/fake/workspaces."
    )

    def _scan() -> None:
        _ = classifier._explicit_path_outside_workspace(text)

    _run_timed("classifier.external_path_scan", _scan, loops=loops * 12)


def _benchmark_memory_io(tmp_root: Path, loops: int) -> None:
    events_path = tmp_root / "memory" / "events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    mem = LexicalMemory(storage_path=str(events_path), autosave_every=10000, backup_revisions=0)
    for idx in range(900):
        mem.add_memory(f"event {idx} has token-{idx % 31}", memory_type="bench", metadata={"i": idx})
    mem.flush()

    def _load() -> None:
        loaded = LexicalMemory(storage_path=str(events_path), autosave_every=10000, backup_revisions=0)
        assert loaded.stats()["count"] == 900

    _run_timed("memory.load_jsonl", _load, loops=loops)

    def _save() -> None:
        target = LexicalMemory(storage_path=str(events_path), autosave_every=10000, backup_revisions=0)
        target.add_memory("new benchmark memory", memory_type="bench")
        target.flush()

    _run_timed("memory.save_jsonl", _save, loops=loops)


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbenchmarks for Alphanus runtime hot paths")
    parser.add_argument("--loops", type=int, default=5, help="Benchmark iterations per case (default: 5)")
    args = parser.parse_args()
    loops = max(1, int(args.loops))
    with tempfile.TemporaryDirectory(prefix="alphanus-bench-") as tmp:
        tmp_root = Path(tmp)
        print(f"bench root: {tmp_root}")
        _benchmark_workspace_search(tmp_root, loops)
        _benchmark_skill_resolution(tmp_root, loops)
        _benchmark_classifier_path_scan(tmp_root, loops)
        _benchmark_memory_io(tmp_root, loops)


if __name__ == "__main__":
    main()
