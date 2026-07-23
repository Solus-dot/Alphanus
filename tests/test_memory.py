from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from core.memory import LexicalMemory
from core.project import ProjectRuntime
from skills.runtime import SkillContext, SkillRuntime


def _memory_runtime(tmp_path: Path) -> tuple[SkillRuntime, str]:
    repo_root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = SkillRuntime(
        skills_dir=str(tmp_path / "user-skills"),
        bundled_skills_dir=str(repo_root / "bundled-skills"),
        project=ProjectRuntime(str(workspace)),
        memory=LexicalMemory(str(tmp_path / "memory.db")),
        config={"retrieval": {"enabled": False}},
    )
    return runtime, str(workspace)


def test_add_search_forget_is_transactional(tmp_path: Path) -> None:
    memory = LexicalMemory(str(tmp_path / "memory.db"), persist_access_updates=True)
    item = memory.add_memory("I like coffee", memory_type="preference")
    hit = memory.search("coffee", top_k=1, min_score=0)[0]
    assert hit["id"] == item["id"]
    assert memory.forget(item["id"])
    assert memory.search("coffee", min_score=0) == []


def test_store_uses_schema_wal_and_bounded_candidates(tmp_path: Path) -> None:
    memory = LexicalMemory(str(tmp_path / "memory.db"))
    for index in range(2500):
        memory.add_memory(f"record {index} alpha")
    assert len(memory.search("alpha", top_k=3, min_score=0)) == 3
    connection = sqlite3.connect(memory.storage_path)
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 2


def test_legacy_unversioned_memory_is_rejected(tmp_path: Path) -> None:
    legacy = tmp_path / "events.jsonl"
    legacy.write_text('{"id":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="does not migrate"):
        LexicalMemory(str(legacy))


def test_memory_skill_contract(tmp_path: Path) -> None:
    runtime, workspace = _memory_runtime(tmp_path)
    skill = runtime.get_skill("memory-rag")
    assert skill is not None
    context = SkillContext(user_input="remember", branch_labels=[], attachments=[], project_root=workspace, memory_hits=[])
    stored = runtime.execute_tool_call("store_memory", {"text": "Favorite editor is Neovim"}, [skill], context)
    recalled = runtime.execute_tool_call("recall_memory", {"query": "favorite editor", "top_k": 3}, [skill], context)
    assert stored["ok"] is True
    assert recalled["ok"] is True
    assert recalled["data"]["hits"][0]["text"] == "Favorite editor is Neovim"


def test_user_skill_cannot_execute_python(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skills = tmp_path / "skills"
    skill = skills / "unsafe"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: unsafe\ndescription: unsafe\n---\n", encoding="utf-8")
    (skill / "tools.py").write_text(
        "TOOL_SPECS={'unsafe_run': {'description':'x','parameters':{'type':'object'}}}\ndef execute(name,args,env): return {'ran': True}\n",
        encoding="utf-8",
    )
    (skill / "run.py").write_text("raise SystemExit('should never execute')\n", encoding="utf-8")
    runtime = SkillRuntime(
        skills_dir=str(skills),
        bundled_skills_dir=str(Path(__file__).resolve().parents[1] / "bundled-skills"),
        project=ProjectRuntime(str(workspace)),
        memory=LexicalMemory(str(tmp_path / "memory.db")),
        config={},
    )
    manifest = runtime.get_skill("unsafe")
    assert manifest is not None
    assert manifest.execution_allowed is False
    assert manifest in runtime.enabled_skills()
    assert runtime.tool_registration("unsafe_run") is None
    context = SkillContext(user_input="run", branch_labels=[], attachments=[], project_root=str(workspace), memory_hits=[])
    result = runtime.execute_tool_call("unsafe_run", {}, [manifest], context)
    assert result["ok"] is False
    assert result["error"]["code"] in {"E_POLICY", "E_UNSUPPORTED"}
