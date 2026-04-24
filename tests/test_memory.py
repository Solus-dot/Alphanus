from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from core.memory import LexicalMemory
from core.skills import SkillContext, SkillRuntime
from core.workspace import WorkspaceManager


def _memory_runtime(tmp_path: Path) -> tuple[SkillRuntime, str]:
    repo_root = Path(__file__).resolve().parents[1]
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    runtime = SkillRuntime(
        skills_dir=str(repo_root / "skills"),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )
    return runtime, str(ws)


def test_add_search_forget(tmp_path: Path):
    mem = LexicalMemory(storage_path=str(tmp_path / "mem.pkl"))

    item = mem.add_memory("I like coffee", memory_type="preference")
    hits = mem.search("coffee", top_k=3, min_score=0.0)

    assert item["id"] >= 1
    assert hits
    assert hits[0]["id"] == item["id"]
    assert mem.forget(item["id"]) is True


def test_first_add_does_not_immediately_autosave(tmp_path: Path):
    path = tmp_path / "mem.pkl"
    mem = LexicalMemory(storage_path=str(path))

    mem.add_memory("first item")

    assert mem._dirty is True  # noqa: SLF001
    assert not path.exists()


def test_corrupt_file_recovery(tmp_path: Path):
    path = tmp_path / "bad.pkl"
    path.write_bytes(b"not-a-pickle")

    mem = LexicalMemory(storage_path=str(path))

    assert mem.memories == []
    assert mem.stats()["load_recovery_count"] == 1


def test_empty_search_does_not_force_disk_write(tmp_path: Path):
    path = tmp_path / "mem.pkl"
    mem = LexicalMemory(storage_path=str(path))

    assert mem.search("anything") == []
    assert not path.exists()


def test_store_memory_replace_query_replaces_user_name(tmp_path: Path):
    runtime, ws = _memory_runtime(tmp_path)
    skill = runtime.get_skill("memory-rag")
    assert skill is not None
    ctx = SkillContext(user_input="remember this", branch_labels=[], attachments=[], workspace_root=ws, memory_hits=[])

    first = runtime.execute_tool_call("store_memory", {"text": "User's name is Sohom"}, selected=[skill], ctx=ctx)
    old_id = int(first["data"]["id"])

    second = runtime.execute_tool_call(
        "store_memory",
        {"text": "User's name is Solus", "replace_query": "User's name is Sohom"},
        selected=[skill],
        ctx=ctx,
    )

    assert second["ok"] is True
    assert old_id in second["meta"].get("forgotten_ids", [])


def test_recall_memory_uses_token_matching_not_substrings(tmp_path: Path):
    runtime, ws = _memory_runtime(tmp_path)
    skill = runtime.get_skill("memory-rag")
    assert skill is not None
    ctx = SkillContext(user_input="remember this", branch_labels=[], attachments=[], workspace_root=ws, memory_hits=[])

    runtime.execute_tool_call("store_memory", {"text": "Joanna likes tea"}, selected=[skill], ctx=ctx)

    recalled = runtime.execute_tool_call(
        "recall_memory",
        {"query": "ann", "top_k": 3, "min_score": 0.01},
        selected=[skill],
        ctx=ctx,
    )

    assert recalled["ok"] is True
    assert recalled["data"]["hits"] == []


def test_recall_memory_finds_facts_by_tokens(tmp_path: Path):
    runtime, ws = _memory_runtime(tmp_path)
    skill = runtime.get_skill("memory-rag")
    assert skill is not None
    ctx = SkillContext(user_input="remember this", branch_labels=[], attachments=[], workspace_root=ws, memory_hits=[])

    runtime.execute_tool_call("store_memory", {"text": "My favorite editor is Neovim"}, selected=[skill], ctx=ctx)

    recalled = runtime.execute_tool_call(
        "recall_memory",
        {"query": "favorite editor", "top_k": 3},
        selected=[skill],
        ctx=ctx,
    )

    assert recalled["ok"] is True
    hits = recalled["data"]["hits"]
    assert hits
    assert "favorite editor is neovim" in str(hits[0]["text"]).lower()


def test_store_memory_rejects_empty_text(tmp_path: Path):
    runtime, ws = _memory_runtime(tmp_path)
    skill = runtime.get_skill("memory-rag")
    assert skill is not None
    ctx = SkillContext(user_input="remember this", branch_labels=[], attachments=[], workspace_root=ws, memory_hits=[])

    result = runtime.execute_tool_call("store_memory", {"text": "   "}, selected=[skill], ctx=ctx)

    assert result["ok"] is False
    assert result["error"]["code"] == "E_VALIDATION"


def test_recall_memory_rejects_empty_query(tmp_path: Path):
    runtime, ws = _memory_runtime(tmp_path)
    skill = runtime.get_skill("memory-rag")
    assert skill is not None
    ctx = SkillContext(user_input="remember this", branch_labels=[], attachments=[], workspace_root=ws, memory_hits=[])

    result = runtime.execute_tool_call("recall_memory", {"query": "   "}, selected=[skill], ctx=ctx)

    assert result["ok"] is False
    assert result["error"]["code"] == "E_VALIDATION"


def test_stats_reports_lexical_backend(tmp_path: Path):
    mem = LexicalMemory(storage_path=str(tmp_path / "mem.pkl"))

    stats = mem.stats()

    assert stats["backend"] == "lexical"
    assert stats["mode_label"] == "lexical"
    assert "count" in stats


def test_load_rejects_invalid_payload_shape(tmp_path: Path):
    path = tmp_path / "mem.pkl"
    path.write_text('{"schema_version":"3.0.0","id":1,"text":"x","metadata":[]}\n', encoding="utf-8")

    mem = LexicalMemory(storage_path=str(path))

    assert mem.memories == []
    assert mem.stats()["load_recovery_count"] == 1
    assert path.exists()


def test_load_rejects_legacy_schema_as_unsupported(tmp_path: Path):
    path = tmp_path / "mem.pkl"
    path.write_text(
        '{"schema_version":"2.0.0","id":1,"text":"legacy fact","metadata":{},"type":"conversation","timestamp":1.0,"access_count":0,"last_accessed":1.0}\n',
        encoding="utf-8",
    )

    mem = LexicalMemory(storage_path=str(path))

    stats = mem.stats()
    assert mem.memories == []
    assert stats["load_unsupported_count"] == 1
    assert path.exists()


def test_load_skips_malformed_json_line(tmp_path: Path):
    path = tmp_path / "mem.pkl"
    path.write_text('{"schema_version":"3.0.0","id":1,"text":"good","metadata":{},"type":"conversation","timestamp":1.0,"access_count":0,"last_accessed":1.0}\nnot-json\n', encoding="utf-8")

    mem = LexicalMemory(storage_path=str(path))

    stats = mem.stats()
    assert len(mem.memories) == 1
    assert stats["load_recovery_count"] == 1


def test_save_rotates_backups(tmp_path: Path):
    path = tmp_path / "mem.pkl"
    mem = LexicalMemory(storage_path=str(path), backup_revisions=1, autosave_every=100)

    mem.add_memory("first")
    mem.flush()
    first_primary = path.read_text(encoding="utf-8")
    mem.add_memory("second")
    mem.flush()

    primary = path.read_text(encoding="utf-8")
    backup = path.with_suffix(".pkl.bak1")

    assert path.exists()
    assert backup.exists()
    assert primary != first_primary
    assert backup.read_text(encoding="utf-8") == first_primary


def test_flush_writes_events_and_facts(tmp_path: Path):
    path = tmp_path / "mem.pkl"
    mem = LexicalMemory(storage_path=str(path), autosave_every=100)

    mem.add_memory("first")
    mem.add_memory("second", metadata={"k": "v"})
    mem.flush()

    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    assert len(records) == 2
    assert all(record["schema_version"] == mem.stats()["memory_schema_version"] for record in records)
    facts = path.parent / "facts.md"
    assert facts.exists()
    facts_text = facts.read_text(encoding="utf-8")
    assert "# Alphanus Memory Facts" in facts_text
    assert "text: first" in facts_text
    assert "text: second" in facts_text


def test_save_replace_failure_keeps_primary_file(tmp_path: Path, monkeypatch):
    path = tmp_path / "mem.pkl"
    mem = LexicalMemory(storage_path=str(path), backup_revisions=2, autosave_every=100)
    mem.add_memory("first")
    mem.flush()
    original = path.read_bytes()

    real_replace = os.replace

    def _failing_replace(src, dst):
        if str(dst) == str(path):
            raise OSError("simulated replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr("core.memory.os.replace", _failing_replace)
    mem.add_memory("second")
    with pytest.raises(OSError, match="simulated replace failure"):
        mem.flush()

    assert path.exists()
    assert path.read_bytes() == original
