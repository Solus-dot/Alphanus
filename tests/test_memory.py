from __future__ import annotations

import hashlib
import importlib.util
import time
from pathlib import Path

import numpy as np

from core.memory import VectorMemory, _HashEncoder
from core.skills import SkillContext, SkillRuntime
from core.workspace import WorkspaceManager


def _fake_encode(self, text: str):
    vec = np.zeros(16, dtype=np.float32)
    for token in text.lower().split():
        vec[hash(token) % 16] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _memory_runtime(tmp_path: Path) -> tuple[SkillRuntime, str]:
    repo_root = Path(__file__).resolve().parents[1]
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    runtime = SkillRuntime(
        skills_dir=str(repo_root / "skills"),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )
    return runtime, str(ws)


def _load_memory_tools_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "skills" / "memory-rag" / "tools.py"
    spec = importlib.util.spec_from_file_location("test_memory_rag_tools", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_add_search_forget(monkeypatch, tmp_path: Path):
    mem = VectorMemory(storage_path=str(tmp_path / "mem.pkl"))
    monkeypatch.setattr(VectorMemory, "_encode", _fake_encode, raising=False)

    a = mem.add_memory("I like coffee", memory_type="preference")
    assert a["id"] >= 1

    hits = mem.search("coffee", top_k=3, min_score=0.0)
    assert hits
    assert hits[0]["id"] == a["id"]

    assert mem.forget(a["id"]) is True


def test_corrupt_file_recovery(tmp_path: Path):
    path = tmp_path / "bad.pkl"
    path.write_bytes(b"not-a-pickle")

    VectorMemory(storage_path=str(path))
    assert path.with_suffix(".pkl.corrupted").exists()


def test_scale_scan_under_500ms(monkeypatch, tmp_path: Path):
    mem = VectorMemory(storage_path=str(tmp_path / "mem.pkl"))
    monkeypatch.setattr(VectorMemory, "_encode", _fake_encode, raising=False)

    for i in range(5000):
        mem.add_memory(f"item {i}")

    t0 = time.perf_counter()
    mem.search("item 123", top_k=5, min_score=0.0)
    dt_ms = (time.perf_counter() - t0) * 1000
    assert dt_ms < 500


def test_hash_encoder_uses_stable_digest_indices():
    enc = _HashEncoder(dim=64)
    vec = enc.encode(["coffee tea coffee"], normalize_embeddings=False)[0]

    coffee_idx = int.from_bytes(hashlib.blake2b(b"coffee", digest_size=8).digest(), "little") % 64
    tea_idx = int.from_bytes(hashlib.blake2b(b"tea", digest_size=8).digest(), "little") % 64

    assert vec[coffee_idx] >= 2.0
    assert vec[tea_idx] >= 1.0


def test_empty_search_does_not_force_disk_write(tmp_path: Path):
    path = tmp_path / "mem.pkl"
    mem = VectorMemory(storage_path=str(path))
    assert mem.search("anything") == []
    assert not path.exists()


def test_store_memory_auto_replaces_conflicting_user_name(tmp_path: Path):
    runtime, ws = _memory_runtime(tmp_path)
    skill = runtime.get_skill("memory-rag")
    assert skill is not None
    ctx = SkillContext(user_input="remember this", branch_labels=[], attachments=[], workspace_root=ws, memory_hits=[])

    first = runtime.execute_tool_call("store_memory", {"text": "User's name is Sohom"}, selected=[skill], ctx=ctx)
    assert first["ok"] is True
    old_id = int(first["data"]["id"])

    second = runtime.execute_tool_call("store_memory", {"text": "User's name is Solus"}, selected=[skill], ctx=ctx)
    assert second["ok"] is True
    assert old_id in second["meta"].get("forgotten_ids", [])

    mem = VectorMemory(storage_path=str(tmp_path / "mem.pkl"))
    texts = [m["text"] for m in mem.list_recent(10)]
    assert "User's name is Sohom" not in texts
    assert "User's name is Solus" in texts


def test_store_memory_auto_replaces_conflicting_attribute(tmp_path: Path):
    runtime, ws = _memory_runtime(tmp_path)
    skill = runtime.get_skill("memory-rag")
    assert skill is not None
    ctx = SkillContext(user_input="remember this", branch_labels=[], attachments=[], workspace_root=ws, memory_hits=[])

    first = runtime.execute_tool_call("store_memory", {"text": "My favorite editor is Vim"}, selected=[skill], ctx=ctx)
    assert first["ok"] is True
    old_id = int(first["data"]["id"])

    second = runtime.execute_tool_call("store_memory", {"text": "My favorite editor is Neovim"}, selected=[skill], ctx=ctx)
    assert second["ok"] is True
    assert old_id in second["meta"].get("forgotten_ids", [])
    assert second["meta"].get("auto_resolution") == "user.favorite_editor"

    mem = VectorMemory(storage_path=str(tmp_path / "mem.pkl"))
    texts = [m["text"] for m in mem.list_recent(10)]
    assert "My favorite editor is Vim" not in texts
    assert "My favorite editor is Neovim" in texts


def test_store_memory_replace_query_can_replace_non_fact(tmp_path: Path):
    runtime, ws = _memory_runtime(tmp_path)
    skill = runtime.get_skill("memory-rag")
    assert skill is not None
    ctx = SkillContext(user_input="remember this", branch_labels=[], attachments=[], workspace_root=ws, memory_hits=[])

    first = runtime.execute_tool_call("store_memory", {"text": "My go-to stack: Flask + SQLite"}, selected=[skill], ctx=ctx)
    assert first["ok"] is True
    old_id = int(first["data"]["id"])

    second = runtime.execute_tool_call(
        "store_memory",
        {
            "text": "My go-to stack: FastAPI + Postgres",
            "replace_query": "go-to stack",
            "replace_min_score": 0.0,
        },
        selected=[skill],
        ctx=ctx,
    )
    assert second["ok"] is True
    assert old_id in second["meta"].get("forgotten_ids", [])
    assert second["meta"].get("auto_resolution") == "replace_query"


def test_recall_memory_fallback_finds_name_fact(tmp_path: Path):
    runtime, ws = _memory_runtime(tmp_path)
    skill = runtime.get_skill("memory-rag")
    assert skill is not None
    ctx = SkillContext(user_input="remember this", branch_labels=[], attachments=[], workspace_root=ws, memory_hits=[])

    stored = runtime.execute_tool_call("store_memory", {"text": "My name is Meems"}, selected=[skill], ctx=ctx)
    assert stored["ok"] is True

    recalled = runtime.execute_tool_call(
        "recall_memory",
        {"query": "user name", "top_k": 3},
        selected=[skill],
        ctx=ctx,
    )
    assert recalled["ok"] is True
    hits = recalled["data"]["hits"]
    assert hits
    assert any("name is meems" in str(hit.get("text", "")).lower() for hit in hits)


def test_fact_from_item_uses_metadata_only():
    tools = _load_memory_tools_module()

    class _Item:
        metadata = {}
        text = "My favorite editor is Vim"

    assert tools._fact_from_item(_Item()) is None


def test_recall_memory_lexical_fallback_avoids_substring_matches(tmp_path: Path):
    runtime, ws = _memory_runtime(tmp_path)
    skill = runtime.get_skill("memory-rag")
    assert skill is not None
    ctx = SkillContext(user_input="remember this", branch_labels=[], attachments=[], workspace_root=ws, memory_hits=[])

    stored = runtime.execute_tool_call("store_memory", {"text": "Joanna likes tea"}, selected=[skill], ctx=ctx)
    assert stored["ok"] is True

    recalled = runtime.execute_tool_call(
        "recall_memory",
        {"query": "ann", "top_k": 3, "min_score": 1.1},
        selected=[skill],
        ctx=ctx,
    )
    assert recalled["ok"] is True
    assert recalled["data"]["hits"] == []


def test_recall_memory_birthday_query_finds_birthdate_fact(tmp_path: Path):
    runtime, ws = _memory_runtime(tmp_path)
    skill = runtime.get_skill("memory-rag")
    assert skill is not None
    ctx = SkillContext(user_input="remember this", branch_labels=[], attachments=[], workspace_root=ws, memory_hits=[])

    stored = runtime.execute_tool_call(
        "store_memory",
        {"text": "My birthdate is August 5, 2005"},
        selected=[skill],
        ctx=ctx,
    )
    assert stored["ok"] is True

    recalled = runtime.execute_tool_call(
        "recall_memory",
        {"query": "birthday", "top_k": 3},
        selected=[skill],
        ctx=ctx,
    )
    assert recalled["ok"] is True
    hits = recalled["data"]["hits"]
    assert hits
    assert any("birthdate is august 5, 2005" in str(hit.get("text", "")).lower() for hit in hits)


def test_recall_memory_favourite_query_finds_favorite_fact(tmp_path: Path):
    runtime, ws = _memory_runtime(tmp_path)
    skill = runtime.get_skill("memory-rag")
    assert skill is not None
    ctx = SkillContext(user_input="remember this", branch_labels=[], attachments=[], workspace_root=ws, memory_hits=[])

    stored = runtime.execute_tool_call(
        "store_memory",
        {"text": "My favorite editor is Neovim"},
        selected=[skill],
        ctx=ctx,
    )
    assert stored["ok"] is True

    recalled = runtime.execute_tool_call(
        "recall_memory",
        {"query": "my favourite editor", "top_k": 3},
        selected=[skill],
        ctx=ctx,
    )
    assert recalled["ok"] is True
    hits = recalled["data"]["hits"]
    assert hits
    assert any("favorite editor is neovim" in str(hit.get("text", "")).lower() for hit in hits)


def test_default_hash_backend_never_attempts_transformer(monkeypatch, tmp_path: Path):
    called = {"count": 0}

    def _never(self, model_name):
        called["count"] += 1
        raise AssertionError("transformer loader should not be called for hash backend")

    monkeypatch.setattr(VectorMemory, "_load_transformer_encoder", _never, raising=True)
    mem = VectorMemory(storage_path=str(tmp_path / "mem.pkl"), embedding_backend="hash")
    mem.add_memory("hash backend only")
    assert called["count"] == 0


def test_transformer_backend_is_lazy(monkeypatch, tmp_path: Path):
    called = {"count": 0}

    def _counted(self, model_name):
        called["count"] += 1
        return _HashEncoder(dim=384)

    monkeypatch.setattr(VectorMemory, "_load_transformer_encoder", _counted, raising=True)
    mem = VectorMemory(storage_path=str(tmp_path / "mem.pkl"), embedding_backend="transformer")
    assert called["count"] == 0
    mem.add_memory("trigger encoder load")
    assert called["count"] == 1
