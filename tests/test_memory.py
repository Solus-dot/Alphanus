from __future__ import annotations

import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

from core.memory import MemoryEncoderUnavailableError, VectorMemory
from core.skills import SkillContext, SkillRuntime
from core.workspace import WorkspaceManager

_REAL_LOAD_TRANSFORMER_ENCODER = VectorMemory._load_transformer_encoder


def _fake_encode(self, text: str):
    vec = np.zeros(16, dtype=np.float32)
    for token in text.lower().split():
        vec[sum(token.encode("utf-8")) % 16] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class _ToyTransformerEncoder:
    dim = 24

    def encode(self, texts, normalize_embeddings: bool = True) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in text.lower().split():
                idx = sum(token.encode("utf-8")) % self.dim
                vec[idx] += 1.0
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
            vectors.append(vec)
        return np.asarray(vectors, dtype=np.float32)


class _ToyTransformerEncoder384:
    dim = 384

    def encode(self, texts, normalize_embeddings: bool = True) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in text.lower().split():
                idx = (sum(token.encode("utf-8")) * 7) % self.dim
                vec[idx] += 1.0
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
            vectors.append(vec)
        return np.asarray(vectors, dtype=np.float32)


class _LegacyHashEncoder:
    dim = 384

    def encode(self, texts, normalize_embeddings: bool = True) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in text.lower().split():
                idx = sum(token.encode("utf-8")) % self.dim
                vec[idx] += 1.0
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
            vectors.append(vec)
        return np.asarray(vectors, dtype=np.float32)


def _write_legacy_hash_payload(path: Path, entries: list[tuple[str, str]]) -> None:
    encoder = _LegacyHashEncoder()
    now = time.time()
    payload = {
        "schema_version": "1.0.0",
        "model_name": "legacy-hash",
        "embedding_backend": "hash",
        "memories": [],
    }
    for idx, (text, memory_type) in enumerate(entries, start=1):
        timestamp = now + idx
        payload["memories"].append(
            {
                "id": idx,
                "text": text,
                "vector": encoder.encode([text], normalize_embeddings=True)[0],
                "metadata": {},
                "type": memory_type,
                "timestamp": timestamp,
                "access_count": 0,
                "last_accessed": timestamp,
            }
        )
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


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


def test_add_search_forget(monkeypatch, tmp_path: Path):
    mem = VectorMemory(storage_path=str(tmp_path / "mem.pkl"))
    monkeypatch.setattr(VectorMemory, "_encode", _fake_encode, raising=False)

    a = mem.add_memory("I like coffee", memory_type="preference")
    assert a["id"] >= 1

    hits = mem.search("coffee", top_k=3, min_score=0.0)
    assert hits
    assert hits[0]["id"] == a["id"]

    assert mem.forget(a["id"]) is True


def test_forget_updates_index_cache_without_full_rebuild(monkeypatch, tmp_path: Path):
    mem = VectorMemory(storage_path=str(tmp_path / "mem.pkl"))
    monkeypatch.setattr(VectorMemory, "_encode", _fake_encode, raising=False)

    coffee = mem.add_memory("I like coffee", memory_type="preference")
    tea = mem.add_memory("I like tea", memory_type="preference")
    guitar = mem.add_memory("I play guitar", memory_type="hobby")

    mem.search("tea", top_k=3, min_score=0.0)
    assert mem._matrix_cache is not None  # noqa: SLF001
    assert mem._type_index_cache["preference"].tolist() == [0, 1]  # noqa: SLF001

    monkeypatch.setattr(mem, "_encode_many", lambda _texts: (_ for _ in ()).throw(AssertionError("unexpected re-embed")))

    assert mem.forget(coffee["id"]) is True
    assert mem._matrix_cache is not None  # noqa: SLF001
    assert mem._matrix_cache.shape[0] == 2  # noqa: SLF001
    assert mem._all_index_cache.tolist() == [0, 1]  # noqa: SLF001
    assert mem._type_index_cache["preference"].tolist() == [0]  # noqa: SLF001
    assert mem._type_index_cache["hobby"].tolist() == [1]  # noqa: SLF001

    hits = mem.search("tea", top_k=3, min_score=0.0)
    assert hits
    assert hits[0]["id"] == tea["id"]
    assert all(hit["id"] != guitar["id"] or hit["type"] == "hobby" for hit in hits)


def test_first_add_does_not_immediately_autosave(monkeypatch, tmp_path: Path):
    path = tmp_path / "mem.pkl"
    mem = VectorMemory(storage_path=str(path))
    monkeypatch.setattr(VectorMemory, "_encode", _fake_encode, raising=False)

    mem.add_memory("first item")

    assert mem._dirty is True  # noqa: SLF001
    assert not path.exists()


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


def test_empty_search_does_not_force_disk_write(tmp_path: Path):
    path = tmp_path / "mem.pkl"
    mem = VectorMemory(storage_path=str(path))
    assert mem.search("anything") == []
    assert not path.exists()


def test_hash_backend_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="hash memory backend has been removed"):
        VectorMemory(storage_path=str(tmp_path / "mem.pkl"), embedding_backend="hash")


def test_store_memory_replace_query_replaces_user_name(tmp_path: Path):
    runtime, ws = _memory_runtime(tmp_path)
    skill = runtime.get_skill("memory-rag")
    assert skill is not None
    ctx = SkillContext(user_input="remember this", branch_labels=[], attachments=[], workspace_root=ws, memory_hits=[])

    first = runtime.execute_tool_call("store_memory", {"text": "User's name is Sohom"}, selected=[skill], ctx=ctx)
    assert first["ok"] is True
    old_id = int(first["data"]["id"])

    second = runtime.execute_tool_call(
        "store_memory",
        {"text": "User's name is Solus", "replace_query": "User's name is Sohom"},
        selected=[skill],
        ctx=ctx,
    )
    assert second["ok"] is True
    assert old_id in second["meta"].get("forgotten_ids", [])
    assert second["meta"].get("auto_resolution") == "replace_query"

    mem = VectorMemory(storage_path=str(tmp_path / "mem.pkl"))
    texts = [m["text"] for m in mem.list_recent(10)]
    assert "User's name is Sohom" not in texts
    assert "User's name is Solus" in texts


def test_store_memory_replace_query_replaces_conflicting_attribute(tmp_path: Path):
    runtime, ws = _memory_runtime(tmp_path)
    skill = runtime.get_skill("memory-rag")
    assert skill is not None
    ctx = SkillContext(user_input="remember this", branch_labels=[], attachments=[], workspace_root=ws, memory_hits=[])

    first = runtime.execute_tool_call("store_memory", {"text": "My favorite editor is Vim"}, selected=[skill], ctx=ctx)
    assert first["ok"] is True
    old_id = int(first["data"]["id"])

    second = runtime.execute_tool_call(
        "store_memory",
        {"text": "My favorite editor is Neovim", "replace_query": "My favorite editor is Vim"},
        selected=[skill],
        ctx=ctx,
    )
    assert second["ok"] is True
    assert old_id in second["meta"].get("forgotten_ids", [])
    assert second["meta"].get("auto_resolution") == "replace_query"

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


def test_recall_memory_semantic_search_finds_name_fact(tmp_path: Path):
    runtime, ws = _memory_runtime(tmp_path)
    skill = runtime.get_skill("memory-rag")
    assert skill is not None
    ctx = SkillContext(user_input="remember this", branch_labels=[], attachments=[], workspace_root=ws, memory_hits=[])

    stored = runtime.execute_tool_call("store_memory", {"text": "My name is Meems"}, selected=[skill], ctx=ctx)
    assert stored["ok"] is True

    recalled = runtime.execute_tool_call(
        "recall_memory",
        {"query": "my name", "top_k": 3},
        selected=[skill],
        ctx=ctx,
    )
    assert recalled["ok"] is True
    hits = recalled["data"]["hits"]
    assert hits
    assert any("name is meems" in str(hit.get("text", "")).lower() for hit in hits)


def test_recall_memory_without_lexical_fallback_avoids_substring_matches(tmp_path: Path):
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


def test_recall_memory_semantic_query_finds_birthdate_fact(tmp_path: Path):
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
        {"query": "birthdate", "top_k": 3},
        selected=[skill],
        ctx=ctx,
    )
    assert recalled["ok"] is True
    hits = recalled["data"]["hits"]
    assert hits
    assert any("birthdate is august 5, 2005" in str(hit.get("text", "")).lower() for hit in hits)


def test_export_memories_relative_path_stays_in_workspace(tmp_path: Path):
    runtime, ws = _memory_runtime(tmp_path)
    skill = runtime.get_skill("memory-rag")
    assert skill is not None
    ctx = SkillContext(user_input="remember this", branch_labels=[], attachments=[], workspace_root=ws, memory_hits=[])

    stored = runtime.execute_tool_call("store_memory", {"text": "My favorite editor is Neovim"}, selected=[skill], ctx=ctx)
    assert stored["ok"] is True

    exported = runtime.execute_tool_call(
        "export_memories",
        {"filepath": "exported_memories.txt"},
        selected=[skill],
        ctx=ctx,
    )
    assert exported["ok"] is True
    target = Path(exported["data"]["filepath"])
    assert target.parent == Path(ws)
    assert target.exists()


def test_recall_memory_semantic_query_finds_favorite_fact(tmp_path: Path):
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
        {"query": "favorite editor", "top_k": 3},
        selected=[skill],
        ctx=ctx,
    )
    assert recalled["ok"] is True
    hits = recalled["data"]["hits"]
    assert hits
    assert any("favorite editor is neovim" in str(hit.get("text", "")).lower() for hit in hits)


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


def test_transformer_backend_is_lazy(monkeypatch, tmp_path: Path):
    called = {"count": 0}

    def _counted(self, model_name):
        called["count"] += 1
        return _ToyTransformerEncoder384()

    monkeypatch.setattr(VectorMemory, "_load_transformer_encoder", _counted, raising=True)
    mem = VectorMemory(storage_path=str(tmp_path / "mem.pkl"))
    assert called["count"] == 0
    mem.add_memory("trigger encoder load")
    assert called["count"] == 1


def test_legacy_hash_payload_is_rejected(tmp_path: Path):
    path = tmp_path / "mem.pkl"
    _write_legacy_hash_payload(path, [("favorite editor is neovim", "preference")])

    with pytest.raises(ValueError, match="Legacy hash-backed memory stores are no longer supported"):
        VectorMemory(storage_path=str(path), model_name="toy-transformer")


def test_memory_operations_raise_when_embeddings_unavailable(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(VectorMemory, "_load_transformer_encoder", lambda self, _model_name: None, raising=True)
    mem = VectorMemory(storage_path=str(tmp_path / "mem.pkl"))
    mem._set_encoder_state("unavailable", "test", "encoder unavailable")  # noqa: SLF001

    with pytest.raises(MemoryEncoderUnavailableError, match="encoder unavailable"):
        mem.add_memory("cannot embed")


def test_stats_reports_encoder_unavailable_before_first_memory_operation(monkeypatch, tmp_path: Path):
    calls: list[tuple[str, bool]] = []

    class _FakeSentenceTransformer:
        def __init__(self, model_name: str, local_files_only: bool) -> None:
            calls.append((model_name, local_files_only))
            raise OSError("missing local weights")

    monkeypatch.setattr(VectorMemory, "_load_transformer_encoder", _REAL_LOAD_TRANSFORMER_ENCODER, raising=True)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
    )
    mem = VectorMemory(
        storage_path=str(tmp_path / "mem.pkl"),
        model_name="toy-model",
        allow_model_download=False,
    )

    stats = mem.stats()

    assert calls == [("toy-model", True)]
    assert stats["embedding_backend"] == "transformer"
    assert stats["encoder_status"] == "unavailable"
    assert stats["encoder_source"] == "transformer-local"
    assert stats["mode_label"] == "semantic-unavailable"


def test_transformer_respects_allow_model_download_false(monkeypatch, tmp_path: Path):
    calls: list[tuple[str, bool]] = []

    class _FakeSentenceTransformer:
        def __init__(self, model_name: str, local_files_only: bool) -> None:
            calls.append((model_name, local_files_only))
            raise OSError("missing local weights")

    monkeypatch.setattr(VectorMemory, "_load_transformer_encoder", _REAL_LOAD_TRANSFORMER_ENCODER, raising=True)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
    )
    mem = VectorMemory(
        storage_path=str(tmp_path / "mem.pkl"),
        model_name="toy-model",
        allow_model_download=False,
    )

    with pytest.raises(MemoryEncoderUnavailableError):
        mem.add_memory("offline transformer request")

    stats = mem.stats()
    assert calls
    assert all(call == ("toy-model", True) for call in calls)
    assert stats["encoder_status"] == "unavailable"
    assert stats["encoder_source"] == "transformer-local"
    assert "allow_model_download=false" in stats["encoder_detail"]


def test_transformer_can_retry_with_download_enabled(monkeypatch, tmp_path: Path):
    calls: list[tuple[str, bool]] = []

    class _DownloadedEncoder:
        dim = 12

        def encode(self, texts, normalize_embeddings: bool = True) -> np.ndarray:
            vectors = np.ones((len(texts), self.dim), dtype=np.float32)
            if normalize_embeddings:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / norms
            return vectors

    class _FakeSentenceTransformer:
        def __init__(self, model_name: str, local_files_only: bool) -> None:
            calls.append((model_name, local_files_only))
            if local_files_only:
                raise OSError("missing local weights")
            self._delegate = _DownloadedEncoder()
            self.dim = self._delegate.dim

        def encode(self, texts, normalize_embeddings: bool = True) -> np.ndarray:
            return self._delegate.encode(texts, normalize_embeddings=normalize_embeddings)

    monkeypatch.setattr(VectorMemory, "_load_transformer_encoder", _REAL_LOAD_TRANSFORMER_ENCODER, raising=True)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
    )
    mem = VectorMemory(
        storage_path=str(tmp_path / "mem.pkl"),
        model_name="toy-model",
        allow_model_download=True,
    )

    mem.add_memory("download-enabled transformer request")
    stats = mem.stats()

    assert calls == [("toy-model", True), ("toy-model", False)]
    assert mem.embedding_backend == "transformer"
    assert stats["encoder_status"] == "ready"
    assert stats["encoder_source"] == "transformer-download"


def test_encoder_can_retry_after_initial_unavailable_probe(monkeypatch, tmp_path: Path):
    calls = {"count": 0}

    def _flaky_loader(self, _model_name):
        calls["count"] += 1
        if calls["count"] == 1:
            self._set_encoder_state("unavailable", "test-loader", "temporary failure")  # noqa: SLF001
            return None
        self._set_encoder_state("ready", "test-loader", "recovered")  # noqa: SLF001
        return _ToyTransformerEncoder384()

    monkeypatch.setattr(VectorMemory, "_load_transformer_encoder", _flaky_loader, raising=True)
    mem = VectorMemory(storage_path=str(tmp_path / "mem.pkl"))

    first_stats = mem.stats()
    assert first_stats["encoder_status"] == "unavailable"
    assert calls["count"] == 1

    item = mem.add_memory("retry succeeds")
    assert item["id"] == 1
    assert calls["count"] == 2

    second_stats = mem.stats()
    assert second_stats["encoder_status"] == "ready"
    assert second_stats["encoder_source"] == "test-loader"


def test_add_memory_keeps_warm_cache_incremental(monkeypatch, tmp_path: Path):
    mem = VectorMemory(storage_path=str(tmp_path / "mem.pkl"))
    monkeypatch.setattr(VectorMemory, "_encode", _fake_encode, raising=False)

    mem.add_memory("alpha beta", memory_type="note")
    assert mem.search("alpha", top_k=1, min_score=0.0)
    assert mem._matrix_cache is not None  # noqa: SLF001

    mem.add_memory("gamma delta", memory_type="note")

    assert mem._matrix_cache is not None  # noqa: SLF001
    assert mem._matrix_cache.shape == (2, 16)  # noqa: SLF001
    assert mem._all_index_cache.tolist() == [0, 1]  # noqa: SLF001
    assert mem._type_index_cache["note"].tolist() == [0, 1]  # noqa: SLF001

