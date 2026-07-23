from __future__ import annotations

import sqlite3

from core.retrieval import SQLiteRetrievalStore, default_retrieval_store_path


def _record(
    store: SQLiteRetrievalStore,
    source: str,
    text: str,
    *,
    workspace: str = "/workspace",
    session_id: str = "session-a",
):
    record = store.upsert_record(
        record_type="tool_outcome",
        source=source,
        canonical_source=source,
        title=source,
        text=text,
        metadata={"workspace": workspace, "session_id": session_id},
    )
    assert record is not None
    return record


def test_dense_search_never_mixes_models_or_dimensions(tmp_path) -> None:
    store = SQLiteRetrievalStore(tmp_path / "retrieval.sqlite")
    first = _record(store, "first", "lexical alpha")
    second = _record(store, "second", "semantic beta")
    first_chunk = store.chunk_texts_for_record(first.id)[0]
    second_chunk = store.chunk_texts_for_record(second.id)[0]
    store.set_chunk_embedding(chunk_id=first_chunk["chunk_id"], model="model-a", vector=[1.0, 0.0])
    store.set_chunk_embedding(chunk_id=second_chunk["chunk_id"], model="model-b", vector=[1.0, 0.0])

    hits = store.dense_search([1.0, 0.0], model="model-a", top_k=10)

    assert [hit["source"] for hit in hits] == ["first"]
    assert store.dense_search([1.0], model="model-a", top_k=10) == []


def test_hybrid_search_uses_weighted_rrf_and_lexical_fallback(tmp_path) -> None:
    store = SQLiteRetrievalStore(tmp_path / "retrieval.sqlite")
    lexical = _record(store, "lexical", "alpha release")
    semantic = _record(store, "semantic", "unrelated words")
    store.set_chunk_embedding(
        chunk_id=store.chunk_texts_for_record(lexical.id)[0]["chunk_id"],
        model="embed",
        vector=[0.0, 1.0],
    )
    store.set_chunk_embedding(
        chunk_id=store.chunk_texts_for_record(semantic.id)[0]["chunk_id"],
        model="embed",
        vector=[1.0, 0.0],
    )

    hits = store.hybrid_search("alpha", [1.0, 0.0], model="embed", top_k=2)

    assert hits[0]["source"] == "semantic"
    assert all(hit["retrieval_mode"] == "hybrid_rrf" for hit in hits)
    assert store.hybrid_search("alpha", [1.0], model="embed", top_k=2)[0]["source"] == "lexical"


def test_embedding_cache_is_keyed_by_content_model_and_dimensions(tmp_path) -> None:
    store = SQLiteRetrievalStore(tmp_path / "retrieval.sqlite")
    record = _record(store, "cached", "same content")
    chunk = store.chunk_texts_for_record(record.id)[0]
    store.set_chunk_embedding(chunk_id=chunk["chunk_id"], model="embed-a", vector=[0.2, 0.8])

    assert store.cached_embedding("same content", model="embed-a", dimensions=2) == [0.2, 0.8]
    assert store.cached_embedding("same content", model="embed-b", dimensions=2) is None
    assert store.cached_embedding("same content", model="embed-a", dimensions=3) is None


def test_tool_outcome_compaction_and_session_deletion_are_scoped(tmp_path) -> None:
    path = tmp_path / "retrieval.sqlite"
    store = SQLiteRetrievalStore(path)
    for index in range(4):
        _record(store, f"a-{index}", f"outcome {index}", session_id="session-a")
    _record(store, "b", "other workspace", workspace="/other", session_id="session-b")
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE records SET updated_at=0 WHERE canonical_source='a-0'")

    removed = store.compact_tool_outcomes("/workspace", retention_days=30, max_records=2)

    assert removed == 2
    assert store.stats()["records"] == 3
    assert store.delete_session_tool_outcomes("session-a") == 2
    assert store.search("other workspace", top_k=5)[0]["source"] == "b"


def test_default_store_is_always_inside_isolated_test_home(isolated_alphanus_home) -> None:
    assert default_retrieval_store_path().is_relative_to(isolated_alphanus_home)
