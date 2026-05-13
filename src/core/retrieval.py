from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def default_retrieval_store_path() -> Path:
    root = os.environ.get("ALPHANUS_APP_ROOT", "").strip()
    state_root = Path(os.path.expanduser(root)).resolve() if root else (Path.home() / ".alphanus").resolve()
    return state_root / "retrieval" / "index.sqlite"


def configured_store_path(config: dict[str, Any] | None) -> Path:
    retrieval = config.get("retrieval", {}) if isinstance(config, dict) else {}
    raw = retrieval.get("store_path") if isinstance(retrieval, dict) else ""
    return Path(os.path.expanduser(str(raw))).resolve() if raw else default_retrieval_store_path()


@dataclass(frozen=True, slots=True)
class RetrievalRecord:
    id: int
    record_type: str
    source: str
    title: str
    content_hash: str


class SQLiteRetrievalStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            existing_fts = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'chunks_fts'"
            ).fetchone()
            if existing_fts and "content=''" in str(existing_fts["sql"] or ""):
                conn.execute("DROP TABLE chunks_fts")
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS records (
                    id INTEGER PRIMARY KEY,
                    record_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    canonical_source TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    content_hash TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    fetched_at INTEGER NOT NULL DEFAULT 0,
                    stale_after INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(record_type, canonical_source)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    record_id INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    UNIQUE(record_id, chunk_index)
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(text, title, source, record_type);
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector_json TEXT NOT NULL
                );
                """
            )
            self._cleanup_orphans(conn)

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _delete_record_chunks(conn: sqlite3.Connection, record_id: int) -> None:
        chunk_ids = [int(item["id"]) for item in conn.execute("SELECT id FROM chunks WHERE record_id = ?", (record_id,))]
        if chunk_ids:
            params = [(chunk_id,) for chunk_id in chunk_ids]
            conn.executemany("DELETE FROM chunks_fts WHERE rowid = ?", params)
            conn.executemany("DELETE FROM embeddings WHERE chunk_id = ?", params)
        conn.execute("DELETE FROM chunks WHERE record_id = ?", (record_id,))

    @staticmethod
    def _cleanup_orphans(conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM embeddings WHERE chunk_id NOT IN (SELECT id FROM chunks)")
        conn.execute("DELETE FROM chunks_fts WHERE rowid NOT IN (SELECT id FROM chunks)")
        conn.execute("DELETE FROM chunks WHERE record_id NOT IN (SELECT id FROM records)")
        conn.execute("DELETE FROM embeddings WHERE chunk_id NOT IN (SELECT id FROM chunks)")
        conn.execute("DELETE FROM chunks_fts WHERE rowid NOT IN (SELECT id FROM chunks)")

    @staticmethod
    def chunk_text(text: str, *, max_chars: int = 1400, overlap: int = 160) -> list[str]:
        clean = "\n".join(line.strip() for line in str(text or "").splitlines() if line.strip())
        if not clean:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(clean):
            end = min(len(clean), start + max_chars)
            chunks.append(clean[start:end].strip())
            if end >= len(clean):
                break
            start = max(0, end - overlap)
        return [chunk for chunk in chunks if chunk]

    def upsert_record(
        self,
        *,
        record_type: str,
        source: str,
        title: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        canonical_source: str = "",
        fetched_at: int = 0,
        ttl_seconds: int = 0,
    ) -> RetrievalRecord | None:
        chunks = self.chunk_text(text)
        if not chunks:
            return None
        now = int(time.time())
        canonical = canonical_source or source
        content_hash = self._hash(text)
        stale_after = (fetched_at or now) + ttl_seconds if ttl_seconds > 0 else 0
        metadata_json = json.dumps(metadata or {}, sort_keys=True)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM records WHERE record_type = ? AND canonical_source = ?",
                (record_type, canonical),
            ).fetchone()
            if row:
                record_id = int(row["id"])
                conn.execute(
                    """
                    UPDATE records
                    SET source = ?, title = ?, metadata_json = ?, content_hash = ?,
                        updated_at = ?, fetched_at = ?, stale_after = ?
                    WHERE id = ?
                    """,
                    (source, title, metadata_json, content_hash, now, fetched_at, stale_after, record_id),
                )
                self._delete_record_chunks(conn, record_id)
            else:
                cur = conn.execute(
                    """
                    INSERT INTO records
                    (record_type, source, canonical_source, title, metadata_json, content_hash, created_at, updated_at, fetched_at, stale_after)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (record_type, source, canonical, title, metadata_json, content_hash, now, now, fetched_at, stale_after),
                )
                if cur.lastrowid is None:
                    raise RuntimeError("SQLite did not return a retrieval record id")
                record_id = int(cur.lastrowid)
            for index, chunk in enumerate(chunks):
                cur = conn.execute(
                    "INSERT INTO chunks(record_id, chunk_index, text) VALUES (?, ?, ?)",
                    (record_id, index, chunk),
                )
                if cur.lastrowid is None:
                    raise RuntimeError("SQLite did not return a retrieval chunk id")
                conn.execute(
                    "INSERT INTO chunks_fts(rowid, text, title, source, record_type) VALUES (?, ?, ?, ?, ?)",
                    (int(cur.lastrowid), chunk, title, source, record_type),
                )
        return RetrievalRecord(record_id, record_type, source, title, content_hash)

    def search(self, query: str, *, top_k: int = 5, sources: list[str] | None = None) -> list[dict[str, Any]]:
        clean = " ".join(str(query or "").split())
        if not clean:
            return []
        quoted_tokens = []
        for token in clean.split()[:12]:
            quoted_tokens.append('"' + token.replace('"', '""') + '"')
        fts_query = " OR ".join(quoted_tokens)
        source_filter = {item.strip() for item in sources or [] if item.strip()}
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id AS chunk_id, c.chunk_index, c.text, r.*
                FROM chunks_fts f
                JOIN chunks c ON c.id = f.rowid
                JOIN records r ON r.id = c.record_id
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
                """,
                (fts_query, max(1, top_k * 3)),
            ).fetchall()
        out: list[dict[str, Any]] = []
        now = int(time.time())
        for row in rows:
            record_type = str(row["record_type"])
            if source_filter and record_type not in source_filter:
                continue
            try:
                metadata = json.loads(str(row["metadata_json"] or "{}"))
            except json.JSONDecodeError:
                metadata = {}
            out.append(
                {
                    "record_id": int(row["id"]),
                    "chunk_id": int(row["chunk_id"]),
                    "record_type": record_type,
                    "source": str(row["source"]),
                    "title": str(row["title"]),
                    "text": str(row["text"]),
                    "chunk_index": int(row["chunk_index"]),
                    "metadata": metadata,
                    "fetched_at": int(row["fetched_at"] or 0),
                    "stale": bool(row["stale_after"] and int(row["stale_after"]) < now),
                }
            )
            if len(out) >= top_k:
                break
        return out

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def chunk_texts_for_record(self, record_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, chunk_index, text FROM chunks WHERE record_id = ? ORDER BY chunk_index",
                (record_id,),
            ).fetchall()
        return [{"chunk_id": int(row["id"]), "chunk_index": int(row["chunk_index"]), "text": str(row["text"])} for row in rows]

    def set_chunk_embedding(self, *, chunk_id: int, model: str, vector: list[float]) -> None:
        payload = json.dumps([float(item) for item in vector])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO embeddings(chunk_id, model, dimensions, vector_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    model = excluded.model,
                    dimensions = excluded.dimensions,
                    vector_json = excluded.vector_json
                """,
                (chunk_id, model, len(vector), payload),
            )

    def dense_search(self, query_vector: list[float], *, top_k: int = 5, sources: list[str] | None = None) -> list[dict[str, Any]]:
        if not query_vector:
            return []
        source_filter = {item.strip() for item in sources or [] if item.strip()}
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id AS chunk_id, c.chunk_index, c.text, e.vector_json, r.*
                FROM embeddings e
                JOIN chunks c ON c.id = e.chunk_id
                JOIN records r ON r.id = c.record_id
                """
            ).fetchall()
        ranked: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            record_type = str(row["record_type"])
            if source_filter and record_type not in source_filter:
                continue
            try:
                vector = [float(item) for item in json.loads(str(row["vector_json"]))]
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            score = self._cosine(query_vector, vector)
            if score > 0:
                ranked.append((score, row))
        ranked.sort(key=lambda item: item[0], reverse=True)
        out: list[dict[str, Any]] = []
        now = int(time.time())
        for score, row in ranked[: max(1, top_k)]:
            try:
                metadata = json.loads(str(row["metadata_json"] or "{}"))
            except json.JSONDecodeError:
                metadata = {}
            out.append(
                {
                    "record_id": int(row["id"]),
                    "chunk_id": int(row["chunk_id"]),
                    "record_type": str(row["record_type"]),
                    "source": str(row["source"]),
                    "title": str(row["title"]),
                    "text": str(row["text"]),
                    "chunk_index": int(row["chunk_index"]),
                    "metadata": metadata,
                    "score": round(score, 4),
                    "retrieval_mode": "dense",
                    "fetched_at": int(row["fetched_at"] or 0),
                    "stale": bool(row["stale_after"] and int(row["stale_after"]) < now),
                }
            )
        return out

    def forget(self, record_id: int) -> bool:
        with self._connect() as conn:
            self._delete_record_chunks(conn, record_id)
            cur = conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
            return cur.rowcount > 0

    def forget_source(self, record_type: str, canonical_source: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM records WHERE record_type = ? AND canonical_source = ?",
                (record_type, canonical_source),
            ).fetchone()
        return self.forget(int(row["id"])) if row else False

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            records = int(conn.execute("SELECT COUNT(*) FROM records").fetchone()[0])
            chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            embeddings = int(conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
            stale = int(conn.execute("SELECT COUNT(*) FROM records WHERE stale_after > 0 AND stale_after < ?", (int(time.time()),)).fetchone()[0])
            by_type = {
                str(row["record_type"]): int(row["count"])
                for row in conn.execute("SELECT record_type, COUNT(*) AS count FROM records GROUP BY record_type")
            }
        return {"path": str(self.path), "records": records, "chunks": chunks, "embeddings": embeddings, "stale_records": stale, "by_type": by_type}
