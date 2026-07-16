import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MemoryItem:
    id: int
    text: str
    metadata: dict[str, Any]
    type: str
    timestamp: float
    access_count: int
    last_accessed: float


def _normalize_threshold(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    if not math.isfinite(parsed):
        parsed = float(default)
    return max(0.0, min(1.0, parsed))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _score_tokens(query_tokens: list[str], text_tokens: list[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    query_set = set(query_tokens)
    overlap = len(query_set & set(text_tokens)) / len(query_set)
    width = len(query_tokens)
    phrase = any(text_tokens[index : index + width] == query_tokens for index in range(len(text_tokens) - width + 1))
    return min(1.0, (0.75 * overlap) + (0.3 if phrase else 0.0))


def _to_public(item: MemoryItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "text": item.text,
        "metadata": item.metadata,
        "type": item.type,
        "timestamp": item.timestamp,
        "access_count": item.access_count,
        "last_accessed": item.last_accessed,
    }
class LexicalMemory:
    """SQLite-backed lexical memory with bounded candidate retrieval."""

    def __init__(self, storage_path: str, min_score: float = 0.3, persist_access_updates: bool = False,
                 backup_revisions: int = 0, **_ignored: Any) -> None:
        legacy_path = Path(os.path.expanduser(storage_path)).resolve()
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        if legacy_path.exists() and legacy_path.suffix != ".db":
            raise ValueError(
                f"Legacy unversioned memory found at {legacy_path}. Alphanus v1 does not migrate it; export or remove it first."
            )
        self.storage_path = legacy_path if legacy_path.suffix == ".db" else legacy_path.parent / "memory.db"
        self.min_score = _normalize_threshold(min_score, default=0.3)
        self.persist_access_updates = bool(persist_access_updates)
        self.backup_revisions = int(backup_revisions)
        self._connection = sqlite3.connect(self.storage_path, timeout=5.0, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=5000")
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS memories(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    normalized_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_type_time ON memories(type, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_memories_time ON memories(timestamp DESC);
                """
            )
            self._connection.execute("INSERT OR IGNORE INTO schema_migrations VALUES (1, ?)", (time.time(),))

    @property
    def memories(self) -> list[MemoryItem]:
        rows = self._connection.execute(
            "SELECT * FROM memories ORDER BY timestamp DESC LIMIT 10000"
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(row["id"], row["text"], json.loads(row["metadata_json"]), row["type"],
                          row["timestamp"], row["access_count"], row["last_accessed"])

    def add_memory(self, text: str, memory_type: str = "conversation", metadata: dict[str, Any] | None = None,
                   importance: float | None = None) -> dict[str, Any]:
        now = time.time()
        md = dict(metadata or {})
        if importance is not None:
            md["importance"] = float(importance)
        with self._connection:
            cursor = self._connection.execute(
                "INSERT INTO memories(text,normalized_text,metadata_json,type,timestamp,last_accessed) VALUES (?,?,?,?,?,?)",
                (str(text), " ".join(str(text).casefold().split()), json.dumps(md, ensure_ascii=False), str(memory_type), now, now),
            )
        return _to_public(MemoryItem(cursor.lastrowid or 0, str(text), md, str(memory_type), now, 0, now))

    def search(self, query: str, top_k: int = 5, memory_type: str | None = None,
               min_score: float | None = None) -> list[dict[str, Any]]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        threshold = self.min_score if min_score is None else _normalize_threshold(min_score, default=self.min_score)
        clauses = ["normalized_text LIKE ?" for _ in query_tokens[:8]]
        params: list[Any] = [f"%{token}%" for token in query_tokens[:8]]
        where = "(" + " OR ".join(clauses) + ")"
        if memory_type:
            where += " AND type=?"
            params.append(memory_type)
        params.append(max(100, min(2000, max(1, int(top_k)) * 40)))
        rows = self._connection.execute(
            f"SELECT * FROM memories WHERE {where} ORDER BY timestamp DESC LIMIT ?", params  # noqa: S608
        ).fetchall()
        scored: list[tuple[float, MemoryItem]] = []
        for row in rows:
            item = self._row_to_item(row)
            score = _score_tokens(query_tokens, _tokenize(item.text))
            if score >= threshold:
                scored.append((score, item))
        selected = sorted(scored, key=lambda pair: (pair[0], pair[1].timestamp), reverse=True)[:max(1, int(top_k))]
        now = time.time()
        if selected and self.persist_access_updates:
            with self._connection:
                self._connection.executemany(
                    "UPDATE memories SET access_count=access_count+1,last_accessed=? WHERE id=?",
                    [(now, item.id) for _score, item in selected],
                )
        output = []
        for score, item in selected:
            item.access_count += 1
            item.last_accessed = now
            record = _to_public(item)
            record["score"] = round(score, 4)
            output.append(record)
        return output

    def forget(self, memory_id: int) -> bool:
        with self._connection:
            cursor = self._connection.execute("DELETE FROM memories WHERE id=?", (int(memory_id),))
        return cursor.rowcount > 0

    def list_recent(self, count: int = 5) -> list[dict[str, Any]]:
        rows = self._connection.execute("SELECT * FROM memories ORDER BY timestamp DESC LIMIT ?", (max(1, int(count)),)).fetchall()
        return [_to_public(self._row_to_item(row)) for row in rows]

    def stats(self) -> dict[str, Any]:
        count, latest = self._connection.execute("SELECT COUNT(*), MAX(timestamp) FROM memories").fetchone()
        by_type = dict(self._connection.execute("SELECT type, COUNT(*) FROM memories GROUP BY type").fetchall())
        return {"count": count, "by_type": by_type, "latest_timestamp": latest, "min_score_default": self.min_score,
                "backend": "sqlite-lexical", "mode_label": "sqlite lexical", "backup_revisions": 0,
                "storage_format": "sqlite-v1", "storage_root": str(self.storage_path.parent), "load_recovery_count": 0}

    def export_txt(self, path: str) -> str:
        target = Path(os.path.expanduser(path)).resolve()
        lines = ["# Alphanus Memory Export", ""]
        for item in reversed(self.memories):
            lines.extend([f"- id: {item.id}", f"  type: {item.type}", f"  timestamp: {item.timestamp}", f"  text: {item.text}", ""])
        target.parent.mkdir(parents=True, exist_ok=True)
        from core.secure_io import atomic_write_text
        atomic_write_text(target, "\n".join(lines), mode=0o600)
        return str(target)

    def flush(self) -> None:
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()
