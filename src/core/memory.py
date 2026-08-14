from __future__ import annotations

import json
import math
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.retrieval import SQLiteRetrievalStore
from core.secure_io import atomic_write_text


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
        parsed = default
    return max(0.0, min(1.0, parsed if math.isfinite(parsed) else default))


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
    """Memory API backed by the shared SQLite retrieval store."""

    def __init__(
        self,
        storage_path: str,
        min_score: float = 0.3,
        persist_access_updates: bool = False,
        legacy_path: str | None = None,
    ) -> None:
        requested = Path(os.path.expanduser(storage_path)).resolve()
        requested.parent.mkdir(parents=True, exist_ok=True)
        if requested.exists() and requested.suffix not in {".db", ".sqlite"}:
            raise ValueError(f"Legacy unversioned memory found at {requested}. Alphanus v1 does not migrate it; export or remove it first.")
        self.storage_path = requested if requested.suffix in {".db", ".sqlite"} else requested.parent / "memory.db"
        self.min_score = _normalize_threshold(min_score, default=0.3)
        self.persist_access_updates = bool(persist_access_updates)
        self.store = SQLiteRetrievalStore(self.storage_path)
        self._migrate_legacy_table(Path(legacy_path).resolve() if legacy_path else self.storage_path)

    def _migrate_legacy_table(self, legacy_path: Path) -> None:
        if not legacy_path.exists():
            return
        existing = {str(record.get("source") or "") for record in self.store.list_records("memory_fact", limit=10_000)}
        with sqlite3.connect(legacy_path) as connection:
            exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'").fetchone()
            rows = (
                connection.execute("SELECT id,text,metadata_json,type,timestamp,access_count,last_accessed FROM memories").fetchall()
                if exists
                else []
            )

        for memory_id, text, metadata_json, memory_type, timestamp, access_count, last_accessed in rows:
            source = f"memory:{memory_id}"
            if source in existing:
                continue
            try:
                user_metadata = json.loads(metadata_json)
            except (TypeError, json.JSONDecodeError):
                user_metadata = {}
            self.store.upsert_record(
                record_type="memory_fact",
                source=source,
                canonical_source=source,
                title=memory_type,
                text=text,
                metadata={
                    "memory_type": memory_type,
                    "timestamp": timestamp,
                    "access_count": access_count,
                    "last_accessed": last_accessed,
                    "metadata": user_metadata if isinstance(user_metadata, dict) else {},
                },
            )

    @staticmethod
    def _item(record: dict[str, Any]) -> MemoryItem:
        raw_metadata = record.get("metadata")
        metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
        return MemoryItem(
            id=int(record["record_id"]),
            text=str(record["text"]),
            metadata=dict(metadata.get("metadata") or {}),
            type=str(metadata.get("memory_type") or record.get("title") or "conversation"),
            timestamp=float(metadata.get("timestamp") or record.get("fetched_at") or 0),
            access_count=int(metadata.get("access_count") or 0),
            last_accessed=float(metadata.get("last_accessed") or 0),
        )

    @property
    def memories(self) -> list[MemoryItem]:
        return [self._item(record) for record in self.store.list_records("memory_fact", limit=10_000)]

    def add_memory(
        self,
        text: str,
        memory_type: str = "conversation",
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        public_metadata = dict(metadata or {})
        if importance is not None:
            public_metadata["importance"] = float(importance)
        record = self.store.upsert_record(
            record_type="memory_fact",
            source=f"memory:{uuid.uuid4().hex}",
            title=memory_type,
            text=str(text),
            metadata={
                "memory_type": memory_type,
                "timestamp": now,
                "access_count": 0,
                "last_accessed": now,
                "metadata": public_metadata,
            },
        )
        if record is None:
            raise ValueError("Memory text must not be empty")
        return _to_public(MemoryItem(record.id, str(text), public_metadata, memory_type, now, 0, now))

    def search(
        self,
        query: str,
        top_k: int = 5,
        memory_type: str | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        threshold = self.min_score if min_score is None else _normalize_threshold(min_score, default=self.min_score)
        records = self.store.search(query, top_k=max(1, int(top_k)) * 8, sources=["memory_fact"])
        selected: list[dict[str, Any]] = []
        now = time.time()
        for record in records:
            item = self._item(record)
            score = float(record.get("score") or 0)
            if score < threshold or memory_type and item.type != memory_type:
                continue
            item.access_count += 1
            item.last_accessed = now
            if self.persist_access_updates:
                metadata = dict(record.get("metadata") or {})
                metadata.update(access_count=item.access_count, last_accessed=now)
                self.store.update_metadata(item.id, metadata)
            result = _to_public(item)
            result["score"] = round(score, 4)
            selected.append(result)
            if len(selected) >= max(1, int(top_k)):
                break
        return selected

    def forget(self, memory_id: int) -> bool:
        return self.store.forget(int(memory_id))

    def list_recent(self, count: int = 5) -> list[dict[str, Any]]:
        return [_to_public(item) for item in self.memories[: max(1, int(count))]]

    def stats(self) -> dict[str, Any]:
        stats = self.store.stats()
        return {
            "count": int(stats["by_type"].get("memory_fact", 0)),
            "backend": "sqlite-lexical",
            "mode_label": "sqlite lexical",
            "min_score_default": self.min_score,
        }

    def export_txt(self, path: str) -> str:
        target = Path(os.path.expanduser(path)).resolve()
        lines = ["# Alphanus Memory Export", ""]
        for item in reversed(self.memories):
            lines.extend([f"- id: {item.id}", f"  type: {item.type}", f"  timestamp: {item.timestamp}", f"  text: {item.text}", ""])
        atomic_write_text(target, "\n".join(lines), mode=0o600)
        return str(target)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
