import json
import math
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

MEMORY_STORAGE_SCHEMA_VERSION = "3.0.0"


@dataclass(slots=True)
class MemoryItem:
    id: int
    text: str
    metadata: Dict[str, Any]
    type: str
    timestamp: float
    access_count: int
    last_accessed: float


class LexicalMemory:
    def __init__(
        self,
        storage_path: str,
        min_score: float = 0.3,
        persist_access_updates: bool = False,
        autosave_interval_s: float = 2.0,
        autosave_every: int = 24,
        backup_revisions: int = 2,
        **_ignored: Any,
    ) -> None:
        self.storage_path = Path(os.path.expanduser(storage_path)).resolve()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.min_score = self._normalize_threshold(min_score, default=0.3)
        self.persist_access_updates = bool(persist_access_updates)
        self.autosave_interval_s = max(0.0, float(autosave_interval_s))
        self.autosave_every = max(1, int(autosave_every))
        self.backup_revisions = max(0, int(backup_revisions))

        self.memories: List[MemoryItem] = []
        self._next_id = 1
        self._dirty = False
        self._pending_writes = 0
        self._last_save_ts = time.time()
        self._load_recovery_count = 0
        self._load_unsupported_count = 0

        self._load()

    @property
    def facts_path(self) -> Path:
        return self.storage_path.parent / "facts.md"

    @staticmethod
    def _normalize_threshold(value: Any, *, default: float) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = float(default)
        if not math.isfinite(parsed):
            parsed = float(default)
        return max(0.0, min(1.0, parsed))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", str(text).lower())

    @staticmethod
    def _contains_phrase(query_tokens: List[str], text_tokens: List[str]) -> bool:
        if not query_tokens or not text_tokens or len(query_tokens) > len(text_tokens):
            return False
        q_len = len(query_tokens)
        for idx in range(0, len(text_tokens) - q_len + 1):
            if text_tokens[idx : idx + q_len] == query_tokens:
                return True
        return False

    def _score_text(self, query_tokens: List[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        text_tokens = self._tokenize(text)
        if not text_tokens:
            return 0.0
        query_set = set(query_tokens)
        text_set = set(text_tokens)
        overlap = len(query_set & text_set) / max(1, len(query_set))
        phrase_bonus = 0.3 if self._contains_phrase(query_tokens, text_tokens) else 0.0
        return min(1.0, (0.75 * overlap) + phrase_bonus)

    def _backup_path(self, revision: int) -> Path:
        return self.storage_path.with_suffix(self.storage_path.suffix + f".bak{revision}")

    def _rotate_backups(self) -> None:
        if self.backup_revisions <= 0 or not self.storage_path.exists():
            return
        for idx in range(self.backup_revisions, 1, -1):
            dest = self._backup_path(idx)
            src = self._backup_path(idx - 1)
            if not src.exists():
                continue
            if dest.exists():
                dest.unlink()
            src.replace(dest)
        bak1 = self._backup_path(1)
        if bak1.exists():
            bak1.unlink()
        shutil.copy2(self.storage_path, bak1)

    @staticmethod
    def _item_to_record(item: MemoryItem) -> Dict[str, Any]:
        return {
            "schema_version": MEMORY_STORAGE_SCHEMA_VERSION,
            "id": item.id,
            "text": item.text,
            "metadata": item.metadata,
            "type": item.type,
            "timestamp": item.timestamp,
            "access_count": item.access_count,
            "last_accessed": item.last_accessed,
        }

    @staticmethod
    def _record_to_item(record: Dict[str, Any]) -> MemoryItem:
        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("memory metadata must be an object")
        return MemoryItem(
            id=int(record["id"]),
            text=str(record["text"]),
            metadata=dict(metadata),
            type=str(record.get("type", "conversation")),
            timestamp=float(record.get("timestamp", time.time())),
            access_count=int(record.get("access_count", 0)),
            last_accessed=float(record.get("last_accessed", record.get("timestamp", time.time()))),
        )

    def _load(self) -> None:
        if not self.storage_path.exists():
            return

        loaded_by_id: Dict[int, MemoryItem] = {}
        ordered_ids: List[int] = []

        try:
            handle = self.storage_path.open("r", encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            self.memories = []
            self._next_id = 1
            self._load_recovery_count += 1
            return

        try:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    self._load_recovery_count += 1
                    continue
                if not isinstance(record, dict):
                    self._load_recovery_count += 1
                    continue

                schema_version = str(record.get("schema_version", "")).strip()
                if schema_version != MEMORY_STORAGE_SCHEMA_VERSION:
                    self._load_unsupported_count += 1
                    continue

                try:
                    item = self._record_to_item(record)
                except (KeyError, TypeError, ValueError):
                    self._load_recovery_count += 1
                    continue

                if item.id not in loaded_by_id:
                    ordered_ids.append(item.id)
                loaded_by_id[item.id] = item
        except (OSError, UnicodeDecodeError):
            self.memories = []
            self._next_id = 1
            self._load_recovery_count += 1
            return
        finally:
            handle.close()

        self.memories = [loaded_by_id[memory_id] for memory_id in ordered_ids if memory_id in loaded_by_id]
        self._next_id = max((m.id for m in self.memories), default=0) + 1
        self._dirty = False
        self._pending_writes = 0
        self._last_save_ts = time.time()

    def _write_facts(self) -> None:
        lines = ["# Alphanus Memory Facts", ""]
        for item in sorted(self.memories, key=lambda m: (m.timestamp, m.id)):
            lines.append(f"- id: {item.id}")
            lines.append(f"  type: {item.type}")
            lines.append(f"  timestamp: {item.timestamp}")
            lines.append(f"  access_count: {item.access_count}")
            lines.append(f"  last_accessed: {item.last_accessed}")
            lines.append(f"  metadata: {json.dumps(item.metadata, sort_keys=True, ensure_ascii=False)}")
            lines.append(f"  text: {item.text}")
            lines.append("")

        fd, tmp = tempfile.mkstemp(prefix=self.facts_path.name + ".", dir=str(self.facts_path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines))
            os.replace(tmp, self.facts_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _save(self) -> None:
        fd, tmp = tempfile.mkstemp(prefix=self.storage_path.name + ".", dir=str(self.storage_path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for item in self.memories:
                    record = self._item_to_record(item)
                    handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False))
                    handle.write("\n")
            # Snapshot the currently committed primary file into backups first,
            # then atomically promote the new snapshot.
            self._rotate_backups()
            os.replace(tmp, self.storage_path)
            self._write_facts()
            self._dirty = False
            self._pending_writes = 0
            self._last_save_ts = time.time()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _mark_dirty(self, force: bool = False) -> None:
        self._dirty = True
        self._pending_writes += 1
        if force:
            self._save()
            return
        if self._pending_writes >= self.autosave_every:
            self._save()
            return
        if self.autosave_interval_s > 0 and (time.time() - self._last_save_ts) >= self.autosave_interval_s:
            self._save()

    def flush(self) -> None:
        if self._dirty:
            self._save()

    def add_memory(
        self,
        text: str,
        memory_type: str = "conversation",
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        md = dict(metadata or {})
        if importance is not None:
            md["importance"] = float(importance)

        item = MemoryItem(
            id=self._next_id,
            text=text,
            metadata=md,
            type=memory_type,
            timestamp=now,
            access_count=0,
            last_accessed=now,
        )
        self._next_id += 1
        self.memories.append(item)
        self._mark_dirty(force=False)
        return self._to_public(item)

    def search(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not self.memories:
            return []

        threshold = self.min_score if min_score is None else self._normalize_threshold(min_score, default=self.min_score)
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored: List[tuple[float, MemoryItem]] = []
        for item in self.memories:
            if memory_type and item.type != memory_type:
                continue
            score = self._score_text(query_tokens, item.text)
            if score >= threshold:
                scored.append((score, item))

        if not scored:
            return []

        scored.sort(key=lambda row: (row[0], row[1].timestamp), reverse=True)
        selected = scored[: max(1, int(top_k))]

        now = time.time()
        touched = False
        out: List[Dict[str, Any]] = []
        for score, item in selected:
            item.access_count += 1
            item.last_accessed = now
            touched = True
            record = self._to_public(item)
            record["score"] = round(float(score), 4)
            out.append(record)

        if touched and self.persist_access_updates:
            self._mark_dirty(force=False)
        return out

    def list_recent(self, count: int = 5) -> List[Dict[str, Any]]:
        ordered = sorted(self.memories, key=lambda m: m.timestamp, reverse=True)
        return [self._to_public(item) for item in ordered[: max(1, count)]]

    def forget(self, memory_id: int) -> bool:
        target_id = int(memory_id)
        memory_idx = next((idx for idx, item in enumerate(self.memories) if item.id == target_id), -1)
        if memory_idx < 0:
            return False

        self.memories.pop(memory_idx)
        self._mark_dirty(force=False)
        return True

    def stats(self) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        for item in self.memories:
            by_type[item.type] = by_type.get(item.type, 0) + 1

        latest = max((item.timestamp for item in self.memories), default=None)
        return {
            "count": len(self.memories),
            "by_type": by_type,
            "latest_timestamp": latest,
            "min_score_default": self.min_score,
            "backend": "lexical",
            "mode_label": "lexical",
            "backup_revisions": self.backup_revisions,
            "memory_schema_version": MEMORY_STORAGE_SCHEMA_VERSION,
            "storage_format": "jsonl",
            "storage_root": str(self.storage_path.parent),
            "load_recovery_count": self._load_recovery_count,
            "load_unsupported_count": self._load_unsupported_count,
        }

    def export_txt(self, path: str) -> str:
        target = Path(os.path.expanduser(path)).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Alphanus Memory Export", ""]
        for item in sorted(self.memories, key=lambda m: m.timestamp):
            lines.append(f"- id: {item.id}")
            lines.append(f"  type: {item.type}")
            lines.append(f"  timestamp: {item.timestamp}")
            lines.append(f"  access_count: {item.access_count}")
            lines.append(f"  text: {item.text}")
            lines.append("")
        target.write_text("\n".join(lines), encoding="utf-8")
        return str(target)

    @staticmethod
    def _to_public(item: MemoryItem) -> Dict[str, Any]:
        return {
            "id": item.id,
            "text": item.text,
            "metadata": item.metadata,
            "type": item.type,
            "timestamp": item.timestamp,
            "access_count": item.access_count,
            "last_accessed": item.last_accessed,
        }

    def __del__(self) -> None:
        try:
            self.flush()
        except Exception:
            pass
