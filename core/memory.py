import os
import pickle
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

RECOMMENDED_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_LEGACY_EMBEDDING_BACKENDS = {"hash", "transformer", "auto"}


class MemoryEncoderUnavailableError(ValueError):
    """Raised when semantic memory operations need embeddings but no encoder is available."""


@dataclass(slots=True)
class MemoryItem:
    id: int
    text: str
    vector: np.ndarray
    metadata: Dict[str, Any]
    type: str
    timestamp: float
    access_count: int
    last_accessed: float


class VectorMemory:
    def __init__(
        self,
        storage_path: str,
        model_name: str = RECOMMENDED_EMBEDDING_MODEL_NAME,
        embedding_backend: str = "transformer",
        min_score: float = 0.3,
        persist_access_updates: bool = False,
        autosave_interval_s: float = 2.0,
        autosave_every: int = 24,
        eager_load_encoder: bool = False,
        allow_model_download: bool = True,
    ) -> None:
        self.storage_path = Path(os.path.expanduser(storage_path)).resolve()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        backend = self._normalize_backend(embedding_backend, default="transformer")
        if backend == "hash":
            raise ValueError("The hash memory backend has been removed; use transformer embeddings.")
        if backend not in {"transformer", "auto"}:
            raise ValueError(f"Unsupported memory embedding backend: {backend}")
        self.embedding_backend = "transformer"

        self.min_score = float(min_score)
        self.persist_access_updates = bool(persist_access_updates)
        self.autosave_interval_s = max(0.0, float(autosave_interval_s))
        self.autosave_every = max(1, int(autosave_every))
        self.eager_load_encoder = bool(eager_load_encoder)
        self.allow_model_download = bool(allow_model_download)

        self.memories: List[MemoryItem] = []
        self._next_id = 1
        self._dirty = False
        self._pending_writes = 0
        self._last_save_ts = 0.0
        self._matrix_cache: Optional[np.ndarray] = None
        self._norm_cache: Optional[np.ndarray] = None
        self._type_index_cache: Dict[str, np.ndarray] = {}
        self._all_index_cache: Optional[np.ndarray] = None
        self._embedding_space_ready = False
        self._stored_embedding_backend: Optional[str] = None
        self._stored_model_name: Optional[str] = None
        self._encoder_status = "uninitialized"
        self._encoder_source = ""
        self._encoder_detail = ""
        self._encoder_attempted = False

        self.encoder = None
        self.dimension = 384
        self._load()
        if self.eager_load_encoder:
            self._ensure_encoder()

    @staticmethod
    def _normalize_backend(value: Any, default: str) -> str:
        backend = str(value or default).strip().lower()
        if backend not in _LEGACY_EMBEDDING_BACKENDS:
            return default
        return backend

    def _invalidate_index_cache(self) -> None:
        self._matrix_cache = None
        self._norm_cache = None
        self._type_index_cache = {}
        self._all_index_cache = None

    def _set_encoder_state(self, status: str, source: str, detail: str = "") -> None:
        self._encoder_status = status
        self._encoder_source = source
        self._encoder_detail = detail.strip()

    def _ensure_index_cache(self) -> None:
        if self._matrix_cache is not None and self._norm_cache is not None and self._all_index_cache is not None:
            return

        if not self.memories:
            self._matrix_cache = np.empty((0, self.dimension), dtype=np.float32)
            self._norm_cache = np.empty((0,), dtype=np.float32)
            self._all_index_cache = np.empty((0,), dtype=np.int32)
            self._type_index_cache = {}
            return

        matrix = np.asarray([item.vector for item in self.memories], dtype=np.float32)
        self._matrix_cache = matrix
        self._norm_cache = np.linalg.norm(matrix, axis=1).astype(np.float32, copy=False)
        self._all_index_cache = np.arange(len(self.memories), dtype=np.int32)

        typed: Dict[str, List[int]] = {}
        for idx, item in enumerate(self.memories):
            typed.setdefault(item.type, []).append(idx)
        self._type_index_cache = {
            key: np.asarray(indexes, dtype=np.int32) for key, indexes in typed.items()
        }

    def _load_transformer_encoder(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:
            self._set_encoder_state(
                "unavailable",
                "transformer-import",
                f"sentence-transformers unavailable: {exc.__class__.__name__}",
            )
            return None

        try:
            model = SentenceTransformer(model_name, local_files_only=True)
            self._set_encoder_state("ready", "transformer-local", "loaded from local cache")
            return model
        except Exception as local_exc:
            if not self.allow_model_download:
                self._set_encoder_state(
                    "unavailable",
                    "transformer-local",
                    f"model not cached locally and allow_model_download=false: {local_exc.__class__.__name__}",
                )
                return None
        try:
            model = SentenceTransformer(model_name, local_files_only=False)
            self._set_encoder_state("ready", "transformer-download", "loaded with download fallback enabled")
            return model
        except Exception as remote_exc:
            self._set_encoder_state(
                "unavailable",
                "transformer-download",
                f"transformer load failed: {remote_exc.__class__.__name__}",
            )
            return None

    def _ensure_encoder(self):
        if self.encoder is not None:
            return self.encoder
        self._encoder_attempted = True
        encoder = self._load_transformer_encoder(self.model_name)
        if encoder is not None:
            self.encoder = encoder
            self.dimension = int(getattr(self.encoder, "dim", 384))
            if self._encoder_status == "uninitialized":
                self._set_encoder_state("ready", "transformer", "semantic transformer encoder")
        else:
            self._encoder_attempted = False
        return self.encoder

    def _require_encoder(self):
        encoder = self._ensure_encoder()
        if encoder is None:
            detail = self._encoder_detail or "sentence-transformers could not be initialized"
            raise MemoryEncoderUnavailableError(f"Memory embeddings are unavailable: {detail}")
        return encoder

    def _encode_many(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        encoder = self._require_encoder()
        emb = encoder.encode(texts, normalize_embeddings=True)
        matrix = np.asarray(emb, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError("Encoder returned an invalid embedding matrix")
        self.dimension = int(matrix.shape[1])
        return matrix

    def _encode(self, text: str) -> np.ndarray:
        return self._encode_many([text])[0]

    def _memories_have_consistent_dimensions(self) -> bool:
        if not self.memories:
            return True
        dims = {
            int(item.vector.shape[0])
            for item in self.memories
            if isinstance(item.vector, np.ndarray) and item.vector.ndim == 1
        }
        if len(dims) != len(self.memories):
            return False
        return len(dims) == 1

    def _stored_backend_matches_current_space(self) -> bool:
        stored_backend = self._normalize_backend(self._stored_embedding_backend or "", default="")
        if stored_backend != "transformer":
            return False
        return self._stored_model_name == self.model_name

    def _needs_reembed(self) -> bool:
        if not self.memories:
            return False
        if not self._memories_have_consistent_dimensions():
            return True
        current_dim = int(self.memories[0].vector.shape[0])
        if current_dim != self.dimension:
            return True
        return not self._stored_backend_matches_current_space()

    def _ensure_embedding_space(self) -> None:
        if self._embedding_space_ready:
            return
        self._require_encoder()
        if not self.memories:
            self._stored_embedding_backend = "transformer"
            self._stored_model_name = self.model_name
            self._embedding_space_ready = True
            return
        if self._needs_reembed():
            vectors = self._encode_many([item.text for item in self.memories])
            for item, vector in zip(self.memories, vectors):
                item.vector = vector
            self._invalidate_index_cache()
            self._mark_dirty(force=False)
        else:
            self.dimension = int(self.memories[0].vector.shape[0])
        self._stored_embedding_backend = "transformer"
        self._stored_model_name = self.model_name
        self._embedding_space_ready = True

    def _storage_metadata(self) -> tuple[str, str]:
        if self._embedding_space_ready:
            return ("transformer", self.model_name)
        stored_backend = self._normalize_backend(self._stored_embedding_backend or "", default="")
        if stored_backend == "hash":
            return ("hash", self._stored_model_name or "")
        if stored_backend == "transformer":
            return ("transformer", self._stored_model_name or self.model_name)
        return ("transformer", self.model_name)

    def _append_to_index_cache(self, item: MemoryItem) -> bool:
        if self._matrix_cache is None or self._norm_cache is None or self._all_index_cache is None:
            return False
        if self._matrix_cache.ndim != 2 or item.vector.ndim != 1:
            return False
        if self._matrix_cache.shape[1] != item.vector.shape[0]:
            return False

        row = item.vector.reshape(1, -1)
        norm = np.asarray([np.linalg.norm(item.vector)], dtype=np.float32)
        new_index = np.asarray([len(self.memories) - 1], dtype=np.int32)
        self._matrix_cache = np.concatenate((self._matrix_cache, row), axis=0)
        self._norm_cache = np.concatenate((self._norm_cache, norm))
        self._all_index_cache = np.concatenate((self._all_index_cache, new_index))

        existing = self._type_index_cache.get(item.type)
        if existing is None:
            self._type_index_cache[item.type] = new_index
        else:
            self._type_index_cache[item.type] = np.concatenate((existing, new_index))
        return True

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with self.storage_path.open("rb") as handle:
                payload = pickle.load(handle)
        except (EOFError, pickle.UnpicklingError, AttributeError, ValueError):
            broken = self.storage_path.with_suffix(self.storage_path.suffix + ".corrupted")
            self.storage_path.replace(broken)
            self.memories = []
            self._next_id = 1
            return

        if isinstance(payload, dict):
            self._stored_embedding_backend = self._normalize_backend(payload.get("embedding_backend", ""), default="")
            raw_model_name = str(payload.get("model_name", "")).strip()
            self._stored_model_name = raw_model_name or None
        raw_memories = payload.get("memories", []) if isinstance(payload, dict) else []
        loaded: List[MemoryItem] = []
        for item in raw_memories:
            loaded.append(
                MemoryItem(
                    id=int(item["id"]),
                    text=str(item["text"]),
                    vector=np.asarray(item["vector"], dtype=np.float32),
                    metadata=dict(item.get("metadata", {})),
                    type=str(item.get("type", "conversation")),
                    timestamp=float(item.get("timestamp", time.time())),
                    access_count=int(item.get("access_count", 0)),
                    last_accessed=float(item.get("last_accessed", item.get("timestamp", time.time()))),
                )
            )

        self.memories = loaded
        if loaded:
            self.dimension = int(loaded[0].vector.shape[0])
        self._next_id = max((m.id for m in loaded), default=0) + 1
        self._dirty = False
        self._pending_writes = 0
        self._last_save_ts = time.time()
        self._embedding_space_ready = False
        self._invalidate_index_cache()

    def _save(self) -> None:
        storage_backend, storage_model_name = self._storage_metadata()
        payload = {
            "schema_version": "1.0.0",
            "model_name": storage_model_name,
            "embedding_backend": storage_backend,
            "memories": [
                {
                    "id": m.id,
                    "text": m.text,
                    "vector": m.vector,
                    "metadata": m.metadata,
                    "type": m.type,
                    "timestamp": m.timestamp,
                    "access_count": m.access_count,
                    "last_accessed": m.last_accessed,
                }
                for m in self.memories
            ],
        }

        fd, tmp = tempfile.mkstemp(prefix=self.storage_path.name + ".", dir=str(self.storage_path.parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, self.storage_path)
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
            if self.memories:
                try:
                    self._ensure_embedding_space()
                except MemoryEncoderUnavailableError:
                    pass
            self._save()

    def add_memory(
        self,
        text: str,
        memory_type: str = "conversation",
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
    ) -> Dict[str, Any]:
        if self.memories:
            self._ensure_embedding_space()
        now = time.time()
        md = dict(metadata or {})
        if importance is not None:
            md["importance"] = float(importance)
        vector = np.asarray(self._encode(text), dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("Encoder returned an invalid embedding vector")
        self.dimension = int(vector.shape[0])

        item = MemoryItem(
            id=self._next_id,
            text=text,
            vector=vector,
            metadata=md,
            type=memory_type,
            timestamp=now,
            access_count=0,
            last_accessed=now,
        )
        self._next_id += 1
        self.memories.append(item)
        self._stored_embedding_backend = "transformer"
        self._stored_model_name = self.model_name
        self._embedding_space_ready = True
        if not self._append_to_index_cache(item):
            self._invalidate_index_cache()
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

        self._ensure_embedding_space()
        threshold = self.min_score if min_score is None else float(min_score)
        q = self._encode(query)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0:
            return []

        self._ensure_index_cache()
        if memory_type:
            candidate_indexes = self._type_index_cache.get(memory_type)
        else:
            candidate_indexes = self._all_index_cache
        if candidate_indexes is None or candidate_indexes.size == 0 or self._matrix_cache is None or self._norm_cache is None:
            return []

        matrix = self._matrix_cache[candidate_indexes]
        vec_norms = self._norm_cache[candidate_indexes]
        denom = vec_norms * q_norm
        dots = matrix @ q
        scores = np.divide(
            dots,
            denom,
            out=np.zeros_like(dots, dtype=np.float32),
            where=denom > 0,
        )

        passing = np.flatnonzero(scores >= threshold)
        if passing.size == 0:
            return []

        k = max(1, int(top_k))
        if passing.size > k:
            passing_scores = scores[passing]
            top_local = np.argpartition(passing_scores, -k)[-k:]
            ordered_local = top_local[np.argsort(passing_scores[top_local])[::-1]]
            selected = passing[ordered_local]
        else:
            selected = passing[np.argsort(scores[passing])[::-1]]

        now = time.time()
        touched = False
        out = []
        for local_idx in selected:
            item = self.memories[int(candidate_indexes[int(local_idx)])]
            score = float(scores[int(local_idx)])
            item.access_count += 1
            item.last_accessed = now
            touched = True
            record = self._to_public(item)
            record["score"] = round(score, 4)
            out.append(record)

        if touched and self.persist_access_updates:
            self._mark_dirty(force=False)
        return out

    def list_recent(self, count: int = 5) -> List[Dict[str, Any]]:
        ordered = sorted(self.memories, key=lambda m: m.timestamp, reverse=True)
        return [self._to_public(item) for item in ordered[: max(1, count)]]

    def forget(self, memory_id: int) -> bool:
        before = len(self.memories)
        self.memories = [m for m in self.memories if m.id != int(memory_id)]
        changed = len(self.memories) != before
        if changed:
            self._embedding_space_ready = False
            self._invalidate_index_cache()
            if self.memories and self.encoder is not None:
                self._ensure_embedding_space()
            self._mark_dirty(force=False)
        return changed

    def stats(self) -> Dict[str, Any]:
        self._ensure_encoder()
        if self.memories and self.encoder is not None:
            self._ensure_embedding_space()
        by_type: Dict[str, int] = {}
        for m in self.memories:
            by_type[m.type] = by_type.get(m.type, 0) + 1

        latest = max((m.timestamp for m in self.memories), default=None)
        return {
            "count": len(self.memories),
            "by_type": by_type,
            "latest_timestamp": latest,
            "dimension": self.dimension,
            "model_name": self.model_name,
            "embedding_backend": self.embedding_backend,
            "allow_model_download": self.allow_model_download,
            "encoder_status": self._encoder_status,
            "encoder_source": self._encoder_source,
            "encoder_detail": self._encoder_detail,
            "mode_label": "semantic" if self._encoder_status == "ready" else "semantic-unavailable",
            "recommended_model_name": RECOMMENDED_EMBEDDING_MODEL_NAME,
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
