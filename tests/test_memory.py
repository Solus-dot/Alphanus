from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from core.memory import VectorMemory


def _fake_encode(self, text: str):
    vec = np.zeros(16, dtype=np.float32)
    for token in text.lower().split():
        vec[hash(token) % 16] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


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
