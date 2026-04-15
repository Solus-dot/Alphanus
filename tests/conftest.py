from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from core.memory import VectorMemory
from core.workspace import WorkspaceManager


class _TestSentenceTransformer:
    dim = 48

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


@pytest.fixture(autouse=True)
def _disable_model_classification():
    """Disable model-based classification unless a test opts in."""
    with patch(
        "agent.classifier.TurnClassifier._should_model_classify",
        return_value=False,
    ):
        yield


@pytest.fixture(autouse=True)
def _stub_memory_transformer_loader(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        VectorMemory,
        "_load_transformer_encoder",
        lambda self, _model_name: _TestSentenceTransformer(),
        raising=True,
    )
    yield


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceManager:
    home = tmp_path / "home"
    ws = home / "workspace"
    home.mkdir(parents=True)
    ws.mkdir(parents=True)
    return WorkspaceManager(workspace_root=str(ws), home_root=str(home))


@pytest.fixture
def memory(tmp_path: Path) -> VectorMemory:
    path = tmp_path / "memory.pkl"
    return VectorMemory(storage_path=str(path))
