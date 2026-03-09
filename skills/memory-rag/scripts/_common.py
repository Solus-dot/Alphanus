from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.memory import VectorMemory  # noqa: E402
from core.tool_script import emit, read_args  # noqa: E402


def load_memory() -> VectorMemory:
    storage_path = os.getenv("ALPHANUS_MEMORY_PATH", "").strip()
    if not storage_path:
        raise ValueError("ALPHANUS_MEMORY_PATH is required")
    model_name = os.getenv("ALPHANUS_MEMORY_MODEL", "all-MiniLM-L6-v2")
    backend = os.getenv("ALPHANUS_MEMORY_BACKEND", "hash")
    eager = os.getenv("ALPHANUS_MEMORY_EAGER_LOAD", "0") == "1"
    return VectorMemory(
        storage_path=storage_path,
        model_name=model_name,
        embedding_backend=backend,
        eager_load_encoder=eager,
    )


__all__ = ["emit", "read_args", "load_memory"]
