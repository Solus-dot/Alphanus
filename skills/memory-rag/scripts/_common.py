from __future__ import annotations

import os

from core.memory import VectorMemory
from core.tool_script import emit, read_args


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
