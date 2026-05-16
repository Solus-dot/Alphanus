from __future__ import annotations

from typing import Final

AUTO_BACKEND_PROFILE: Final[str] = "auto"
UNKNOWN_BACKEND_PROFILE: Final[str] = "unknown"
LOCAL_BACKEND_PROFILES: Final[frozenset[str]] = frozenset({"mlx_vlm", "llamacpp", "ollama", "vllm", "lmstudio"})
VALID_BACKEND_PROFILES: Final[frozenset[str]] = frozenset({AUTO_BACKEND_PROFILE, *LOCAL_BACKEND_PROFILES})


def normalize_backend_profile(value: object) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in VALID_BACKEND_PROFILES else AUTO_BACKEND_PROFILE
