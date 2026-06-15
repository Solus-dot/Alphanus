from __future__ import annotations

from typing import Final

AUTO_BACKEND_PROFILE: Final[str] = "auto"
UNKNOWN_BACKEND_PROFILE: Final[str] = "unknown"
LOCAL_BACKEND_PROFILES: Final[frozenset[str]] = frozenset({"mlx_vlm", "llamacpp", "ollama", "vllm", "lmstudio"})
VALID_BACKEND_PROFILES: Final[frozenset[str]] = frozenset({AUTO_BACKEND_PROFILE, *LOCAL_BACKEND_PROFILES})
BACKEND_PROFILE_LABELS: Final[dict[str, str]] = {
    AUTO_BACKEND_PROFILE: "detect backend and apply compatibility rewrites",
    "mlx_vlm": "MLX-VLM tuned multimodal compatibility",
    "llamacpp": "llama.cpp-style local backend",
    "ollama": "Ollama OpenAI-compatible backend",
    "vllm": "vLLM OpenAI-compatible backend",
    "lmstudio": "LM Studio OpenAI-compatible backend",
}


def normalize_backend_profile(value: object) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in VALID_BACKEND_PROFILES else AUTO_BACKEND_PROFILE
