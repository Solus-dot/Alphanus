AUTO_BACKEND_PROFILE = "auto"
UNKNOWN_BACKEND_PROFILE = "unknown"
LOCAL_BACKEND_PROFILES = frozenset({"mlx_vlm", "llamacpp", "ollama", "vllm", "lmstudio"})
VALID_BACKEND_PROFILES = frozenset({AUTO_BACKEND_PROFILE, *LOCAL_BACKEND_PROFILES})
BACKEND_PROFILE_LABELS = {
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
