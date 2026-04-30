from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

AUTO_BACKEND_PROFILE = "auto"
UNKNOWN_BACKEND_PROFILE = "unknown"
LOCAL_BACKEND_PROFILES = {"mlx_vlm", "llamacpp", "ollama", "vllm", "lmstudio"}
VALID_BACKEND_PROFILES = {AUTO_BACKEND_PROFILE, *LOCAL_BACKEND_PROFILES}


@dataclass(slots=True)
class BackendCapabilities:
    supports_chat: bool = True
    supports_responses: bool = True
    supports_tools: bool = True
    supports_multimodal_input: bool = True
    strip_stream_options: bool = False
    strip_chat_template_kwargs: bool = False
    flatten_chat_image_url: bool = False
    responses_input_blocks: bool = False
    strict_model_integrity: bool = False

    def to_json(self) -> dict[str, object]:
        return {
            "supports_chat": self.supports_chat,
            "supports_responses": self.supports_responses,
            "supports_tools": self.supports_tools,
            "supports_multimodal_input": self.supports_multimodal_input,
            "strip_stream_options": self.strip_stream_options,
            "strip_chat_template_kwargs": self.strip_chat_template_kwargs,
            "flatten_chat_image_url": self.flatten_chat_image_url,
            "responses_input_blocks": self.responses_input_blocks,
            "strict_model_integrity": self.strict_model_integrity,
        }


def normalize_backend_profile(value: object) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in VALID_BACKEND_PROFILES else AUTO_BACKEND_PROFILE


def profile_capabilities(profile: str) -> BackendCapabilities:
    normalized = str(profile or "").strip().lower()
    if normalized == "mlx_vlm":
        return BackendCapabilities(
            supports_chat=True,
            supports_responses=True,
            supports_tools=True,
            supports_multimodal_input=True,
            strip_stream_options=True,
            strip_chat_template_kwargs=True,
            flatten_chat_image_url=True,
            responses_input_blocks=True,
            strict_model_integrity=True,
        )
    if normalized == "llamacpp":
        return BackendCapabilities(
            supports_chat=True,
            supports_responses=False,
            supports_tools=True,
            supports_multimodal_input=True,
            strip_stream_options=False,
            strip_chat_template_kwargs=True,
            strict_model_integrity=True,
        )
    if normalized == "ollama":
        return BackendCapabilities(
            supports_chat=True,
            supports_responses=False,
            supports_tools=True,
            supports_multimodal_input=True,
            strip_stream_options=True,
            strip_chat_template_kwargs=True,
            strict_model_integrity=False,
        )
    if normalized == "vllm":
        return BackendCapabilities(
            supports_chat=True,
            supports_responses=False,
            supports_tools=True,
            supports_multimodal_input=True,
            strip_stream_options=False,
            strip_chat_template_kwargs=True,
            strict_model_integrity=False,
        )
    if normalized == "lmstudio":
        return BackendCapabilities(
            supports_chat=True,
            supports_responses=False,
            supports_tools=True,
            supports_multimodal_input=True,
            strip_stream_options=True,
            strip_chat_template_kwargs=True,
            flatten_chat_image_url=True,
            strict_model_integrity=False,
        )
    return BackendCapabilities(
        supports_chat=True,
        supports_responses=True,
        supports_tools=True,
        supports_multimodal_input=True,
    )


def _collect_strings(value: object) -> list[str]:
    out: list[str] = []
    seen: set[int] = set()

    def walk(item: object) -> None:
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append(text)
            return
        if isinstance(item, dict):
            marker = id(item)
            if marker in seen:
                return
            seen.add(marker)
            for sub in item.values():
                walk(sub)
            return
        if isinstance(item, list):
            marker = id(item)
            if marker in seen:
                return
            seen.add(marker)
            for sub in item:
                walk(sub)

    walk(value)
    return out


def _payload_owned_by(models_payload: object) -> str:
    if not isinstance(models_payload, dict):
        return ""
    data = models_payload.get("data")
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            owner = str(item.get("owned_by", "")).strip().lower()
            if owner:
                return owner
    owner = str(models_payload.get("owned_by", "")).strip().lower()
    return owner


def detect_backend_profile(
    *,
    requested: str,
    base_url: str,
    model_endpoint: str,
    responses_endpoint: str,
    models_endpoint: str,
    models_payload: Optional[object] = None,
) -> tuple[str, str]:
    requested_normalized = normalize_backend_profile(requested)
    if requested_normalized != AUTO_BACKEND_PROFILE:
        return requested_normalized, "manual override"

    endpoints_blob = " ".join(
        part.lower()
        for part in (base_url, model_endpoint, responses_endpoint, models_endpoint)
        if str(part or "").strip()
    )
    if "ollama" in endpoints_blob or ":11434" in endpoints_blob:
        return "ollama", "endpoint fingerprint"
    if "lmstudio" in endpoints_blob or ":1234" in endpoints_blob:
        return "lmstudio", "endpoint fingerprint"
    if "vllm" in endpoints_blob:
        return "vllm", "endpoint fingerprint"
    if "llama.cpp" in endpoints_blob or "llamacpp" in endpoints_blob:
        return "llamacpp", "endpoint fingerprint"
    if "mlx" in endpoints_blob:
        return "mlx_vlm", "endpoint fingerprint"

    owner = _payload_owned_by(models_payload)
    if owner:
        if "ollama" in owner:
            return "ollama", "models payload owner"
        if "vllm" in owner:
            return "vllm", "models payload owner"
        if "lmstudio" in owner:
            return "lmstudio", "models payload owner"

    joined_models = " ".join(value.lower() for value in _collect_strings(models_payload))
    if joined_models:
        if "mlx" in joined_models and ("vl" in joined_models or "vision" in joined_models):
            return "mlx_vlm", "models payload signature"
        if "ollama" in joined_models:
            return "ollama", "models payload signature"
        if "vllm" in joined_models:
            return "vllm", "models payload signature"
        if "llama.cpp" in joined_models or "llamacpp" in joined_models:
            return "llamacpp", "models payload signature"

    return UNKNOWN_BACKEND_PROFILE, "insufficient signals"


def is_local_backend_profile(profile: str) -> bool:
    return str(profile or "").strip().lower() in LOCAL_BACKEND_PROFILES


def rewrite_payload_for_profile(
    payload: dict[str, object],
    *,
    mode: str,
    profile: str,
) -> tuple[dict[str, object], list[str]]:
    normalized = str(profile or "").strip().lower()
    capabilities = profile_capabilities(normalized)
    out = dict(payload)
    changes: list[str] = []

    if capabilities.strip_stream_options and "stream_options" in out:
        out.pop("stream_options", None)
        changes.append("drop_stream_options")
    if capabilities.strip_chat_template_kwargs and "chat_template_kwargs" in out:
        out.pop("chat_template_kwargs", None)
        changes.append("drop_chat_template_kwargs")

    def to_image_url(value: object) -> str:
        if isinstance(value, dict):
            url = str(value.get("url", "")).strip()
            return url
        return str(value or "").strip()

    def rewrite_content_parts(parts: list[object], *, responses_blocks: bool, flatten_chat_image: bool) -> list[object]:
        updated: list[object] = []
        mutated = False
        for part in parts:
            if not isinstance(part, dict):
                updated.append(part)
                continue
            item = dict(part)
            kind = str(item.get("type", "")).strip().lower()
            if responses_blocks and kind in {"text", "input_text"}:
                text = str(item.get("text", "")).strip()
                updated.append({"type": "input_text", "text": text})
                mutated = True
                continue
            if kind in {"image_url", "input_image"}:
                image_url = to_image_url(item.get("image_url"))
                if responses_blocks:
                    updated.append({"type": "input_image", "image_url": image_url})
                    mutated = True
                    continue
                if flatten_chat_image and image_url:
                    item["image_url"] = image_url
                    mutated = True
            updated.append(item)
        if mutated:
            changes.append("rewrite_multimodal_blocks")
        return updated

    key = "input" if mode == "responses" else "messages"
    items = out.get(key)
    if isinstance(items, list):
        updated_messages: list[object] = []
        list_mutated = False
        for message in items:
            if not isinstance(message, dict):
                updated_messages.append(message)
                continue
            msg = dict(message)
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = rewrite_content_parts(
                    content,
                    responses_blocks=bool(capabilities.responses_input_blocks and mode == "responses"),
                    flatten_chat_image=bool(capabilities.flatten_chat_image_url and mode == "chat"),
                )
                if msg["content"] != content:
                    list_mutated = True
            updated_messages.append(msg)
        if list_mutated:
            out[key] = updated_messages

    return out, changes


def looks_like_backend_model_fallback_error(message: str) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False
    markers = (
        "downloading",
        "download model",
        "pulling",
        "loading model",
        "model not loaded",
        "fetching model",
        "unloading",
    )
    return any(marker in text for marker in markers)
