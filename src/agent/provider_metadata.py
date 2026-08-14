from __future__ import annotations

import urllib.parse
from collections.abc import Callable, Iterable

from core.endpoint_modes import LOCAL_PROPS_PATH, LOCAL_SLOTS_PATH, OPENAI_MODELS_PATH


def _walk(payload: object, keys: Iterable[str], convert: Callable[[object], object | None]) -> object | None:
    visited: set[int] = set()

    def walk(item: object) -> object | None:
        if not isinstance(item, (dict, list)) or id(item) in visited:
            return None
        visited.add(id(item))
        if isinstance(item, dict):
            for key in keys:
                if (value := convert(item.get(key))) is not None:
                    return value
            children = item.values()
        else:
            children = item
        return next((value for child in children if (value := walk(child)) is not None), None)

    return walk(payload)


def extract_model_name(payload: object) -> str | None:
    keys = (
        "id",
        "name",
        "model",
        "model_id",
        "model_name",
        "loaded_model",
        "default_model",
    )

    def candidate(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        text = value.strip()
        return text or None

    result = _walk(payload, keys, candidate)
    return result if isinstance(result, str) else None


def extract_model_context_window(payload: object) -> int | None:
    context_keys = (
        "context_length",
        "context_window",
        "max_context_length",
        "max_model_len",
        "num_ctx",
        "n_ctx",
        "n_ctx_slot",
        "n_ctx_train",
    )

    def candidate_int(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, float):
            parsed = int(value)
            return parsed if parsed > 0 else None
        if isinstance(value, str):
            try:
                parsed = int(value.strip())
            except ValueError:
                return None
            return parsed if parsed > 0 else None
        return None

    search_root = payload.get("data", payload) if isinstance(payload, dict) else payload
    result = _walk(search_root, context_keys, candidate_int)
    if result is None and search_root is not payload:
        result = _walk(payload, context_keys, candidate_int)
    return result if isinstance(result, int) else None


def _metadata_endpoint(models_endpoint: str, target_path: str) -> str:
    parsed = urllib.parse.urlparse(models_endpoint)
    path = parsed.path or ""
    suffix = OPENAI_MODELS_PATH if path.endswith(OPENAI_MODELS_PATH) else "/models" if path.endswith("/models") else ""
    path = f"{path[: -len(suffix)]}{target_path}" if suffix else target_path
    return urllib.parse.urlunparse(parsed._replace(path=path, params="", query="", fragment=""))


def props_endpoint_from_models_endpoint(models_endpoint: str) -> str:
    return _metadata_endpoint(models_endpoint, LOCAL_PROPS_PATH)


def slots_endpoint_from_models_endpoint(models_endpoint: str) -> str:
    return _metadata_endpoint(models_endpoint, LOCAL_SLOTS_PATH)
