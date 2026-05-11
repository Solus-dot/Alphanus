from __future__ import annotations

import urllib.parse


class ProviderMetadataExtractor:
    @staticmethod
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
        visited: set[int] = set()

        def candidate(value: object) -> str | None:
            if not isinstance(value, str):
                return None
            text = value.strip()
            return text or None

        def walk(item: object) -> str | None:
            if isinstance(item, dict):
                marker = id(item)
                if marker in visited:
                    return None
                visited.add(marker)
                for key in keys:
                    picked = candidate(item.get(key))
                    if picked:
                        return picked
                for value in item.values():
                    if isinstance(value, (dict, list)):
                        picked = walk(value)
                        if picked:
                            return picked
                return None
            if isinstance(item, list):
                marker = id(item)
                if marker in visited:
                    return None
                visited.add(marker)
                for value in item:
                    picked = walk(value)
                    if picked:
                        return picked
                return None
            return None

        if isinstance(payload, (dict, list)):
            return walk(payload)
        return None

    @staticmethod
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
        visited: set[int] = set()

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

        def from_item(item: object) -> int | None:
            if isinstance(item, dict):
                marker = id(item)
                if marker in visited:
                    return None
                visited.add(marker)
                for key in context_keys:
                    picked = candidate_int(item.get(key))
                    if picked is not None:
                        return picked
                for value in item.values():
                    picked = from_item(value)
                    if picked is not None:
                        return picked
                return None
            if isinstance(item, list):
                marker = id(item)
                if marker in visited:
                    return None
                visited.add(marker)
                for value in item:
                    picked = from_item(value)
                    if picked is not None:
                        return picked
                return None
            return None

        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list):
                for item in data:
                    picked = from_item(item)
                    if picked is not None:
                        return picked
            elif data is not None:
                picked = from_item(data)
                if picked is not None:
                    return picked
            return from_item(payload)

        if isinstance(payload, list):
            for item in payload:
                picked = from_item(item)
                if picked is not None:
                    return picked
        return None

    @staticmethod
    def props_endpoint_from_models_endpoint(models_endpoint: str) -> str:
        parsed = urllib.parse.urlparse(models_endpoint)
        path = parsed.path or ""
        if path.endswith("/v1/models"):
            path = path[: -len("/v1/models")] + "/props"
        elif path.endswith("/models"):
            path = path[: -len("/models")] + "/props"
        else:
            path = "/props"
        return urllib.parse.urlunparse(parsed._replace(path=path, params="", query="", fragment=""))

    @staticmethod
    def slots_endpoint_from_models_endpoint(models_endpoint: str) -> str:
        parsed = urllib.parse.urlparse(models_endpoint)
        path = parsed.path or ""
        if path.endswith("/v1/models"):
            path = path[: -len("/v1/models")] + "/slots"
        elif path.endswith("/models"):
            path = path[: -len("/models")] + "/slots"
        else:
            path = "/slots"
        return urllib.parse.urlunparse(parsed._replace(path=path, params="", query="", fragment=""))
