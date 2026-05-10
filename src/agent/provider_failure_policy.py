from __future__ import annotations

import urllib.parse

from core.streaming import should_retry


def is_endpoint_unsupported(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {400, 404, 405, 415, 422}:
        return True
    text = str(exc).lower()
    markers = (
        "unsupported",
        "unknown endpoint",
        "not found",
        "unrecognized request argument",
        "unknown field",
        "validation error",
        "responses",
        "chat/completions",
    )
    return any(marker in text for marker in markers)


def is_local_endpoint(endpoint: str) -> bool:
    parsed = urllib.parse.urlparse(endpoint)
    host = (parsed.hostname or "").strip().lower()
    return host in {"127.0.0.1", "localhost"}


def is_connection_refused_error(exc: Exception) -> bool:
    return "refused" in str(exc).lower()


def should_retry_provider_exception(exc: Exception, *, model_endpoint: str) -> bool:
    if is_local_endpoint(model_endpoint) and is_connection_refused_error(exc):
        return False
    return should_retry(exc)


def is_transport_failure(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) is None
