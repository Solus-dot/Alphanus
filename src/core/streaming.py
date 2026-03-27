from __future__ import annotations

import logging

import json
import select
import ssl
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, Generator, Optional


RETRYABLE_STATUS = {429, 500, 502, 503, 504}
RETRYABLE_URL_ERRORS = (TimeoutError, ConnectionResetError)
STREAM_POLL_TIMEOUT_S = 0.25


class StreamError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


def build_ssl_context(tls_verify: bool = True, ca_bundle_path: Optional[str] = None) -> Optional[ssl.SSLContext]:
    if tls_verify and not ca_bundle_path:
        return None
    context = ssl.create_default_context(cafile=ca_bundle_path)
    if not tls_verify:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    return context


def should_retry(exc: Exception) -> bool:
    if isinstance(exc, StreamError):
        return exc.retryable
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in RETRYABLE_STATUS
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, RETRYABLE_URL_ERRORS):
            return True
        text = str(reason).lower()
        return any(tok in text for tok in ("timed out", "reset", "temporarily", "refused"))
    return False


def _extract_stream_socket(resp):
    candidates = [
        getattr(getattr(getattr(resp, "fp", None), "raw", None), "_sock", None),
        getattr(getattr(resp, "fp", None), "raw", None),
        getattr(resp, "fp", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        fileno = getattr(candidate, "fileno", None)
        if callable(fileno):
            try:
                fileno()
                return candidate
            except Exception as exc:
                logging.debug("Socket fileno() check failed for %s: %s", type(candidate).__name__, exc)
                continue
    return None


def stream_chat_completions(
    endpoint: str,
    payload: Dict,
    timeout_s: float,
    headers: Optional[Dict[str, str]] = None,
    ssl_context: Optional[ssl.SSLContext] = None,
    stop_event=None,
    on_debug_event: Optional[Callable[[Dict], None]] = None,
) -> Generator[Dict, None, None]:
    body = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(endpoint, data=body, headers=req_headers, method="POST")
    idle_timeout_s = max(0.1, float(timeout_s))
    idle_deadline = time.monotonic() + idle_timeout_s

    try:
        if on_debug_event:
            on_debug_event(
                {
                    "type": "http_request",
                    "endpoint": endpoint,
                    "method": "POST",
                    "timeout_s": timeout_s,
                    "payload_bytes": len(body),
                    "header_keys": sorted(req_headers.keys()),
                }
            )
        with urllib.request.urlopen(req, timeout=timeout_s, context=ssl_context) as resp:
            if on_debug_event:
                on_debug_event(
                    {
                        "type": "http_response",
                        "endpoint": endpoint,
                        "status": getattr(resp, "status", None),
                        "reason": getattr(resp, "reason", ""),
                    }
                )
            readline = getattr(resp, "readline", None)
            stream_sock = _extract_stream_socket(resp)
            if callable(readline):
                while True:
                    if stop_event is not None and stop_event.is_set():
                        return
                    if time.monotonic() >= idle_deadline:
                        raise StreamError(
                            f"Network error: stream was idle for {timeout_s}s",
                            retryable=True,
                        )
                    if stream_sock is not None:
                        ready, _, _ = select.select([stream_sock], [], [], STREAM_POLL_TIMEOUT_S)
                        if not ready:
                            continue
                    try:
                        raw = readline()
                    except TimeoutError:
                        continue
                    if not raw:
                        return
                    idle_deadline = time.monotonic() + idle_timeout_s
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload_str = line[5:].strip()
                    if payload_str == "[DONE]":
                        if on_debug_event:
                            on_debug_event({"type": "http_done", "endpoint": endpoint})
                        return
                    try:
                        chunk = json.loads(payload_str)
                        yield chunk
                    except json.JSONDecodeError:
                        if on_debug_event:
                            on_debug_event(
                                {
                                    "type": "http_chunk_decode_error",
                                    "endpoint": endpoint,
                                    "payload_preview": payload_str[:500],
                                }
                            )
                        continue
                return
            for raw in resp:
                if stop_event is not None and stop_event.is_set():
                    return
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload_str = line[5:].strip()
                if payload_str == "[DONE]":
                    if on_debug_event:
                        on_debug_event({"type": "http_done", "endpoint": endpoint})
                    return
                try:
                    chunk = json.loads(payload_str)
                    yield chunk
                except json.JSONDecodeError:
                    if on_debug_event:
                        on_debug_event(
                            {
                                "type": "http_chunk_decode_error",
                                "endpoint": endpoint,
                                "payload_preview": payload_str[:500],
                            }
                        )
                    continue
    except urllib.error.HTTPError as exc:
        message = exc.read().decode(errors="replace") if hasattr(exc, "read") else str(exc)
        if on_debug_event:
            on_debug_event(
                {
                    "type": "http_error",
                    "endpoint": endpoint,
                    "status": exc.code,
                    "message": message[:1000],
                }
            )
        raise StreamError(
            f"HTTP {exc.code}: {message[:400]}",
            status_code=exc.code,
            retryable=exc.code in RETRYABLE_STATUS,
        ) from exc
    except urllib.error.URLError as exc:
        reason = str(exc.reason)
        retryable = should_retry(exc)
        if on_debug_event:
            on_debug_event(
                {
                    "type": "network_error",
                    "endpoint": endpoint,
                    "reason": reason,
                    "retryable": retryable,
                }
            )
        raise StreamError(f"Network error: {reason}", retryable=retryable) from exc
