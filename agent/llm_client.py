from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from core.streaming import build_ssl_context, should_retry, stream_chat_completions

from agent.telemetry import TelemetryEmitter
from agent.types import StreamPassResult, ToolCallAccumulator


class LLMClient:
    def __init__(self, config: Dict[str, Any], debug: bool = False, telemetry: Optional[TelemetryEmitter] = None) -> None:
        self.debug = debug
        self.telemetry = telemetry or TelemetryEmitter()
        self.auth_header = (
            os.environ.get("ALPHANUS_AUTH_HEADER", "").strip()
            or os.environ.get("AUTH_HEADER", "").strip()
            or None
        )
        self.connect_timeout_s = 10
        self.per_turn_retries = 1
        self.retry_backoff_s = 0.5
        self._ready_checked = False
        self.reload_config(config)

    def reload_config(self, config: Dict[str, Any]) -> None:
        self.config = config
        agent_cfg = config.get("agent", {}) if isinstance(config.get("agent"), dict) else {}
        self.model_endpoint = agent_cfg.get("model_endpoint", "http://127.0.0.1:8080/v1/chat/completions")
        self.models_endpoint = agent_cfg.get("models_endpoint", "http://127.0.0.1:8080/v1/models")
        self.tls_verify = bool(agent_cfg.get("tls_verify", True))
        self.ca_bundle_path = agent_cfg.get("ca_bundle_path")
        self.allow_cross_host = bool(agent_cfg.get("allow_cross_host_endpoints", False))
        self.request_timeout_s = float(agent_cfg.get("request_timeout_s", 180))
        self.readiness_timeout_s = float(agent_cfg.get("readiness_timeout_s", 30))
        self.readiness_poll_s = float(agent_cfg.get("readiness_poll_s", 0.5))
        raw_max_tokens = agent_cfg.get("max_tokens")
        if raw_max_tokens in (None, "", 0):
            self.default_max_tokens = None
        else:
            value = int(raw_max_tokens)
            self.default_max_tokens = value if value > 0 else None
        self.classifier_model = str(agent_cfg.get("classifier_model", "")).strip()
        self.classifier_use_primary_model = bool(agent_cfg.get("classifier_use_primary_model", True))
        self.enable_structured_classification = bool(agent_cfg.get("enable_structured_classification", True))
        self.max_classifier_tokens = max(32, int(agent_cfg.get("max_classifier_tokens", 256)))
        self.ssl_context = build_ssl_context(self.tls_verify, self.ca_bundle_path)
        self._ready_checked = False

    def headers(self) -> Dict[str, str]:
        headers = {}
        if self.auth_header and ":" in self.auth_header:
            key, value = self.auth_header.split(":", 1)
            headers[key.strip()] = value.strip()
        return headers

    @staticmethod
    def stop_requested(stop_event) -> bool:
        return bool(stop_event is not None and stop_event.is_set())

    @staticmethod
    def sleep_with_stop(duration_s: float, stop_event) -> bool:
        if duration_s <= 0:
            return not LLMClient.stop_requested(stop_event)
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            if LLMClient.stop_requested(stop_event):
                return False
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        return not LLMClient.stop_requested(stop_event)

    @staticmethod
    def extract_model_name(payload: Any) -> Optional[str]:
        def candidate(value: Any) -> Optional[str]:
            text = str(value or "").strip()
            return text or None

        def from_item(item: Any) -> Optional[str]:
            if isinstance(item, dict):
                for key in ("id", "name", "model"):
                    picked = candidate(item.get(key))
                    if picked:
                        return picked
            return candidate(item)

        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list):
                for item in data:
                    picked = from_item(item)
                    if picked:
                        return picked
            elif data is not None:
                picked = from_item(data)
                if picked:
                    return picked
            return from_item(payload)

        if isinstance(payload, list):
            for item in payload:
                picked = from_item(item)
                if picked:
                    return picked
        return None

    @staticmethod
    def extract_model_context_window(payload: Any) -> Optional[int]:
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

        def candidate_int(value: Any) -> Optional[int]:
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
                except Exception:
                    return None
                return parsed if parsed > 0 else None
            return None

        def from_item(item: Any) -> Optional[int]:
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

    def fetch_json(self, url: str, timeout_s: Optional[float] = None) -> Any:
        request = urllib.request.Request(url, headers=self.headers(), method="GET")
        timeout = self.connect_timeout_s if timeout_s is None else max(0.1, float(timeout_s))
        with urllib.request.urlopen(request, timeout=timeout, context=self.ssl_context) as response:
            return json.loads(response.read().decode("utf-8"))

    def fetch_model_metadata(self, timeout_s: Optional[float] = None) -> tuple[Optional[str], Optional[int]]:
        try:
            payload = self.fetch_json(self.models_endpoint, timeout_s=timeout_s)
        except Exception as exc:
            self.telemetry.emit("model_fetch_failed", endpoint=self.models_endpoint, error=str(exc))
            return None, None
        model_name = self.extract_model_name(payload)
        context_window = self.extract_model_context_window(payload)
        if context_window is not None:
            return model_name, context_window
        props_endpoint = self.props_endpoint_from_models_endpoint(self.models_endpoint)
        try:
            props_payload = self.fetch_json(props_endpoint, timeout_s=timeout_s)
        except Exception as exc:
            self.telemetry.emit("model_props_fetch_failed", endpoint=props_endpoint, error=str(exc))
            return model_name, None
        return model_name, self.extract_model_context_window(props_payload)

    def ensure_ready(
        self,
        stop_event=None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        timeout_s: Optional[float] = None,
    ) -> Optional[bool]:
        timeout = self.readiness_timeout_s if timeout_s is None else max(0.0, float(timeout_s))
        deadline = time.monotonic() + timeout
        req = urllib.request.Request(self.models_endpoint, headers=self.headers(), method="GET")
        attempt_timeout = min(self.connect_timeout_s, 1.0)
        self.telemetry.emit("readiness_start", endpoint=self.models_endpoint, timeout_s=timeout)
        self._emit(on_event, {"type": "info", "text": f"waiting for endpoint handshake: {self.models_endpoint}"})

        while time.monotonic() < deadline:
            if self.stop_requested(stop_event):
                self._ready_checked = False
                self.telemetry.emit("readiness_cancelled", endpoint=self.models_endpoint)
                return None
            try:
                with urllib.request.urlopen(req, timeout=attempt_timeout, context=self.ssl_context) as resp:
                    if 200 <= resp.status < 300:
                        self._ready_checked = True
                        self.telemetry.emit("readiness_ok", endpoint=self.models_endpoint, status=resp.status)
                        return True
            except Exception as exc:
                self.telemetry.emit("readiness_retry", endpoint=self.models_endpoint, error=str(exc))
                if not self.sleep_with_stop(self.readiness_poll_s, stop_event):
                    self._ready_checked = False
                    self.telemetry.emit("readiness_cancelled", endpoint=self.models_endpoint)
                    return None
        self._ready_checked = False
        self.telemetry.emit("readiness_failed", endpoint=self.models_endpoint)
        return False

    def build_payload(
        self,
        model_messages: List[Dict[str, Any]],
        thinking: bool,
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        max_tokens_override: Optional[int] = None,
        model_override: str = "",
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "messages": model_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        }
        limit = self.default_max_tokens if max_tokens_override is None else max_tokens_override
        if limit is not None and int(limit) > 0:
            payload["max_tokens"] = int(limit)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if model_override.strip():
            payload["model"] = model_override.strip()
        return payload

    def _stream_one_pass(
        self,
        payload: Dict[str, Any],
        stop_event,
        on_event: Optional[Callable[[Dict[str, Any]], None]],
        pass_id: str,
    ) -> StreamPassResult:
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        finish_reason = "stop"
        tool_acc = ToolCallAccumulator(pass_id=pass_id)
        tool_phase_started = False
        suppress_content_stream = False
        usage: Dict[str, int] = {}
        self.telemetry.emit("chat_pass_start", pass_id=pass_id, endpoint=self.model_endpoint, payload=payload)

        for chunk in stream_chat_completions(
            endpoint=self.model_endpoint,
            payload=payload,
            timeout_s=self.request_timeout_s,
            headers=self.headers(),
            ssl_context=self.ssl_context,
            stop_event=stop_event,
            on_debug_event=lambda event: self.telemetry.emit("http_stream", pass_id=pass_id, **event),
        ):
            if self.stop_requested(stop_event):
                self.telemetry.emit("chat_pass_cancelled", pass_id=pass_id)
                return StreamPassResult(finish_reason="cancelled")
            choices = chunk.get("choices", [])
            chunk_usage = chunk.get("usage") or {}
            if isinstance(chunk_usage, dict) and chunk_usage:
                usage = {
                    str(key): int(value)
                    for key, value in chunk_usage.items()
                    if isinstance(value, (int, float))
                }
                if usage:
                    self._emit(on_event, {"type": "usage", "usage": dict(usage)})
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {}) or {}

            reasoning = delta.get("reasoning_content") or ""
            if reasoning:
                reasoning_parts.append(reasoning)
                self._emit(on_event, {"type": "reasoning_token", "text": reasoning})

            content = delta.get("content") or ""
            if content:
                content_parts.append(content)
                if self._contains_tool_markup(content):
                    suppress_content_stream = True
                if not suppress_content_stream:
                    self._emit(on_event, {"type": "content_token", "text": content})

            tool_deltas = delta.get("tool_calls") or []
            if tool_deltas:
                if not tool_phase_started:
                    tool_phase_started = True
                    self._emit(on_event, {"type": "tool_phase_started"})
                for update in tool_acc.ingest(tool_deltas):
                    self._emit(
                        on_event,
                        {
                            "type": "tool_call_delta",
                            "stream_id": update["stream_id"],
                            "id": update.get("id") or "",
                            "name": update.get("name") or "",
                            "raw_arguments": update.get("raw_arguments") or "",
                        },
                    )

            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])

        tool_calls = tool_acc.finalize()
        if tool_calls and finish_reason not in {"tool_calls", "cancelled"}:
            finish_reason = "tool_calls"

        self.telemetry.emit(
            "chat_pass_end",
            pass_id=pass_id,
            finish_reason=finish_reason,
            content_chars=len("".join(content_parts)),
            reasoning_chars=len("".join(reasoning_parts)),
            tool_call_count=len(tool_calls),
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            tool_calls=[
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in tool_calls
            ],
        )
        return StreamPassResult(
            finish_reason=finish_reason,
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            tool_calls=tool_calls,
            usage=usage,
        )

    def call_with_retry(self, payload: Dict[str, Any], stop_event, on_event, pass_id: str) -> StreamPassResult:
        attempt = 0
        while True:
            try:
                return self._stream_one_pass(payload, stop_event, on_event, pass_id=pass_id)
            except Exception as exc:
                if should_retry(exc) and attempt < self.per_turn_retries:
                    attempt += 1
                    self._emit(
                        on_event,
                        {"type": "info", "text": f"Retrying request ({attempt}/{self.per_turn_retries})..."},
                    )
                    self.telemetry.emit("request_retry", pass_id=pass_id, attempt=attempt, error=str(exc))
                    if not self.sleep_with_stop(self.retry_backoff_s, stop_event):
                        return StreamPassResult(finish_reason="cancelled")
                    ready = self.ensure_ready(
                        stop_event=stop_event,
                        on_event=on_event,
                        timeout_s=min(self.readiness_timeout_s, 5.0),
                    )
                    if ready is None:
                        return StreamPassResult(finish_reason="cancelled")
                    if not ready:
                        raise
                    continue
                raise

    @staticmethod
    def _emit(on_event: Optional[Callable[[Dict[str, Any]], None]], event: Dict[str, Any]) -> None:
        if not on_event:
            return
        try:
            on_event(event)
        except Exception:
            return

    @staticmethod
    def _contains_tool_markup(text: str) -> bool:
        if not text:
            return False
        markers = ("<tool_call", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>")
        return any(marker in text.lower() for marker in markers)
