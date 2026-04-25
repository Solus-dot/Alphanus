from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from core.runtime_config import ProviderConfig
from core.streaming import build_ssl_context, should_retry, stream_chat_completions as core_stream_chat_completions

from agent.telemetry import TelemetryEmitter
from core.types import JsonObject, ModelStatus, StreamPassResult, ToolCallAccumulator


@dataclass(slots=True)
class ProviderCapabilities:
    streaming: bool = True
    tool_calls: bool = True
    reasoning_tokens: bool = True
    context_window_probe: bool = True
    responses_endpoint: bool = True
    multimodal: bool = True
    structured_output: bool = True


@dataclass(slots=True)
class ProviderCompatibilityProfile:
    selected_endpoint_mode: str = "chat"
    supports_responses: bool = False
    supports_chat: bool = True
    supports_tools: bool = True
    supports_stream: bool = True
    supports_reasoning: bool = True
    supports_multimodal_input: bool = False
    supports_multimodal_output: bool = False
    supports_structured_output: bool = False
    tier: str = "tier2-best-effort"

    def to_json(self) -> dict[str, object]:
        return {
            "selected_endpoint_mode": self.selected_endpoint_mode,
            "supports_responses": self.supports_responses,
            "supports_chat": self.supports_chat,
            "supports_tools": self.supports_tools,
            "supports_stream": self.supports_stream,
            "supports_reasoning": self.supports_reasoning,
            "supports_multimodal_input": self.supports_multimodal_input,
            "supports_multimodal_output": self.supports_multimodal_output,
            "supports_structured_output": self.supports_structured_output,
            "tier": self.tier,
        }

class OpenAICompatibleProvider:
    ONLINE_STATUS_TTL_S = 5.0
    OFFLINE_STATUS_TTL_S = 2.0

    def __init__(
        self,
        config: ProviderConfig,
        *,
        telemetry: Optional[TelemetryEmitter] = None,
        debug: bool = False,
        stream_chat_completions_fn: Optional[Callable[..., object]] = None,
    ) -> None:
        self.debug = debug
        self.telemetry = telemetry or TelemetryEmitter()
        self._stream_chat_completions = stream_chat_completions_fn or core_stream_chat_completions
        self._ready_checked = False
        self._model_status = ModelStatus(endpoint=config.models_endpoint)
        self.reload_config(config)

    def reload_config(self, config: ProviderConfig) -> None:
        self.config = config
        self.auth_header = config.auth_header
        self.base_url = config.base_url
        self.model_endpoint = config.model_endpoint
        self.responses_endpoint = config.responses_endpoint
        self.models_endpoint = config.models_endpoint
        endpoint_mode = str(config.endpoint_mode or "auto").strip().lower()
        self.endpoint_mode = endpoint_mode if endpoint_mode in {"auto", "responses", "chat"} else "auto"
        self.tls_verify = config.tls_verify
        self.ca_bundle_path = config.ca_bundle_path
        self.allow_cross_host = config.allow_cross_host
        self.request_timeout_s = config.request_timeout_s
        self.readiness_timeout_s = config.readiness_timeout_s
        self.readiness_poll_s = config.readiness_poll_s
        self.connect_timeout_s = config.connect_timeout_s
        self.per_turn_retries = config.per_turn_retries
        self.retry_backoff_s = config.retry_backoff_s
        self.default_max_tokens = config.default_max_tokens
        self.api_key_env = config.api_key_env
        self.auth_header_template = config.auth_header_template
        self.ssl_context = build_ssl_context(self.tls_verify, self.ca_bundle_path)
        self._ready_checked = False
        self._model_status = ModelStatus(endpoint=self.models_endpoint)
        self._profile_cache_key: tuple[str, str, str, str] = ("", "", "", "")
        self._resolved_endpoint_mode = "chat"
        self._fallback_events: list[dict[str, object]] = []
        self._compatibility = ProviderCompatibilityProfile(selected_endpoint_mode="chat")

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    def headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.auth_header and ":" in self.auth_header:
            key, value = self.auth_header.split(":", 1)
            headers[key.strip()] = value.strip()
        return headers

    def compatibility_profile(self) -> dict[str, object]:
        return self._compatibility.to_json()

    def fallback_events(self) -> list[dict[str, object]]:
        return [dict(item) for item in self._fallback_events[-10:]]

    def _compatibility_cache_key(self) -> tuple[str, str, str, str]:
        return (
            str(self.base_url),
            str(self.model_endpoint),
            str(self.responses_endpoint),
            str(self.auth_header or ""),
        )

    def _is_endpoint_unsupported(self, exc: Exception) -> bool:
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

    def _select_endpoint_mode(self) -> str:
        if self.endpoint_mode in {"responses", "chat"}:
            selected = self.endpoint_mode
            self._resolved_endpoint_mode = selected
            self._compatibility = ProviderCompatibilityProfile(
                selected_endpoint_mode=selected,
                supports_responses=selected == "responses",
                supports_chat=selected == "chat",
                supports_tools=True,
                supports_stream=True,
                supports_reasoning=True,
                supports_multimodal_input=selected == "responses",
                supports_multimodal_output=selected == "responses",
                supports_structured_output=selected == "responses",
                tier="tier1-guaranteed",
            )
            self._profile_cache_key = self._compatibility_cache_key()
            return selected

        key = self._compatibility_cache_key()
        if key == self._profile_cache_key and self._resolved_endpoint_mode in {"responses", "chat"}:
            return self._resolved_endpoint_mode
        self._profile_cache_key = key
        self._resolved_endpoint_mode = "responses"
        self._compatibility = ProviderCompatibilityProfile(
            selected_endpoint_mode="responses",
            supports_responses=True,
            supports_chat=True,
            supports_tools=True,
            supports_stream=True,
            supports_reasoning=True,
            supports_multimodal_input=True,
            supports_multimodal_output=True,
            supports_structured_output=True,
            tier="tier1-guaranteed",
        )
        return self._resolved_endpoint_mode

    @staticmethod
    def _chat_tools_to_responses(tools: list[JsonObject]) -> list[JsonObject]:
        converted: list[JsonObject] = []
        for item in tools:
            if not isinstance(item, dict):
                continue
            function = item.get("function")
            if isinstance(function, dict):
                converted.append(
                    {
                        "type": "function",
                        "name": str(function.get("name", "")).strip(),
                        "description": str(function.get("description", "")).strip(),
                        "parameters": function.get("parameters") if isinstance(function.get("parameters"), dict) else {},
                    }
                )
                continue
            if item.get("type") == "function" and isinstance(item.get("name"), str):
                converted.append(item)
        return converted

    @staticmethod
    def _responses_tools_to_chat(tools: list[JsonObject]) -> list[JsonObject]:
        converted: list[JsonObject] = []
        for item in tools:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("function"), dict):
                converted.append(item)
                continue
            if str(item.get("type", "")).strip() != "function":
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(item.get("description", "")).strip(),
                        "parameters": item.get("parameters") if isinstance(item.get("parameters"), dict) else {},
                    },
                }
            )
        return converted

    def _payload_to_mode(self, payload: dict[str, object], mode: str) -> JsonObject:
        if mode not in {"responses", "chat"}:
            mode = "chat"
        messages = payload.get("messages")
        if not isinstance(messages, list):
            messages = payload.get("input")
        message_list = [item for item in messages] if isinstance(messages, list) else []
        tools_raw = payload.get("tools")
        tools = [item for item in tools_raw] if isinstance(tools_raw, list) else None
        thinking = False
        template = payload.get("chat_template_kwargs")
        if isinstance(template, dict):
            thinking = bool(template.get("enable_thinking"))
        max_tokens_override = None
        if isinstance(payload.get("max_output_tokens"), (int, float)):
            max_tokens_override = int(payload["max_output_tokens"])
        elif isinstance(payload.get("max_tokens"), (int, float)):
            max_tokens_override = int(payload["max_tokens"])
        model_override = str(payload.get("model", "")).strip()
        converted_tools: Optional[list[JsonObject]] = None
        if tools is not None:
            converted_tools = self._chat_tools_to_responses(tools) if mode == "responses" else self._responses_tools_to_chat(tools)
        return self.build_payload(
            model_messages=message_list,  # type: ignore[arg-type]
            thinking=thinking,
            tools=converted_tools,
            max_tokens_override=max_tokens_override,
            model_override=model_override,
            mode=mode,
        )

    def build_payload(
        self,
        *,
        model_messages: list[JsonObject],
        thinking: bool,
        tools: Optional[list[JsonObject]] = None,
        max_tokens_override: Optional[int] = None,
        model_override: str = "",
        mode: Optional[str] = None,
    ) -> JsonObject:
        selected_mode = mode if mode in {"responses", "chat"} else self._select_endpoint_mode()
        limit = self.default_max_tokens if max_tokens_override is None else max_tokens_override
        if selected_mode == "responses":
            payload: JsonObject = {
                "input": model_messages,
                "stream": True,
            }
            if limit is not None and int(limit) > 0:
                payload["max_output_tokens"] = int(limit)
            if tools:
                payload["tools"] = self._chat_tools_to_responses(tools)
                payload["tool_choice"] = "auto"
            if model_override.strip():
                payload["model"] = model_override.strip()
            if thinking:
                payload["reasoning"] = {"summary": "auto"}
            return payload

        payload = {
            "messages": model_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        }
        if limit is not None and int(limit) > 0:
            payload["max_tokens"] = int(limit)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if model_override.strip():
            payload["model"] = model_override.strip()
        return payload

    @staticmethod
    def stop_requested(stop_event) -> bool:
        return bool(stop_event is not None and stop_event.is_set())

    @staticmethod
    def sleep_with_stop(duration_s: float, stop_event) -> bool:
        if duration_s <= 0:
            return not OpenAICompatibleProvider.stop_requested(stop_event)
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            if OpenAICompatibleProvider.stop_requested(stop_event):
                return False
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        return not OpenAICompatibleProvider.stop_requested(stop_event)

    @staticmethod
    def extract_model_name(payload: object) -> Optional[str]:
        def candidate(value: object) -> Optional[str]:
            text = str(value or "").strip()
            return text or None

        def from_item(item: object) -> Optional[str]:
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
    def extract_model_context_window(payload: object) -> Optional[int]:
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

        def candidate_int(value: object) -> Optional[int]:
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

        def from_item(item: object) -> Optional[int]:
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

    def fetch_json(self, url: str, timeout_s: Optional[float] = None) -> object:
        request = urllib.request.Request(url, headers=self.headers(), method="GET")
        timeout = self.connect_timeout_s if timeout_s is None else max(0.1, float(timeout_s))
        with urllib.request.urlopen(request, timeout=timeout, context=self.ssl_context) as response:
            raw = response.read().decode("utf-8")
            if not raw.strip():
                return {}
            return json.loads(raw)

    def list_models(self, timeout_s: Optional[float] = None) -> object:
        return self.fetch_json(self.models_endpoint, timeout_s=timeout_s)

    def get_model_status(self) -> ModelStatus:
        status = self._model_status
        return ModelStatus(
            state=status.state,
            model_name=status.model_name,
            context_window=status.context_window,
            last_checked_at=status.last_checked_at,
            last_success_at=status.last_success_at,
            last_error=status.last_error,
            endpoint=status.endpoint,
        )

    def is_model_status_fresh(self, status: Optional[ModelStatus] = None, *, now: Optional[float] = None) -> bool:
        current_status = self.get_model_status() if status is None else status
        return current_status.is_fresh(
            now=now,
            online_ttl_s=self.ONLINE_STATUS_TTL_S,
            offline_ttl_s=self.OFFLINE_STATUS_TTL_S,
        )

    def _store_model_status(self, status: ModelStatus) -> ModelStatus:
        self._model_status = ModelStatus(
            state=status.state,
            model_name=status.model_name,
            context_window=status.context_window,
            last_checked_at=status.last_checked_at,
            last_success_at=status.last_success_at,
            last_error=status.last_error,
            endpoint=status.endpoint or self.models_endpoint,
        )
        self._ready_checked = self._model_status.state == "online"
        return self.get_model_status()

    def _model_status_error_state(self, exc: Exception) -> str:
        if isinstance(exc, urllib.error.HTTPError):
            return "offline"
        if isinstance(exc, urllib.error.URLError):
            return "offline"
        if isinstance(exc, (ConnectionError, TimeoutError, ConnectionResetError)):
            return "offline"
        return "unknown"

    def refresh_model_status(self, timeout_s: Optional[float] = None, force: bool = False) -> ModelStatus:
        current = self.get_model_status()
        if not force and self.is_model_status_fresh(current):
            return current

        now = time.monotonic()
        deadline = None if timeout_s is None else now + max(0.0, float(timeout_s))

        def remaining_timeout() -> Optional[float]:
            if deadline is None:
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Model status probe timed out")
            return remaining

        context_window = current.context_window
        slots_endpoint = self.slots_endpoint_from_models_endpoint(self.models_endpoint)
        try:
            slots_payload = self.fetch_json(slots_endpoint, timeout_s=remaining_timeout())
        except Exception as exc:
            self.telemetry.emit("model_slots_fetch_failed", endpoint=slots_endpoint, error=str(exc))
        else:
            context_window = self.extract_model_context_window(slots_payload) or context_window

        try:
            payload = self.list_models(timeout_s=remaining_timeout())
        except Exception as exc:
            self.telemetry.emit("model_fetch_failed", endpoint=self.models_endpoint, error=str(exc))
            return self._store_model_status(
                ModelStatus(
                    state=self._model_status_error_state(exc),
                    model_name=current.model_name,
                    context_window=context_window,
                    last_checked_at=now,
                    last_success_at=current.last_success_at,
                    last_error=str(exc),
                    endpoint=self.models_endpoint,
                )
            )

        model_name = self.extract_model_name(payload) or current.model_name
        if context_window is None:
            context_window = self.extract_model_context_window(payload)
        if context_window is None:
            props_endpoint = self.props_endpoint_from_models_endpoint(self.models_endpoint)
            try:
                props_payload = self.fetch_json(props_endpoint, timeout_s=remaining_timeout())
            except Exception as exc:
                self.telemetry.emit("model_props_fetch_failed", endpoint=props_endpoint, error=str(exc))
            else:
                context_window = self.extract_model_context_window(props_payload)
        return self._store_model_status(
            ModelStatus(
                state="online",
                model_name=model_name,
                context_window=context_window,
                last_checked_at=now,
                last_success_at=now,
                last_error="",
                endpoint=self.models_endpoint,
            )
        )

    def mark_model_transport_failure(self, exc: Exception) -> None:
        current = self.get_model_status()
        now = time.monotonic()
        self.telemetry.emit("model_transport_failure", endpoint=self.models_endpoint, error=str(exc))
        self._store_model_status(
            ModelStatus(
                state="offline",
                model_name=current.model_name,
                context_window=current.context_window,
                last_checked_at=now,
                last_success_at=current.last_success_at,
                last_error=str(exc),
                endpoint=self.models_endpoint,
            )
        )

    def check_ready(
        self,
        stop_event=None,
        on_event: Optional[Callable[[JsonObject], None]] = None,
        timeout_s: Optional[float] = None,
    ) -> Optional[bool]:
        timeout = self.readiness_timeout_s if timeout_s is None else max(0.0, float(timeout_s))
        deadline = time.monotonic() + timeout
        attempt_timeout = min(self.connect_timeout_s, 1.0)
        self.telemetry.emit("readiness_start", endpoint=self.models_endpoint, timeout_s=timeout)
        self._emit(on_event, {"type": "info", "text": f"waiting for endpoint handshake: {self.models_endpoint}"})

        while time.monotonic() < deadline:
            if self.stop_requested(stop_event):
                self._ready_checked = False
                self.telemetry.emit("readiness_cancelled", endpoint=self.models_endpoint)
                return None
            status = self.refresh_model_status(timeout_s=attempt_timeout, force=True)
            if status.state == "online":
                self.telemetry.emit("readiness_ok", endpoint=self.models_endpoint, status=200)
                return True
            self.telemetry.emit("readiness_retry", endpoint=self.models_endpoint, error=status.last_error)
            if not self.sleep_with_stop(self.readiness_poll_s, stop_event):
                self._ready_checked = False
                self.telemetry.emit("readiness_cancelled", endpoint=self.models_endpoint)
                return None
        self._ready_checked = False
        self.telemetry.emit("readiness_failed", endpoint=self.models_endpoint)
        return False

    def _status_probe_timeout_s(self) -> float:
        return min(self.connect_timeout_s, 1.0)

    def _status_allows_immediate_send(self) -> ModelStatus:
        status = self.get_model_status()
        if self.is_model_status_fresh(status):
            return status
        return self.refresh_model_status(timeout_s=self._status_probe_timeout_s(), force=True)

    def should_fail_fast_on_offline_status(self, status: ModelStatus) -> bool:
        if status.state != "offline":
            return False
        if not self.is_model_status_fresh(status):
            return False
        if self._is_local_endpoint(self.models_endpoint) and status.last_success_at <= 0:
            return False
        return True

    @staticmethod
    def _is_local_endpoint(endpoint: str) -> bool:
        parsed = urllib.parse.urlparse(endpoint)
        host = (parsed.hostname or "").strip().lower()
        return host in {"127.0.0.1", "localhost"}

    @staticmethod
    def _is_connection_refused_error(exc: Exception) -> bool:
        return "refused" in str(exc).lower()

    def _should_retry_exception(self, exc: Exception) -> bool:
        if self._is_local_endpoint(self.model_endpoint) and self._is_connection_refused_error(exc):
            return False
        return should_retry(exc)

    @staticmethod
    def _is_transport_failure(exc: Exception) -> bool:
        return getattr(exc, "status_code", None) is None

    def complete(self, payload: dict[str, object], timeout_s: Optional[float] = None) -> dict[str, object]:
        request = urllib.request.Request(
            self.model_endpoint,
            headers={"Content-Type": "application/json", **self.headers()},
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
        )
        timeout = self.request_timeout_s if timeout_s is None else max(0.1, float(timeout_s))
        with urllib.request.urlopen(request, timeout=timeout, context=self.ssl_context) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}

    def stream_completion(
        self,
        payload: dict[str, object],
        stop_event,
        on_event: Optional[Callable[[JsonObject], None]],
        pass_id: str,
        mode: Optional[str] = None,
    ) -> StreamPassResult:
        selected_mode = mode if mode in {"responses", "chat"} else self._select_endpoint_mode()
        return self._stream_one_pass(payload, stop_event, on_event, pass_id, selected_mode)

    def _stream_one_pass(
        self,
        payload: dict[str, object],
        stop_event,
        on_event: Optional[Callable[[JsonObject], None]],
        pass_id: str,
        mode: str,
    ) -> StreamPassResult:
        if self.stop_requested(stop_event):
            self.telemetry.emit("chat_pass_cancelled", pass_id=pass_id)
            return StreamPassResult(finish_reason="cancelled")

        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        finish_reason = "stop"
        tool_acc = ToolCallAccumulator(pass_id=pass_id)
        tool_phase_started = False
        suppress_content_stream = False
        usage: Dict[str, int] = {}
        stream_started_at = time.time()
        first_output_at: Optional[float] = None
        endpoint = self.responses_endpoint if mode == "responses" else self.model_endpoint
        self.telemetry.emit("chat_pass_start", pass_id=pass_id, endpoint=endpoint, payload=payload, mode=mode)

        for chunk in self._stream_chat_completions(
            endpoint=endpoint,
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
            parsed = (
                self._parse_responses_chunk(chunk, tool_acc)
                if mode == "responses"
                else self._parse_chat_chunk(chunk, tool_acc)
            )
            if parsed["usage"]:
                usage = parsed["usage"]
                self._emit(on_event, {"type": "usage", "usage": dict(usage)})
            reasoning = parsed["reasoning"]
            if reasoning:
                if first_output_at is None:
                    first_output_at = time.time()
                reasoning_parts.append(reasoning)
                self._emit(on_event, {"type": "reasoning_token", "text": reasoning})
            content = parsed["content"]
            if content:
                if first_output_at is None:
                    first_output_at = time.time()
                content_parts.append(content)
                if self._contains_tool_markup(content):
                    if not suppress_content_stream:
                        self._emit(
                            on_event,
                            {
                                "type": "info",
                                "text": "Model emitted tool markup; generating a clean final response...",
                            },
                        )
                    suppress_content_stream = True
                if not suppress_content_stream:
                    self._emit(on_event, {"type": "content_token", "text": content})

            tool_deltas = parsed["tool_deltas"]
            if tool_deltas:
                if first_output_at is None:
                    first_output_at = time.time()
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
            if parsed["finish_reason"]:
                finish_reason = parsed["finish_reason"]

        tool_calls = tool_acc.finalize()
        if tool_calls and finish_reason not in {"tool_calls", "cancelled"}:
            finish_reason = "tool_calls"

        self.telemetry.emit(
            "chat_pass_end",
            pass_id=pass_id,
            mode=mode,
            finish_reason=finish_reason,
            content_chars=len("".join(content_parts)),
            reasoning_chars=len("".join(reasoning_parts)),
            tool_call_count=len(tool_calls),
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            tool_calls=[{"id": call.id, "name": call.name, "arguments": call.arguments} for call in tool_calls],
        )
        first_token_latency_ms: Optional[int] = None
        if first_output_at is not None:
            first_token_latency_ms = max(0, int((first_output_at - stream_started_at) * 1000))
        return StreamPassResult(
            finish_reason=finish_reason,
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            tool_calls=tool_calls,
            usage=usage,
            first_token_latency_ms=first_token_latency_ms,
        )

    @staticmethod
    def _normalize_usage(payload: object) -> Dict[str, int]:
        if not isinstance(payload, dict):
            return {}
        usage: Dict[str, int] = {}
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                usage[str(key)] = int(value)
        return usage

    def _parse_chat_chunk(self, chunk: dict[str, object], _tool_acc: ToolCallAccumulator) -> dict[str, object]:
        out: dict[str, object] = {
            "content": "",
            "reasoning": "",
            "tool_deltas": [],
            "finish_reason": "",
            "usage": {},
        }
        choices = chunk.get("choices", [])
        chunk_usage = self._normalize_usage(chunk.get("usage"))
        if chunk_usage:
            out["usage"] = chunk_usage
        if not isinstance(choices, list) or not choices:
            return out
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta", {}) if isinstance(choice.get("delta"), dict) else {}
        out["reasoning"] = str(delta.get("reasoning_content") or "")
        out["content"] = str(delta.get("content") or "")
        tool_deltas = delta.get("tool_calls")
        if isinstance(tool_deltas, list):
            out["tool_deltas"] = tool_deltas
        if choice.get("finish_reason"):
            out["finish_reason"] = str(choice["finish_reason"])
        return out

    def _parse_responses_chunk(self, chunk: dict[str, object], tool_acc: ToolCallAccumulator) -> dict[str, object]:
        out: dict[str, object] = {
            "content": "",
            "reasoning": "",
            "tool_deltas": [],
            "finish_reason": "",
            "usage": {},
        }
        event_type = str(chunk.get("type") or "").strip()
        if not event_type:
            return out

        if event_type in {"response.output_text.delta", "response.output_text.annotation.added"}:
            out["content"] = str(chunk.get("delta") or "")
            return out

        if event_type in {
            "response.reasoning.delta",
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        }:
            out["reasoning"] = str(chunk.get("delta") or "")
            return out

        if event_type == "response.output_item.added":
            item = chunk.get("item") if isinstance(chunk.get("item"), dict) else {}
            item_type = str(item.get("type") or "")
            if item_type in {"function_call", "tool_call"}:
                index = int(chunk.get("output_index") or item.get("index") or 0)
                function_name = str(item.get("name") or item.get("function_name") or "").strip()
                call_id = str(item.get("call_id") or item.get("id") or "").strip()
                args = str(item.get("arguments") or "")
                out["tool_deltas"] = [
                    {
                        "index": index,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": function_name, "arguments": args},
                    }
                ]
            return out

        if event_type == "response.function_call_arguments.delta":
            index = int(chunk.get("output_index") or 0)
            call_id = str(chunk.get("call_id") or chunk.get("item_id") or "").strip()
            delta = str(chunk.get("delta") or "")
            out["tool_deltas"] = [
                {
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {"arguments": delta},
                }
            ]
            return out

        if event_type in {"response.completed", "response.incomplete", "response.failed"}:
            response_obj = chunk.get("response") if isinstance(chunk.get("response"), dict) else {}
            usage = self._normalize_usage(response_obj.get("usage"))
            if usage:
                out["usage"] = usage
            if event_type == "response.completed":
                out["finish_reason"] = "stop"
            elif event_type == "response.incomplete":
                out["finish_reason"] = "length"
            else:
                out["finish_reason"] = "error"
            return out

        if event_type == "response.output_item.done":
            item = chunk.get("item") if isinstance(chunk.get("item"), dict) else {}
            if str(item.get("type") or "") in {"function_call", "tool_call"}:
                index = int(chunk.get("output_index") or item.get("index") or 0)
                call_id = str(item.get("call_id") or item.get("id") or "").strip()
                args = str(item.get("arguments") or "")
                if args:
                    out["tool_deltas"] = [
                        {
                            "index": index,
                            "id": call_id,
                            "type": "function",
                            "function": {"arguments": args},
                        }
                    ]
                if tool_acc.finalize():
                    out["finish_reason"] = "tool_calls"
            return out

        return out

    def call_with_retry(self, payload: dict[str, object], stop_event, on_event, pass_id: str) -> StreamPassResult:
        attempt = 0
        mode = self._select_endpoint_mode()
        fallback_attempted = False
        while True:
            if self.stop_requested(stop_event):
                return StreamPassResult(finish_reason="cancelled")
            try:
                status = self._status_allows_immediate_send()
                if status.state == "offline":
                    message = status.last_error or f"Model endpoint offline: {self.models_endpoint}"
                    raise RuntimeError(f"Model endpoint offline: {message}")
                result = self.stream_completion(payload, stop_event, on_event, pass_id=pass_id, mode=mode)
                self._resolved_endpoint_mode = mode
                self._compatibility.selected_endpoint_mode = mode
                if mode == "responses":
                    self._compatibility.supports_responses = True
                if mode == "chat":
                    self._compatibility.supports_chat = True
                return result
            except Exception as exc:
                if (
                    self.endpoint_mode == "auto"
                    and mode == "responses"
                    and not fallback_attempted
                    and self._is_endpoint_unsupported(exc)
                ):
                    fallback_attempted = True
                    mode = "chat"
                    self._resolved_endpoint_mode = "chat"
                    self._compatibility.selected_endpoint_mode = "chat"
                    self._compatibility.supports_responses = False
                    self._compatibility.supports_chat = True
                    self._compatibility.tier = "tier2-best-effort"
                    event = {
                        "from_mode": "responses",
                        "to_mode": "chat",
                        "reason": str(exc),
                        "at": time.time(),
                    }
                    self._fallback_events.append(event)
                    self.telemetry.emit("endpoint_mode_fallback", pass_id=pass_id, **event)
                    self._emit(
                        on_event,
                        {
                            "type": "info",
                            "text": "Responses endpoint unsupported for this backend; falling back to chat completions.",
                        },
                    )
                    payload = self._payload_to_mode(payload, "chat")
                    continue
                if self._is_transport_failure(exc):
                    self.mark_model_transport_failure(exc)
                if self._should_retry_exception(exc) and attempt < self.per_turn_retries:
                    attempt += 1
                    self._emit(
                        on_event,
                        {"type": "info", "text": f"Retrying request ({attempt}/{self.per_turn_retries})..."},
                    )
                    self.telemetry.emit("request_retry", pass_id=pass_id, attempt=attempt, error=str(exc))
                    if not self.sleep_with_stop(self.retry_backoff_s, stop_event):
                        return StreamPassResult(finish_reason="cancelled")
                    status = self.refresh_model_status(timeout_s=self._status_probe_timeout_s(), force=True)
                    if status.state != "online":
                        ready = self.check_ready(
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
    def _emit(on_event: Optional[Callable[[JsonObject], None]], event: JsonObject) -> None:
        if not on_event:
            return
        try:
            on_event(event)
        except Exception as exc:
            logging.debug("event callback failed: %s", exc)

    @staticmethod
    def _contains_tool_markup(text: str) -> bool:
        if not text:
            return False
        markers = ("<tool_call", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>")
        return any(marker in text.lower() for marker in markers)
