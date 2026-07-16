from __future__ import annotations

import json
import logging
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, cast

from agent.backend_profiles import (
    AUTO_BACKEND_PROFILE,
    UNKNOWN_BACKEND_PROFILE,
    detect_backend_profile,
    is_local_backend_profile,
    looks_like_backend_model_fallback_error,
    normalize_backend_profile,
    profile_capabilities,
    rewrite_payload_for_profile,
)
from agent.provider_failure_policy import (
    is_endpoint_unsupported,
    is_local_endpoint,
    is_transport_failure,
    should_retry_provider_exception,
)
from agent.provider_metadata import ProviderMetadataExtractor
from agent.provider_payload import ProviderPayloadAdapter
from agent.provider_stream_parser import ProviderStreamParser
from agent.telemetry import TelemetryEmitter
from core.config_model import ProviderConfig
from core.endpoint_modes import CONCRETE_ENDPOINT_MODES, ENDPOINT_MODE_AUTO, ENDPOINT_MODE_CHAT, ENDPOINT_MODE_RESPONSES, ENDPOINT_MODES
from core.message_types import ToolCallDelta
from core.streaming import build_ssl_context
from core.streaming import stream_chat_completions as core_stream_chat_completions
from core.types import JsonObject, ModelStatus, StreamPassResult, ToolCallAccumulator


@dataclass(slots=True)
class ProviderCompatibilityProfile:
    selected_endpoint_mode: str = ENDPOINT_MODE_CHAT
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
        telemetry: TelemetryEmitter | None = None,
        debug: bool = False,
        stream_chat_completions_fn: Callable[..., object] | None = None,
    ) -> None:
        self.debug = debug
        self.telemetry = telemetry or TelemetryEmitter()
        self._stream_chat_completions = stream_chat_completions_fn or core_stream_chat_completions
        self._payload_adapter = ProviderPayloadAdapter()
        self._metadata = ProviderMetadataExtractor()
        self._stream_parser = ProviderStreamParser()
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
        endpoint_mode = str(config.endpoint_mode or ENDPOINT_MODE_AUTO).strip().lower()
        self.endpoint_mode = endpoint_mode if endpoint_mode in ENDPOINT_MODES else ENDPOINT_MODE_AUTO
        self.backend_profile_requested = normalize_backend_profile(config.backend_profile)
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
        self._resolved_endpoint_mode = ENDPOINT_MODE_CHAT
        self._fallback_events: list[dict[str, object]] = []
        self._compatibility = ProviderCompatibilityProfile(selected_endpoint_mode=ENDPOINT_MODE_CHAT)
        self._backend_profile_detected = UNKNOWN_BACKEND_PROFILE
        self._backend_profile_reason = "not detected"
        self._backend_incompatibility_last = ""
        self._backend_model_integrity_state = "unknown"
        self._refresh_backend_profile(None)

    def headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.auth_header and ":" in self.auth_header:
            key, value = self.auth_header.split(":", 1)
            headers[key.strip()] = value.strip()
        return headers

    def compatibility_profile(self) -> dict[str, object]:
        profile = self._compatibility.to_json()
        profile["backend_profile"] = self.backend_profile_info()
        return profile

    def fallback_events(self) -> list[dict[str, object]]:
        return [dict(item) for item in self._fallback_events[-10:]]

    def backend_profile_info(self) -> dict[str, object]:
        selected = self._selected_backend_profile()
        return {
            "requested": self.backend_profile_requested,
            "detected": self._backend_profile_detected,
            "selected": selected,
            "reason": self._backend_profile_reason,
            "capabilities": profile_capabilities(selected).to_json(),
            "model_integrity": self._backend_model_integrity_state,
            "incompatibility_last": self._backend_incompatibility_last,
        }

    def _selected_backend_profile(self) -> str:
        if self.backend_profile_requested != AUTO_BACKEND_PROFILE:
            return self.backend_profile_requested
        return self._backend_profile_detected

    def _record_backend_incompatibility(self, message: str, *, pass_id: str = "") -> None:
        self._backend_incompatibility_last = str(message or "").strip()
        self._backend_model_integrity_state = "violation"
        payload: dict[str, object] = {
            "profile": self._selected_backend_profile(),
            "reason": self._backend_incompatibility_last,
        }
        if pass_id:
            payload["pass_id"] = pass_id
        self.telemetry.emit("backend_model_integrity_violation", **payload)

    def _refresh_backend_profile(self, models_payload: object | None) -> None:
        detected, reason = detect_backend_profile(
            requested=self.backend_profile_requested,
            base_url=self.base_url,
            model_endpoint=self.model_endpoint,
            responses_endpoint=self.responses_endpoint,
            models_endpoint=self.models_endpoint,
            models_payload=models_payload,
        )
        changed = (detected != self._backend_profile_detected) or (reason != self._backend_profile_reason)
        self._backend_profile_detected = detected
        self._backend_profile_reason = reason
        if changed:
            self.telemetry.emit(
                "backend_profile_detected",
                requested=self.backend_profile_requested,
                detected=detected,
                selected=self._selected_backend_profile(),
                reason=reason,
            )

    def _normalize_payload_for_backend(self, payload: dict[str, object], *, mode: str, pass_id: str) -> dict[str, object]:
        selected = self._selected_backend_profile()
        normalized, changes = rewrite_payload_for_profile(payload, mode=mode, profile=selected)
        if changes:
            self.telemetry.emit(
                "backend_payload_rewrite_applied",
                pass_id=pass_id,
                profile=selected,
                changes=changes,
            )
        return normalized

    def _enforce_model_integrity(self, payload: dict[str, object], status: ModelStatus, *, pass_id: str) -> None:
        selected = self._selected_backend_profile()
        if not is_local_backend_profile(selected):
            return
        if not profile_capabilities(selected).strict_model_integrity:
            self._backend_model_integrity_state = "ok"
            return
        requested_model = str(payload.get("model", "")).strip()
        active_model = str(status.model_name or "").strip()
        if not requested_model or not active_model:
            return
        if requested_model == active_model:
            self._backend_model_integrity_state = "ok"
            return
        message = (
            f"Backend model mismatch for profile '{selected}': requested '{requested_model}' "
            f"but backend reports loaded model '{active_model}'. "
            "Load the requested model first or set an explicit backend profile that supports model switching."
        )
        self._record_backend_incompatibility(message, pass_id=pass_id)
        raise RuntimeError(message)

    def _compatibility_cache_key(self) -> tuple[str, str, str, str]:
        return (
            str(self.base_url),
            str(self.model_endpoint),
            str(self.responses_endpoint),
            f"{self.auth_header or ''}|{self.backend_profile_requested}|{self._backend_profile_detected}",
        )

    def _select_endpoint_mode(self) -> str:
        selected_backend = self._selected_backend_profile()
        backend_capabilities = profile_capabilities(selected_backend)
        if self.endpoint_mode in CONCRETE_ENDPOINT_MODES:
            selected = self.endpoint_mode
            if selected == ENDPOINT_MODE_RESPONSES and not backend_capabilities.supports_responses:
                selected = ENDPOINT_MODE_CHAT
            self._resolved_endpoint_mode = selected
            self._compatibility = ProviderCompatibilityProfile(
                selected_endpoint_mode=selected,
                supports_responses=backend_capabilities.supports_responses and selected == ENDPOINT_MODE_RESPONSES,
                supports_chat=selected == ENDPOINT_MODE_CHAT,
                supports_tools=backend_capabilities.supports_tools,
                supports_stream=True,
                supports_reasoning=True,
                supports_multimodal_input=backend_capabilities.supports_multimodal_input and selected == ENDPOINT_MODE_RESPONSES,
                supports_multimodal_output=selected == ENDPOINT_MODE_RESPONSES,
                supports_structured_output=selected == ENDPOINT_MODE_RESPONSES,
                tier="tier1-guaranteed",
            )
            self._profile_cache_key = self._compatibility_cache_key()
            return selected

        key = self._compatibility_cache_key()
        if key == self._profile_cache_key and self._resolved_endpoint_mode in CONCRETE_ENDPOINT_MODES:
            return self._resolved_endpoint_mode
        self._profile_cache_key = key
        self._resolved_endpoint_mode = ENDPOINT_MODE_RESPONSES if backend_capabilities.supports_responses else ENDPOINT_MODE_CHAT
        self._compatibility = ProviderCompatibilityProfile(
            selected_endpoint_mode=self._resolved_endpoint_mode,
            supports_responses=backend_capabilities.supports_responses,
            supports_chat=True,
            supports_tools=backend_capabilities.supports_tools,
            supports_stream=True,
            supports_reasoning=True,
            supports_multimodal_input=backend_capabilities.supports_multimodal_input,
            supports_multimodal_output=backend_capabilities.supports_multimodal_input
            and self._resolved_endpoint_mode == ENDPOINT_MODE_RESPONSES,
            supports_structured_output=self._resolved_endpoint_mode == ENDPOINT_MODE_RESPONSES,
            tier="tier1-guaranteed",
        )
        return self._resolved_endpoint_mode

    def build_payload(
        self,
        *,
        model_messages: list[JsonObject],
        thinking: bool,
        tools: list[JsonObject] | None = None,
        max_tokens_override: int | None = None,
        model_override: str = "",
        mode: str | None = None,
    ) -> JsonObject:
        selected_mode = mode if mode in CONCRETE_ENDPOINT_MODES else self._select_endpoint_mode()
        return self._payload_adapter.build_payload(
            model_messages=model_messages,
            thinking=thinking,
            tools=tools,
            max_tokens_override=max_tokens_override,
            model_override=model_override,
            mode=selected_mode,
            default_max_tokens=self.default_max_tokens,
        )

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

    def fetch_json(self, url: str, timeout_s: float | None = None) -> object:
        request = urllib.request.Request(url, headers=self.headers(), method="GET")
        timeout = self.connect_timeout_s if timeout_s is None else max(0.1, float(timeout_s))
        with urllib.request.urlopen(request, timeout=timeout, context=self.ssl_context) as response:
            raw = response.read().decode("utf-8")
            if not raw.strip():
                return {}
            return json.loads(raw)

    def list_models(self, timeout_s: float | None = None) -> object:
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

    def is_model_status_fresh(self, status: ModelStatus | None = None, *, now: float | None = None) -> bool:
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

    def _friendly_endpoint_error(self, exc_or_message: Exception | str) -> str:
        raw = str(exc_or_message or "").strip()
        reason = getattr(exc_or_message, "reason", None)
        reason_text = str(reason or "").strip()
        errno_value = getattr(reason, "errno", None)
        text = f"{raw} {reason_text}".lower()
        endpoint = self.models_endpoint
        if isinstance(reason, ConnectionRefusedError) or errno_value in {61, 111} or "connection refused" in text:
            return f"Connection refused by model endpoint {endpoint}. Is the local model server running and listening on that port?"
        if isinstance(reason, TimeoutError) or isinstance(exc_or_message, TimeoutError) or "timed out" in text or "timeout" in text:
            return f"Timed out while contacting model endpoint {endpoint}. The server may be starting, overloaded, or unreachable."
        if isinstance(reason, socket.gaierror) or "name or service not known" in text or "nodename nor servname" in text:
            return f"Could not resolve the model endpoint host for {endpoint}. Check the configured base URL."
        if raw:
            return raw
        return f"Model endpoint is unreachable: {endpoint}"

    def _model_status_error_state(self, exc: Exception) -> Literal["offline", "unknown"]:
        if isinstance(exc, urllib.error.HTTPError):
            return "offline"
        if isinstance(exc, urllib.error.URLError):
            return "offline"
        if isinstance(exc, (ConnectionError, TimeoutError, ConnectionResetError)):
            return "offline"
        return "unknown"

    def refresh_model_status(self, timeout_s: float | None = None, force: bool = False) -> ModelStatus:
        current = self.get_model_status()
        if not force and self.is_model_status_fresh(current):
            return current

        now = time.monotonic()
        deadline = None if timeout_s is None else now + max(0.0, float(timeout_s))

        def remaining_timeout() -> float | None:
            if deadline is None:
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Model status probe timed out")
            return remaining

        context_window = current.context_window
        loaded_model_name: str | None = None
        slots_endpoint = self._metadata.slots_endpoint_from_models_endpoint(self.models_endpoint)
        try:
            slots_payload = self.fetch_json(slots_endpoint, timeout_s=remaining_timeout())
        except Exception as exc:
            self.telemetry.emit("model_slots_fetch_failed", endpoint=slots_endpoint, error=str(exc))
        else:
            context_window = self._metadata.extract_model_context_window(slots_payload) or context_window
            loaded_model_name = self._metadata.extract_model_name(slots_payload) or loaded_model_name

        try:
            payload = self.list_models(timeout_s=remaining_timeout())
        except Exception as exc:
            friendly_error = self._friendly_endpoint_error(exc)
            self.telemetry.emit("model_fetch_failed", endpoint=self.models_endpoint, error=str(exc), user_error=friendly_error)
            return self._store_model_status(
                ModelStatus(
                    state=self._model_status_error_state(exc),
                    model_name=current.model_name,
                    context_window=context_window,
                    last_checked_at=now,
                    last_success_at=current.last_success_at,
                    last_error=friendly_error,
                    endpoint=self.models_endpoint,
                )
            )
        self._refresh_backend_profile(payload)

        model_name = loaded_model_name or self._metadata.extract_model_name(payload) or current.model_name
        if context_window is None:
            context_window = self._metadata.extract_model_context_window(payload)
        if context_window is None:
            props_endpoint = self._metadata.props_endpoint_from_models_endpoint(self.models_endpoint)
            try:
                props_payload = self.fetch_json(props_endpoint, timeout_s=remaining_timeout())
            except Exception as exc:
                self.telemetry.emit("model_props_fetch_failed", endpoint=props_endpoint, error=str(exc))
            else:
                context_window = self._metadata.extract_model_context_window(props_payload)
                model_name = self._metadata.extract_model_name(props_payload) or model_name
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
        friendly_error = self._friendly_endpoint_error(exc)
        self.telemetry.emit("model_transport_failure", endpoint=self.models_endpoint, error=str(exc), user_error=friendly_error)
        self._store_model_status(
            ModelStatus(
                state="offline",
                model_name=current.model_name,
                context_window=current.context_window,
                last_checked_at=now,
                last_success_at=current.last_success_at,
                last_error=friendly_error,
                endpoint=self.models_endpoint,
            )
        )

    def check_ready(
        self,
        stop_event=None,
        on_event: Callable[[JsonObject], None] | None = None,
        timeout_s: float | None = None,
    ) -> bool | None:
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
        if is_local_endpoint(self.models_endpoint) and status.last_success_at <= 0:
            return False
        return True

    def _should_retry_exception(self, exc: Exception) -> bool:
        return should_retry_provider_exception(exc, model_endpoint=self.model_endpoint)

    def stream_completion(
        self,
        payload: dict[str, object],
        stop_event,
        on_event: Callable[[JsonObject], None] | None,
        pass_id: str,
        mode: str | None = None,
    ) -> StreamPassResult:
        selected_mode = mode if mode in CONCRETE_ENDPOINT_MODES else self._select_endpoint_mode()
        return self._stream_one_pass(payload, stop_event, on_event, pass_id, selected_mode)

    def _stream_one_pass(
        self,
        payload: dict[str, object],
        stop_event,
        on_event: Callable[[JsonObject], None] | None,
        pass_id: str,
        mode: str,
    ) -> StreamPassResult:
        if self.stop_requested(stop_event):
            self.telemetry.emit("chat_pass_cancelled", pass_id=pass_id)
            return StreamPassResult(finish_reason="cancelled")

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason = "stop"
        tool_acc = ToolCallAccumulator(pass_id=pass_id)
        tool_phase_started = False
        suppress_content_stream = False
        usage: dict[str, int] = {}
        stream_started_at = time.time()
        first_output_at: float | None = None
        endpoint = self.responses_endpoint if mode == ENDPOINT_MODE_RESPONSES else self.model_endpoint
        payload_messages = payload.get("messages")
        self.telemetry.emit(
            "chat_pass_start",
            pass_id=pass_id,
            endpoint=endpoint,
            mode=mode,
            message_count=len(payload_messages) if isinstance(payload_messages, list) else 0,
        )

        stream_chunks = cast(
            Iterable[dict[str, object]],
            self._stream_chat_completions(
            endpoint=endpoint,
            payload=payload,
            timeout_s=self.request_timeout_s,
            headers=self.headers(),
            ssl_context=self.ssl_context,
            stop_event=stop_event,
            on_debug_event=lambda event: self.telemetry.emit("http_stream", pass_id=pass_id, **event),
            ),
        )
        for chunk in stream_chunks:
            if self.stop_requested(stop_event):
                self.telemetry.emit("chat_pass_cancelled", pass_id=pass_id)
                return StreamPassResult(finish_reason="cancelled")
            parsed_chunk = cast(dict[str, object], chunk)
            parsed = (
                self._stream_parser.parse_responses_chunk(parsed_chunk, tool_acc)
                if mode == ENDPOINT_MODE_RESPONSES
                else self._stream_parser.parse_chat_chunk(parsed_chunk, tool_acc)
            )
            parsed_usage = parsed["usage"]
            if isinstance(parsed_usage, dict) and parsed_usage:
                usage = {str(k): int(v) for k, v in parsed_usage.items() if isinstance(v, (int, float))}
                self._emit(on_event, {"type": "usage", "usage": dict(usage)})
            reasoning = str(parsed.get("reasoning") or "")
            if reasoning:
                if first_output_at is None:
                    first_output_at = time.time()
                reasoning_parts.append(reasoning)
                self._emit(on_event, {"type": "reasoning_token", "text": reasoning})
            content = str(parsed.get("content") or "")
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

            tool_deltas_raw = parsed.get("tool_deltas")
            tool_deltas = cast(list[ToolCallDelta], tool_deltas_raw) if isinstance(tool_deltas_raw, list) else []
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
            parsed_finish_reason = str(parsed.get("finish_reason") or "")
            if parsed_finish_reason:
                finish_reason = parsed_finish_reason

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
            tool_names=[call.name for call in tool_calls],
        )
        first_token_latency_ms: int | None = None
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
                    message = self._friendly_endpoint_error(status.last_error) if status.last_error else f"Model endpoint offline: {self.models_endpoint}"
                    raise RuntimeError(f"Model endpoint offline: {message}")
                normalized_payload = self._normalize_payload_for_backend(payload, mode=mode, pass_id=pass_id)
                if not str(normalized_payload.get("model", "")).strip():
                    model_name = str(status.model_name or "").strip()
                    if model_name:
                        normalized_payload = dict(normalized_payload)
                        normalized_payload["model"] = model_name
                self._enforce_model_integrity(normalized_payload, status, pass_id=pass_id)
                payload = normalized_payload
                result = self.stream_completion(normalized_payload, stop_event, on_event, pass_id=pass_id, mode=mode)
                self._resolved_endpoint_mode = mode
                self._compatibility.selected_endpoint_mode = mode
                if mode == ENDPOINT_MODE_RESPONSES:
                    self._compatibility.supports_responses = True
                if mode == ENDPOINT_MODE_CHAT:
                    self._compatibility.supports_chat = True
                if self._backend_model_integrity_state != "violation":
                    self._backend_model_integrity_state = "ok"
                    self._backend_incompatibility_last = ""
                return result
            except Exception as exc:
                if looks_like_backend_model_fallback_error(str(exc)) and is_local_backend_profile(self._selected_backend_profile()):
                    message = (
                        "Backend attempted implicit model fallback/download during this request. "
                        "Alphanus aborted to preserve model integrity. "
                        "Load the intended model explicitly and retry."
                    )
                    self._record_backend_incompatibility(message, pass_id=pass_id)
                    raise RuntimeError(message) from exc
                if (
                    self.endpoint_mode == ENDPOINT_MODE_AUTO
                    and mode == ENDPOINT_MODE_RESPONSES
                    and not fallback_attempted
                    and is_endpoint_unsupported(exc)
                ):
                    fallback_attempted = True
                    mode = ENDPOINT_MODE_CHAT
                    self._resolved_endpoint_mode = ENDPOINT_MODE_CHAT
                    self._compatibility.selected_endpoint_mode = ENDPOINT_MODE_CHAT
                    self._compatibility.supports_responses = False
                    self._compatibility.supports_chat = True
                    self._compatibility.tier = "tier2-best-effort"
                    event = {
                        "from_mode": ENDPOINT_MODE_RESPONSES,
                        "to_mode": ENDPOINT_MODE_CHAT,
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
                    payload = cast(
                        dict[str, object],
                        self._payload_adapter.payload_to_mode(payload, ENDPOINT_MODE_CHAT, default_max_tokens=self.default_max_tokens),
                    )
                    continue
                if is_transport_failure(exc):
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
    def _emit(on_event: Callable[[JsonObject], None] | None, event: JsonObject) -> None:
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
