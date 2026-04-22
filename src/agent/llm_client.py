from __future__ import annotations

import os
from typing import Callable, Dict, Optional

from core.message_types import ChatMessage
from core.runtime_config import ProviderConfig
from core.streaming import stream_chat_completions as _stream_chat_completions

from agent.provider import OpenAICompatibleProvider
from agent.telemetry import TelemetryEmitter
from core.types import JsonObject, ModelStatus, StreamPassResult

stream_chat_completions = _stream_chat_completions


class LLMClient:
    ONLINE_STATUS_TTL_S = OpenAICompatibleProvider.ONLINE_STATUS_TTL_S
    OFFLINE_STATUS_TTL_S = OpenAICompatibleProvider.OFFLINE_STATUS_TTL_S

    def __init__(self, config: JsonObject, debug: bool = False, telemetry: Optional[TelemetryEmitter] = None) -> None:
        self.debug = debug
        self.telemetry = telemetry or TelemetryEmitter()
        self.auth_header = (
            os.environ.get("ALPHANUS_AUTH_HEADER", "").strip()
            or os.environ.get("AUTH_HEADER", "").strip()
            or None
        )
        self.reload_config(config)

    def reload_config(self, config: JsonObject) -> None:
        self.config = config
        agent_cfg = config.get("agent", {}) if isinstance(config.get("agent"), dict) else {}
        self.provider_config = ProviderConfig.from_config(config, auth_header=self.auth_header)
        self.provider = OpenAICompatibleProvider(
            self.provider_config,
            telemetry=self.telemetry,
            debug=self.debug,
            stream_chat_completions_fn=lambda *args, **kwargs: stream_chat_completions(*args, **kwargs),
        )
        self.connect_timeout_s = self.provider.connect_timeout_s
        self.model_endpoint = self.provider.model_endpoint
        self.models_endpoint = self.provider.models_endpoint
        self.allow_cross_host = self.provider.allow_cross_host
        self.request_timeout_s = self.provider.request_timeout_s
        self.readiness_timeout_s = self.provider.readiness_timeout_s
        self.readiness_poll_s = self.provider.readiness_poll_s
        self.per_turn_retries = self.provider.per_turn_retries
        self.retry_backoff_s = self.provider.retry_backoff_s
        self.default_max_tokens = self.provider.default_max_tokens
        self.ssl_context = self.provider.ssl_context
        self.classifier_model = str(agent_cfg.get("classifier_model", "")).strip()
        self.classifier_use_primary_model = bool(agent_cfg.get("classifier_use_primary_model", True))
        self.enable_structured_classification = bool(agent_cfg.get("enable_structured_classification", True))
        self.max_classifier_tokens = max(32, int(agent_cfg.get("max_classifier_tokens", 256)))

    @property
    def _ready_checked(self) -> bool:
        return bool(getattr(self.provider, "_ready_checked", False))

    @_ready_checked.setter
    def _ready_checked(self, value: bool) -> None:
        self.provider._ready_checked = bool(value)

    def headers(self) -> Dict[str, str]:
        return self.provider.headers()

    @staticmethod
    def stop_requested(stop_event) -> bool:
        return OpenAICompatibleProvider.stop_requested(stop_event)

    @staticmethod
    def sleep_with_stop(duration_s: float, stop_event) -> bool:
        return OpenAICompatibleProvider.sleep_with_stop(duration_s, stop_event)

    @staticmethod
    def extract_model_name(payload: object) -> Optional[str]:
        return OpenAICompatibleProvider.extract_model_name(payload)

    @staticmethod
    def extract_model_context_window(payload: object) -> Optional[int]:
        return OpenAICompatibleProvider.extract_model_context_window(payload)

    @staticmethod
    def props_endpoint_from_models_endpoint(models_endpoint: str) -> str:
        return OpenAICompatibleProvider.props_endpoint_from_models_endpoint(models_endpoint)

    def fetch_json(self, url: str, timeout_s: Optional[float] = None) -> object:
        return self.provider.fetch_json(url, timeout_s=timeout_s)

    def get_model_status(self) -> ModelStatus:
        return self.provider.get_model_status()

    def _store_model_status(self, status: ModelStatus) -> ModelStatus:
        return self.provider._store_model_status(status)

    def is_model_status_fresh(self, status: Optional[ModelStatus] = None, *, now: Optional[float] = None) -> bool:
        return self.provider.is_model_status_fresh(status, now=now)

    def refresh_model_status(self, timeout_s: Optional[float] = None, force: bool = False) -> ModelStatus:
        return self.provider.refresh_model_status(timeout_s=timeout_s, force=force)

    def mark_model_transport_failure(self, exc: Exception) -> None:
        self.provider.mark_model_transport_failure(exc)

    def fetch_model_metadata(self, timeout_s: Optional[float] = None) -> tuple[Optional[str], Optional[int]]:
        status = self.refresh_model_status(timeout_s=timeout_s, force=True)
        return status.model_name, status.context_window

    def ensure_ready(
        self,
        stop_event=None,
        on_event: Optional[Callable[[JsonObject], None]] = None,
        timeout_s: Optional[float] = None,
    ) -> Optional[bool]:
        return self.provider.check_ready(stop_event=stop_event, on_event=on_event, timeout_s=timeout_s)

    def _status_probe_timeout_s(self) -> float:
        return self.provider._status_probe_timeout_s()

    def _status_allows_immediate_send(self) -> ModelStatus:
        return self.provider._status_allows_immediate_send()

    def should_fail_fast_on_offline_status(self, status: ModelStatus) -> bool:
        return self.provider.should_fail_fast_on_offline_status(status)

    @staticmethod
    def _is_local_endpoint(endpoint: str) -> bool:
        return OpenAICompatibleProvider._is_local_endpoint(endpoint)

    @staticmethod
    def _is_connection_refused_error(exc: Exception) -> bool:
        return OpenAICompatibleProvider._is_connection_refused_error(exc)

    def _should_retry_exception(self, exc: Exception) -> bool:
        return self.provider._should_retry_exception(exc)

    @staticmethod
    def _is_transport_failure(exc: Exception) -> bool:
        return OpenAICompatibleProvider._is_transport_failure(exc)

    def build_payload(
        self,
        model_messages: list[ChatMessage],
        thinking: bool,
        tools: Optional[list[JsonObject]] = None,
        *,
        max_tokens_override: Optional[int] = None,
        model_override: str = "",
    ) -> JsonObject:
        payload: JsonObject = {
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

    def call_with_retry(self, payload: JsonObject, stop_event, on_event, pass_id: str) -> StreamPassResult:
        attempt = 0
        while True:
            if self.stop_requested(stop_event):
                return StreamPassResult(finish_reason="cancelled")
            try:
                status = self._status_allows_immediate_send()
                if status.state == "offline":
                    message = status.last_error or f"Model endpoint offline: {self.models_endpoint}"
                    raise RuntimeError(f"Model endpoint offline: {message}")
                return self.provider.stream_completion(payload, stop_event, on_event, pass_id=pass_id)
            except Exception as exc:
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
    def _emit(on_event: Optional[Callable[[JsonObject], None]], event: JsonObject) -> None:
        OpenAICompatibleProvider._emit(on_event, event)

    @staticmethod
    def _contains_tool_markup(text: str) -> bool:
        return OpenAICompatibleProvider._contains_tool_markup(text)
