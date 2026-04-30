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
        self.auth_header = None
        self.api_key = ""
        self.auth_source = "none"
        self.reload_config(config)

    def reload_config(self, config: JsonObject) -> None:
        self.config = config
        agent_cfg = config.get("agent", {}) if isinstance(config.get("agent"), dict) else {}
        self.provider_config = ProviderConfig.from_config(config, auth_header=self._resolve_auth_header(config))
        self.provider = OpenAICompatibleProvider(
            self.provider_config,
            telemetry=self.telemetry,
            debug=self.debug,
            stream_chat_completions_fn=lambda *args, **kwargs: stream_chat_completions(*args, **kwargs),
        )
        self.auth_header = self.provider_config.auth_header
        self.connect_timeout_s = self.provider.connect_timeout_s
        self.model_endpoint = self.provider.model_endpoint
        self.responses_endpoint = self.provider.responses_endpoint
        self.models_endpoint = self.provider.models_endpoint
        self.base_url = self.provider.base_url
        self.endpoint_mode = self.provider.endpoint_mode
        self.backend_profile = self.provider.backend_profile_requested
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

    def _resolve_auth_header(self, config: JsonObject) -> Optional[str]:
        agent_cfg = config.get("agent", {}) if isinstance(config.get("agent"), dict) else {}
        api_key_ref = str(agent_cfg.get("api_key", "")).strip()
        api_key_env = str(agent_cfg.get("api_key_env", "ALPHANUS_API_KEY")).strip() or "ALPHANUS_API_KEY"
        auth_template = (
            str(agent_cfg.get("auth_header_template", "Authorization: Bearer {api_key}")).strip()
            or "Authorization: Bearer {api_key}"
        )

        explicit_header = (
            os.environ.get("ALPHANUS_AUTH_HEADER", "").strip()
            or os.environ.get("AUTH_HEADER", "").strip()
            or ""
        )
        if api_key_ref.lower().startswith("env:"):
            env_name = api_key_ref[4:].strip()
            key = os.environ.get(env_name, "").strip() if env_name else ""
        elif api_key_ref:
            key = api_key_ref
        else:
            key = os.environ.get(api_key_env, "").strip()
        self.api_key = key

        if key:
            try:
                rendered = auth_template.format(api_key=key)
            except Exception:
                rendered = f"Authorization: Bearer {key}"
            if ":" in rendered:
                self.auth_source = "api_key"
                return rendered.strip()
        if explicit_header:
            self.auth_source = "env"
            return explicit_header
        self.auth_source = "none"
        return None

    @property
    def _ready_checked(self) -> bool:
        return bool(getattr(self.provider, "_ready_checked", False))

    @_ready_checked.setter
    def _ready_checked(self, value: bool) -> None:
        self.provider._ready_checked = bool(value)

    def headers(self) -> Dict[str, str]:
        return self.provider.headers()

    def compatibility_profile(self) -> Dict[str, object]:
        return self.provider.compatibility_profile()

    def fallback_events(self) -> list[dict[str, object]]:
        return self.provider.fallback_events()

    def backend_profile_info(self) -> Dict[str, object]:
        return self.provider.backend_profile_info()

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
        return self.provider.build_payload(
            model_messages=model_messages,
            thinking=thinking,
            tools=tools,
            max_tokens_override=max_tokens_override,
            model_override=model_override,
        )

    def call_with_retry(self, payload: JsonObject, stop_event, on_event, pass_id: str) -> StreamPassResult:
        return self.provider.call_with_retry(payload, stop_event, on_event, pass_id=pass_id)

    @staticmethod
    def _emit(on_event: Optional[Callable[[JsonObject], None]], event: JsonObject) -> None:
        OpenAICompatibleProvider._emit(on_event, event)

    @staticmethod
    def _contains_tool_markup(text: str) -> bool:
        return OpenAICompatibleProvider._contains_tool_markup(text)
