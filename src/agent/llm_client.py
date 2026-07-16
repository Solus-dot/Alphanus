from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, cast

from agent.provider import OpenAICompatibleProvider
from agent.telemetry import TelemetryEmitter
from core.config_model import ProviderConfig
from core.message_types import ChatMessage
from core.message_types import JsonObject as MessageJsonObject
from core.streaming import stream_chat_completions as _stream_chat_completions
from core.types import JsonObject, ModelStatus, StreamPassResult

stream_chat_completions = _stream_chat_completions


class LLMClient:
    ONLINE_STATUS_TTL_S = OpenAICompatibleProvider.ONLINE_STATUS_TTL_S
    OFFLINE_STATUS_TTL_S = OpenAICompatibleProvider.OFFLINE_STATUS_TTL_S

    def __init__(self, config: dict[str, Any], debug: bool = False, telemetry: TelemetryEmitter | None = None) -> None:
        self.debug = debug
        self.telemetry = telemetry or TelemetryEmitter()
        self.auth_header = None
        self.api_key = ""
        self.auth_source = "none"
        self.reload_config(config)

    def reload_config(self, config: dict[str, Any]) -> None:
        self.config = config
        agent_cfg = config["agent"]
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
        try:
            self.max_classifier_tokens = max(32, int(agent_cfg.get("max_classifier_tokens", 256)))
        except (TypeError, ValueError):
            self.max_classifier_tokens = 256

    def _resolve_auth_header(self, config: dict[str, Any]) -> str | None:
        agent_cfg = config["agent"]
        api_key_ref = str(agent_cfg.get("api_key", "")).strip()
        api_key_env = str(agent_cfg.get("api_key_env", "ALPHANUS_API_KEY")).strip() or "ALPHANUS_API_KEY"
        auth_template = (
            str(agent_cfg.get("auth_header_template", "Authorization: Bearer {api_key}")).strip() or "Authorization: Bearer {api_key}"
        )

        explicit_header = os.environ.get("ALPHANUS_AUTH_HEADER", "").strip()
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
            except Exception as exc:
                self.telemetry.emit("auth_header_template_invalid", template=auth_template, error_type=type(exc).__name__)
                rendered = f"Authorization: Bearer {key}"
            if ":" in rendered:
                self.auth_source = "api_key"
                return rendered.strip()
        if explicit_header:
            self.auth_source = "env"
            return explicit_header
        self.auth_source = "none"
        return None

    def compatibility_profile(self) -> dict[str, object]:
        return self.provider.compatibility_profile()

    def fallback_events(self) -> list[dict[str, object]]:
        return self.provider.fallback_events()

    def backend_profile_info(self) -> dict[str, object]:
        return self.provider.backend_profile_info()

    @staticmethod
    def stop_requested(stop_event) -> bool:
        return OpenAICompatibleProvider.stop_requested(stop_event)

    def get_model_status(self) -> ModelStatus:
        return self.provider.get_model_status()

    def is_model_status_fresh(self, status: ModelStatus | None = None, *, now: float | None = None) -> bool:
        return self.provider.is_model_status_fresh(status, now=now)

    def refresh_model_status(self, timeout_s: float | None = None, force: bool = False) -> ModelStatus:
        return self.provider.refresh_model_status(timeout_s=timeout_s, force=force)

    def ensure_ready(
        self,
        stop_event=None,
        on_event: Callable[[JsonObject], None] | None = None,
        timeout_s: float | None = None,
    ) -> bool | None:
        return self.provider.check_ready(stop_event=stop_event, on_event=on_event, timeout_s=timeout_s)

    def should_fail_fast_on_offline_status(self, status: ModelStatus) -> bool:
        return self.provider.should_fail_fast_on_offline_status(status)

    def friendly_endpoint_error(self, exc_or_message: Exception | str) -> str:
        return self.provider._friendly_endpoint_error(exc_or_message)  # noqa: SLF001

    def build_payload(
        self,
        model_messages: list[ChatMessage],
        thinking: bool,
        tools: list[JsonObject] | None = None,
        *,
        max_tokens_override: int | None = None,
        model_override: str = "",
    ) -> JsonObject:
        message_payload = cast(list[MessageJsonObject], model_messages)
        return self.provider.build_payload(
            model_messages=message_payload,
            thinking=thinking,
            tools=tools,
            max_tokens_override=max_tokens_override,
            model_override=model_override,
        )

    def call_with_retry(self, payload: JsonObject, stop_event, on_event, pass_id: str) -> StreamPassResult:
        payload_obj = cast(dict[str, object], payload)
        return self.provider.call_with_retry(payload_obj, stop_event, on_event, pass_id=pass_id)
