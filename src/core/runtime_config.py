from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from core.coercion import coerce_bool
from core.configuration import DEFAULT_CONFIG
from core.theme_catalog import DEFAULT_THEME_ID, normalize_theme_id


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    return value if isinstance(value, dict) else {}


def _coerce_bool(value: Any, default: bool) -> bool:
    return coerce_bool(value, default)


def _coerce_int(value: Any, default: int, *, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None and parsed < minimum:
        return minimum
    return parsed


def _coerce_float(value: Any, default: float, *, minimum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None and parsed < minimum:
        return minimum
    return parsed


def _coerce_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


@dataclass(slots=True)
class UiTimingConfig:
    stream_drain_interval_s: float = 0.016
    scroll_interval_s: float = 0.05
    model_refresh_interval_s: float = 5.0
    model_refresh_fast_interval_s: float = 2.0
    model_refresh_fast_window_s: float = 6.0
    shell_confirm_timeout_s: float = 60.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> UiTimingConfig:
        tui_cfg = _section(config, "tui")
        timing_cfg = _section(tui_cfg, "timing")
        default_timing = _section(_section(DEFAULT_CONFIG, "tui"), "timing")
        return cls(
            stream_drain_interval_s=_coerce_float(
                timing_cfg.get("stream_drain_interval_s"),
                float(default_timing["stream_drain_interval_s"]),
                minimum=0.001,
            ),
            scroll_interval_s=_coerce_float(
                timing_cfg.get("scroll_interval_s"),
                float(default_timing["scroll_interval_s"]),
                minimum=0.001,
            ),
            model_refresh_interval_s=_coerce_float(
                timing_cfg.get("model_refresh_interval_s"),
                float(default_timing["model_refresh_interval_s"]),
                minimum=0.1,
            ),
            model_refresh_fast_interval_s=_coerce_float(
                timing_cfg.get("model_refresh_fast_interval_s"),
                float(default_timing["model_refresh_fast_interval_s"]),
                minimum=0.1,
            ),
            model_refresh_fast_window_s=_coerce_float(
                timing_cfg.get("model_refresh_fast_window_s"),
                float(default_timing["model_refresh_fast_window_s"]),
                minimum=0.0,
            ),
            shell_confirm_timeout_s=_coerce_float(
                timing_cfg.get("shell_confirm_timeout_s"),
                float(default_timing["shell_confirm_timeout_s"]),
                minimum=1.0,
            ),
        )


@dataclass(slots=True)
class UiRuntimeConfig:
    theme: str
    chat_log_max_lines: int | None
    tree_compaction_enabled: bool
    inactive_assistant_char_limit: int
    inactive_tool_argument_char_limit: int
    inactive_tool_content_char_limit: int
    timing: UiTimingConfig = field(default_factory=UiTimingConfig)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> UiRuntimeConfig:
        tui_cfg = _section(config, "tui")
        tree_cfg = _section(tui_cfg, "tree_compaction")
        default_tui = _section(DEFAULT_CONFIG, "tui")
        default_tree = _section(default_tui, "tree_compaction")
        chat_log_max_lines = _coerce_int(tui_cfg.get("chat_log_max_lines"), int(default_tui["chat_log_max_lines"]), minimum=0)
        theme_raw = _coerce_string(tui_cfg.get("theme"), str(default_tui["theme"]))
        available_themes = None
        try:
            from tui.themes import available_theme_ids

            available_themes = available_theme_ids()
        except ImportError:
            available_themes = None
        theme, _ = normalize_theme_id(theme_raw, default=DEFAULT_THEME_ID, available=available_themes)
        return cls(
            theme=theme,
            chat_log_max_lines=chat_log_max_lines if chat_log_max_lines > 0 else None,
            tree_compaction_enabled=_coerce_bool(tree_cfg.get("enabled"), bool(default_tree["enabled"])),
            inactive_assistant_char_limit=_coerce_int(
                tree_cfg.get("inactive_assistant_char_limit"),
                int(default_tree["inactive_assistant_char_limit"]),
                minimum=1,
            ),
            inactive_tool_argument_char_limit=_coerce_int(
                tree_cfg.get("inactive_tool_argument_char_limit"),
                int(default_tree["inactive_tool_argument_char_limit"]),
                minimum=1,
            ),
            inactive_tool_content_char_limit=_coerce_int(
                tree_cfg.get("inactive_tool_content_char_limit"),
                int(default_tree["inactive_tool_content_char_limit"]),
                minimum=1,
            ),
            timing=UiTimingConfig.from_config(config),
        )


@dataclass(slots=True)
class ProviderConfig:
    provider_name: str
    base_url: str
    model_endpoint: str
    responses_endpoint: str
    models_endpoint: str
    endpoint_mode: str
    backend_profile: str
    tls_verify: bool
    ca_bundle_path: str
    allow_cross_host: bool
    request_timeout_s: float
    readiness_timeout_s: float
    readiness_poll_s: float
    connect_timeout_s: float
    per_turn_retries: int
    retry_backoff_s: float
    default_max_tokens: int | None
    api_key: str
    api_key_env: str
    auth_header_template: str
    auth_header: str | None

    @classmethod
    def from_config(cls, config: dict[str, Any], *, auth_header: str | None = None) -> ProviderConfig:
        agent_cfg = _section(config, "agent")
        default_agent = _section(DEFAULT_CONFIG, "agent")
        max_tokens_value = agent_cfg.get("max_tokens")
        default_max_tokens = None
        if max_tokens_value not in (None, "", 0):
            parsed_max_tokens = _coerce_int(max_tokens_value, 0)
            if parsed_max_tokens > 0:
                default_max_tokens = parsed_max_tokens
        return cls(
            provider_name=_coerce_string(agent_cfg.get("provider"), str(default_agent["provider"])),
            base_url=_coerce_string(
                agent_cfg.get("base_url"),
                str(default_agent["base_url"]),
            ),
            model_endpoint=_coerce_string(
                agent_cfg.get("model_endpoint"),
                str(default_agent["model_endpoint"]),
            ),
            responses_endpoint=_coerce_string(
                agent_cfg.get("responses_endpoint"),
                str(default_agent["responses_endpoint"]),
            ),
            models_endpoint=_coerce_string(
                agent_cfg.get("models_endpoint"),
                str(default_agent["models_endpoint"]),
            ),
            endpoint_mode=_coerce_string(
                agent_cfg.get("endpoint_mode"),
                str(default_agent["endpoint_mode"]),
            ).lower(),
            backend_profile=_coerce_string(
                agent_cfg.get("backend_profile"),
                str(default_agent["backend_profile"]),
            ).lower(),
            tls_verify=_coerce_bool(agent_cfg.get("tls_verify"), bool(default_agent["tls_verify"])),
            ca_bundle_path=_coerce_string(agent_cfg.get("ca_bundle_path"), str(default_agent["ca_bundle_path"])),
            allow_cross_host=_coerce_bool(
                agent_cfg.get("allow_cross_host_endpoints"),
                bool(default_agent["allow_cross_host_endpoints"]),
            ),
            request_timeout_s=_coerce_float(
                agent_cfg.get("request_timeout_s"),
                float(default_agent["request_timeout_s"]),
                minimum=5.0,
            ),
            readiness_timeout_s=_coerce_float(
                agent_cfg.get("readiness_timeout_s"),
                float(default_agent["readiness_timeout_s"]),
                minimum=1.0,
            ),
            readiness_poll_s=_coerce_float(
                agent_cfg.get("readiness_poll_s"),
                float(default_agent["readiness_poll_s"]),
                minimum=0.05,
            ),
            connect_timeout_s=_coerce_float(
                agent_cfg.get("connect_timeout_s"),
                float(default_agent["connect_timeout_s"]),
                minimum=0.1,
            ),
            per_turn_retries=_coerce_int(agent_cfg.get("per_turn_retries"), int(default_agent["per_turn_retries"]), minimum=0),
            retry_backoff_s=_coerce_float(
                agent_cfg.get("retry_backoff_s"),
                float(default_agent["retry_backoff_s"]),
                minimum=0.0,
            ),
            default_max_tokens=default_max_tokens,
            api_key=_coerce_string(agent_cfg.get("api_key"), str(default_agent["api_key"])),
            api_key_env=_coerce_string(agent_cfg.get("api_key_env"), str(default_agent["api_key_env"])),
            auth_header_template=_coerce_string(
                agent_cfg.get("auth_header_template"),
                str(default_agent["auth_header_template"]),
            ),
            auth_header=(auth_header or "").strip() or None,
        )


@dataclass(slots=True)
class SkillsRuntimeConfig:
    python_executable: str = sys.executable
    paths: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SkillsRuntimeConfig:
        skills_cfg = _section(config, "skills")
        configured_python = _coerce_string(
            skills_cfg.get("python_executable"),
            sys.executable,
        )
        return cls(
            python_executable=configured_python or sys.executable,
            paths=[str(item) for item in skills_cfg.get("paths", []) if str(item).strip()]
            if isinstance(skills_cfg.get("paths", []), list)
            else [],
        )
