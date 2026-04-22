from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.theme_catalog import DEFAULT_THEME_ID, normalize_theme_id


def _section(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key)
    return value if isinstance(value, dict) else {}


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_int(value: Any, default: int, *, minimum: Optional[int] = None) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    if minimum is not None and parsed < minimum:
        return minimum
    return parsed


def _coerce_float(value: Any, default: float, *, minimum: Optional[float] = None) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    if minimum is not None and parsed < minimum:
        return minimum
    return parsed


def _coerce_optional_positive_int(value: Any) -> Optional[int]:
    if value in (None, "", 0):
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _coerce_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _coerce_string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []
    out: List[str] = []
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
    def from_config(cls, config: Dict[str, Any]) -> UiTimingConfig:
        tui_cfg = _section(config, "tui")
        timing_cfg = _section(tui_cfg, "timing")
        return cls(
            stream_drain_interval_s=_coerce_float(timing_cfg.get("stream_drain_interval_s"), 0.016, minimum=0.001),
            scroll_interval_s=_coerce_float(timing_cfg.get("scroll_interval_s"), 0.05, minimum=0.001),
            model_refresh_interval_s=_coerce_float(timing_cfg.get("model_refresh_interval_s"), 5.0, minimum=0.1),
            model_refresh_fast_interval_s=_coerce_float(
                timing_cfg.get("model_refresh_fast_interval_s"),
                2.0,
                minimum=0.1,
            ),
            model_refresh_fast_window_s=_coerce_float(
                timing_cfg.get("model_refresh_fast_window_s"),
                6.0,
                minimum=0.0,
            ),
            shell_confirm_timeout_s=_coerce_float(timing_cfg.get("shell_confirm_timeout_s"), 60.0, minimum=1.0),
        )


@dataclass(slots=True)
class UiRuntimeConfig:
    theme: str
    chat_log_max_lines: Optional[int]
    tree_compaction_enabled: bool
    inactive_assistant_char_limit: int
    inactive_tool_argument_char_limit: int
    inactive_tool_content_char_limit: int
    timing: UiTimingConfig = field(default_factory=UiTimingConfig)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> UiRuntimeConfig:
        tui_cfg = _section(config, "tui")
        tree_cfg = _section(tui_cfg, "tree_compaction")
        chat_log_max_lines = _coerce_int(tui_cfg.get("chat_log_max_lines"), 5000, minimum=1)
        theme_raw = _coerce_string(tui_cfg.get("theme"), DEFAULT_THEME_ID)
        theme, _ = normalize_theme_id(theme_raw, default=DEFAULT_THEME_ID)
        return cls(
            theme=theme,
            chat_log_max_lines=chat_log_max_lines if chat_log_max_lines > 0 else None,
            tree_compaction_enabled=_coerce_bool(tree_cfg.get("enabled"), True),
            inactive_assistant_char_limit=_coerce_int(
                tree_cfg.get("inactive_assistant_char_limit"),
                12000,
                minimum=1,
            ),
            inactive_tool_argument_char_limit=_coerce_int(
                tree_cfg.get("inactive_tool_argument_char_limit"),
                5000,
                minimum=1,
            ),
            inactive_tool_content_char_limit=_coerce_int(
                tree_cfg.get("inactive_tool_content_char_limit"),
                8000,
                minimum=1,
            ),
            timing=UiTimingConfig.from_config(config),
        )


@dataclass(slots=True)
class ProviderConfig:
    provider_name: str
    model_endpoint: str
    models_endpoint: str
    tls_verify: bool
    ca_bundle_path: str
    allow_cross_host: bool
    request_timeout_s: float
    readiness_timeout_s: float
    readiness_poll_s: float
    connect_timeout_s: float
    per_turn_retries: int
    retry_backoff_s: float
    default_max_tokens: Optional[int]
    auth_header: Optional[str]

    @classmethod
    def from_config(cls, config: Dict[str, Any], *, auth_header: Optional[str] = None) -> ProviderConfig:
        agent_cfg = _section(config, "agent")
        return cls(
            provider_name=_coerce_string(agent_cfg.get("provider"), "openai-compatible"),
            model_endpoint=_coerce_string(
                agent_cfg.get("model_endpoint"),
                "http://127.0.0.1:8080/v1/chat/completions",
            ),
            models_endpoint=_coerce_string(
                agent_cfg.get("models_endpoint"),
                "http://127.0.0.1:8080/v1/models",
            ),
            tls_verify=_coerce_bool(agent_cfg.get("tls_verify"), True),
            ca_bundle_path=_coerce_string(agent_cfg.get("ca_bundle_path"), ""),
            allow_cross_host=_coerce_bool(agent_cfg.get("allow_cross_host_endpoints"), False),
            request_timeout_s=_coerce_float(agent_cfg.get("request_timeout_s"), 180.0, minimum=5.0),
            readiness_timeout_s=_coerce_float(agent_cfg.get("readiness_timeout_s"), 30.0, minimum=1.0),
            readiness_poll_s=_coerce_float(agent_cfg.get("readiness_poll_s"), 0.5, minimum=0.05),
            connect_timeout_s=_coerce_float(agent_cfg.get("connect_timeout_s"), 10.0, minimum=0.1),
            per_turn_retries=_coerce_int(agent_cfg.get("per_turn_retries"), 1, minimum=0),
            retry_backoff_s=_coerce_float(agent_cfg.get("retry_backoff_s"), 0.5, minimum=0.0),
            default_max_tokens=_coerce_optional_positive_int(agent_cfg.get("max_tokens")),
            auth_header=(auth_header or "").strip() or None,
        )


@dataclass(slots=True)
class SkillsRuntimeConfig:
    python_executable: str = sys.executable

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> SkillsRuntimeConfig:
        skills_cfg = _section(config, "skills")
        configured_python = _coerce_string(
            skills_cfg.get("python_executable"),
            sys.executable,
        )
        return cls(
            python_executable=configured_python or sys.executable,
        )
