from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlparse

from core.theme_catalog import DEFAULT_THEME_ID, normalize_theme_id

SCHEMA_VERSION = "1.0.0"
MAX_CONFIG_BYTES = 512 * 1024

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_SECRET_KEYS = {
    "auth_header",
    "authorization",
    "api_key",
    "apikey",
    "tavily_api_key",
    "brave_search_api_key",
    "access_token",
    "refresh_token",
    "bearer_token",
    "password",
    "client_secret",
    "secret",
}
_SECRET_SUFFIXES = ("_api_key", "_apikey", "_token", "_secret", "_password")
_ALLOWED_SEARCH_PROVIDERS = {"tavily", "brave"}
_VALID_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

DEFAULT_CONFIG: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "agent": {
        "provider": "openai-compatible",
        "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
        "models_endpoint": "http://127.0.0.1:8080/v1/models",
        "connect_timeout_s": 10,
        "request_timeout_s": 180,
        "readiness_timeout_s": 30,
        "readiness_poll_s": 0.5,
        "per_turn_retries": 1,
        "retry_backoff_s": 0.5,
        "enable_thinking": True,
        "tls_verify": True,
        "ca_bundle_path": "",
        "allow_cross_host_endpoints": False,
        "max_tokens": None,
        "context_budget_max_tokens": 2048,
        "max_action_depth": 10,
        "max_tool_result_chars": 12000,
        "max_reasoning_chars": 20000,
        "compact_tool_results_in_history": False,
        "compact_tool_result_tools": [],
        "classifier_model": "",
        "classifier_use_primary_model": True,
        "enable_structured_classification": True,
        "max_classifier_tokens": 256,
    },
    "workspace": {
        "path": "~/Desktop/Alphanus-Workspace",
    },
    "memory": {
        "min_score_default": 0.3,
        "recall_min_score_default": 0.18,
        "replace_min_score_default": 0.72,
        "backup_revisions": 2,
    },
    "context": {
        "context_limit": 8192,
        "keep_last_n": 10,
        "safety_margin": 500,
    },
    "capabilities": {
        "shell_require_confirmation": True,
        "dangerously_skip_permissions": False,
    },
    "skills": {
        "strict_capability_policy": False,
        "python_executable": "",
    },
    "agents": {
        "enable_skill_agents": True,
    },
    "runtime": {
        "ask_user_tool": True,
    },
    "tools": {},
    "search": {
        "provider": "tavily",
    },
    "logging": {
        "level": "INFO",
        "format": "json",
        "path": "./logs/runtime.jsonl",
    },
    "tui": {
        "theme": DEFAULT_THEME_ID,
        "chat_log_max_lines": 5000,
        "timing": {
            "stream_drain_interval_s": 0.016,
            "scroll_interval_s": 0.05,
            "model_refresh_interval_s": 5.0,
            "model_refresh_fast_interval_s": 2.0,
            "model_refresh_fast_window_s": 6.0,
            "shell_confirm_timeout_s": 60.0,
        },
        "tree_compaction": {
            "enabled": True,
            "inactive_assistant_char_limit": 12000,
            "inactive_tool_argument_char_limit": 5000,
            "inactive_tool_content_char_limit": 8000,
        },
    },
}


def deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _major(version: str) -> int:
    try:
        return int(str(version).strip().split(".", 1)[0])
    except ValueError as exc:
        raise ValueError(f"Invalid schema_version: {version!r}") from exc


def _warn(warnings: List[str], message: str) -> None:
    warnings.append(message)


def _coerce_bool(value: Any, default: bool, *, path: str, warnings: List[str]) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_VALUES:
            return True
        if lowered in _FALSE_VALUES:
            return False
    _warn(warnings, f"{path}: expected boolean, using default")
    return default


def _coerce_int(
    value: Any,
    default: int,
    *,
    path: str,
    warnings: List[str],
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    original = value
    parsed: int
    if isinstance(value, bool):
        _warn(warnings, f"{path}: expected integer, using default")
        parsed = default
    elif isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        parsed = int(value)
    elif isinstance(value, str):
        try:
            parsed = int(value.strip())
        except Exception:
            _warn(warnings, f"{path}: expected integer, using default")
            parsed = default
    else:
        _warn(warnings, f"{path}: expected integer, using default")
        parsed = default

    if minimum is not None and parsed < minimum:
        _warn(warnings, f"{path}: clamped {original!r} -> {minimum}")
        parsed = minimum
    if maximum is not None and parsed > maximum:
        _warn(warnings, f"{path}: clamped {original!r} -> {maximum}")
        parsed = maximum
    return parsed


def _coerce_float(
    value: Any,
    default: float,
    *,
    path: str,
    warnings: List[str],
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    original = value
    parsed: float
    if isinstance(value, bool):
        _warn(warnings, f"{path}: expected number, using default")
        parsed = default
    elif isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value.strip())
        except Exception:
            _warn(warnings, f"{path}: expected number, using default")
            parsed = default
    else:
        _warn(warnings, f"{path}: expected number, using default")
        parsed = default

    if minimum is not None and parsed < minimum:
        _warn(warnings, f"{path}: clamped {original!r} -> {minimum}")
        parsed = minimum
    if maximum is not None and parsed > maximum:
        _warn(warnings, f"{path}: clamped {original!r} -> {maximum}")
        parsed = maximum
    return parsed


def _coerce_string(
    value: Any,
    default: str,
    *,
    path: str,
    warnings: List[str],
    allow_empty: bool = True,
) -> str:
    if isinstance(value, str):
        out = value.strip()
        if out or allow_empty:
            return out
    _warn(warnings, f"{path}: expected string, using default")
    return default


def _coerce_string_list(value: Any, default: Iterable[str], *, path: str, warnings: List[str]) -> List[str]:
    if not isinstance(value, list):
        if value is not None:
            _warn(warnings, f"{path}: expected list[string], using default")
        return list(default)
    out: List[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _is_secret_key(key: str) -> bool:
    lowered = key.strip().lower()
    if not lowered:
        return False
    if lowered in _SECRET_KEYS:
        return True
    if lowered.endswith(_SECRET_SUFFIXES):
        return True
    return False


def strip_secret_fields(config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    changed = False

    def _strip(obj: Any) -> Any:
        nonlocal changed
        if isinstance(obj, dict):
            cleaned: Dict[str, Any] = {}
            for key, value in obj.items():
                if _is_secret_key(str(key)):
                    changed = True
                    continue
                cleaned[str(key)] = _strip(value)
            return cleaned
        if isinstance(obj, list):
            return [_strip(item) for item in obj]
        return obj

    stripped = _strip(copy.deepcopy(config))
    return (stripped if isinstance(stripped, dict) else {}, changed)


def config_for_editor_view(config: Dict[str, Any]) -> Dict[str, Any]:
    cleaned, _ = strip_secret_fields(config)
    agent_cfg = cleaned.get("agent")
    if isinstance(agent_cfg, dict):
        agent_cfg.pop("context_budget_max_tokens", None)
    context_cfg = cleaned.get("context")
    if isinstance(context_cfg, dict):
        context_cfg.pop("context_limit", None)
        context_cfg.pop("safety_margin", None)
    return cleaned


def _normalize_endpoint(url: Any, *, default: str, path: str, warnings: List[str]) -> str:
    endpoint = _coerce_string(url, default, path=path, warnings=warnings, allow_empty=False)
    parsed = urlparse(endpoint)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        _warn(warnings, f"{path}: invalid URL {endpoint!r}, using default")
        return default
    if parsed.username or parsed.password:
        _warn(warnings, f"{path}: credentials in URL are not allowed, using default")
        return default
    return endpoint


def normalize_config(raw_config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    if not isinstance(raw_config, dict):
        raise ValueError("Global config must be a JSON object")

    warnings: List[str] = []
    schema_raw = str(raw_config.get("schema_version", SCHEMA_VERSION)).strip() or SCHEMA_VERSION
    if _major(schema_raw) != _major(SCHEMA_VERSION):
        raise ValueError(f"Unsupported config schema_version: {schema_raw}")

    sanitized_input, stripped = strip_secret_fields(raw_config)
    if stripped:
        _warn(warnings, "Removed secret-like fields from config; use environment variables for secrets.")

    merged = deep_merge(DEFAULT_CONFIG, sanitized_input)
    merged["schema_version"] = schema_raw

    default_agent = DEFAULT_CONFIG["agent"]
    agent_cfg = merged.get("agent", {}) if isinstance(merged.get("agent"), dict) else {}
    agent_cfg["provider"] = _coerce_string(
        agent_cfg.get("provider"),
        str(default_agent["provider"]),
        path="agent.provider",
        warnings=warnings,
        allow_empty=False,
    )
    agent_cfg["model_endpoint"] = _normalize_endpoint(
        agent_cfg.get("model_endpoint"),
        default=default_agent["model_endpoint"],
        path="agent.model_endpoint",
        warnings=warnings,
    )
    agent_cfg["models_endpoint"] = _normalize_endpoint(
        agent_cfg.get("models_endpoint"),
        default=default_agent["models_endpoint"],
        path="agent.models_endpoint",
        warnings=warnings,
    )
    agent_cfg["connect_timeout_s"] = _coerce_float(
        agent_cfg.get("connect_timeout_s"),
        float(default_agent["connect_timeout_s"]),
        path="agent.connect_timeout_s",
        warnings=warnings,
        minimum=0.1,
        maximum=60.0,
    )
    agent_cfg["request_timeout_s"] = _coerce_float(
        agent_cfg.get("request_timeout_s"),
        float(default_agent["request_timeout_s"]),
        path="agent.request_timeout_s",
        warnings=warnings,
        minimum=5.0,
        maximum=600.0,
    )
    agent_cfg["readiness_timeout_s"] = _coerce_float(
        agent_cfg.get("readiness_timeout_s"),
        float(default_agent["readiness_timeout_s"]),
        path="agent.readiness_timeout_s",
        warnings=warnings,
        minimum=1.0,
        maximum=300.0,
    )
    agent_cfg["readiness_poll_s"] = _coerce_float(
        agent_cfg.get("readiness_poll_s"),
        float(default_agent["readiness_poll_s"]),
        path="agent.readiness_poll_s",
        warnings=warnings,
        minimum=0.05,
        maximum=10.0,
    )
    agent_cfg["per_turn_retries"] = _coerce_int(
        agent_cfg.get("per_turn_retries"),
        int(default_agent["per_turn_retries"]),
        path="agent.per_turn_retries",
        warnings=warnings,
        minimum=0,
        maximum=5,
    )
    agent_cfg["retry_backoff_s"] = _coerce_float(
        agent_cfg.get("retry_backoff_s"),
        float(default_agent["retry_backoff_s"]),
        path="agent.retry_backoff_s",
        warnings=warnings,
        minimum=0.0,
        maximum=30.0,
    )
    agent_cfg["enable_thinking"] = _coerce_bool(
        agent_cfg.get("enable_thinking"),
        bool(default_agent["enable_thinking"]),
        path="agent.enable_thinking",
        warnings=warnings,
    )
    agent_cfg["tls_verify"] = _coerce_bool(
        agent_cfg.get("tls_verify"),
        bool(default_agent["tls_verify"]),
        path="agent.tls_verify",
        warnings=warnings,
    )
    agent_cfg["allow_cross_host_endpoints"] = _coerce_bool(
        agent_cfg.get("allow_cross_host_endpoints"),
        bool(default_agent["allow_cross_host_endpoints"]),
        path="agent.allow_cross_host_endpoints",
        warnings=warnings,
    )
    agent_cfg["ca_bundle_path"] = _coerce_string(
        agent_cfg.get("ca_bundle_path"),
        str(default_agent["ca_bundle_path"]),
        path="agent.ca_bundle_path",
        warnings=warnings,
        allow_empty=True,
    )
    raw_max_tokens = agent_cfg.get("max_tokens")
    if raw_max_tokens is None:
        agent_cfg["max_tokens"] = None
    elif isinstance(raw_max_tokens, str) and raw_max_tokens.strip() in {"", "0"}:
        agent_cfg["max_tokens"] = None
    else:
        parsed_max_tokens = _coerce_int(
            raw_max_tokens,
            0,
            path="agent.max_tokens",
            warnings=warnings,
            minimum=-1,
            maximum=131072,
        )
        agent_cfg["max_tokens"] = parsed_max_tokens if parsed_max_tokens > 0 else None
    agent_cfg["context_budget_max_tokens"] = _coerce_int(
        agent_cfg.get("context_budget_max_tokens"),
        int(default_agent["context_budget_max_tokens"]),
        path="agent.context_budget_max_tokens",
        warnings=warnings,
        minimum=256,
        maximum=262144,
    )
    agent_cfg["max_action_depth"] = _coerce_int(
        agent_cfg.get("max_action_depth"),
        int(default_agent["max_action_depth"]),
        path="agent.max_action_depth",
        warnings=warnings,
        minimum=1,
        maximum=100,
    )
    agent_cfg["max_tool_result_chars"] = _coerce_int(
        agent_cfg.get("max_tool_result_chars"),
        int(default_agent["max_tool_result_chars"]),
        path="agent.max_tool_result_chars",
        warnings=warnings,
        minimum=500,
        maximum=200000,
    )
    agent_cfg["max_reasoning_chars"] = _coerce_int(
        agent_cfg.get("max_reasoning_chars"),
        int(default_agent["max_reasoning_chars"]),
        path="agent.max_reasoning_chars",
        warnings=warnings,
        minimum=0,
        maximum=200000,
    )
    agent_cfg["compact_tool_results_in_history"] = _coerce_bool(
        agent_cfg.get("compact_tool_results_in_history"),
        bool(default_agent["compact_tool_results_in_history"]),
        path="agent.compact_tool_results_in_history",
        warnings=warnings,
    )
    agent_cfg["compact_tool_result_tools"] = _coerce_string_list(
        agent_cfg.get("compact_tool_result_tools"),
        default_agent.get("compact_tool_result_tools", []),
        path="agent.compact_tool_result_tools",
        warnings=warnings,
    )
    agent_cfg["classifier_model"] = _coerce_string(
        agent_cfg.get("classifier_model"),
        str(default_agent.get("classifier_model", "")),
        path="agent.classifier_model",
        warnings=warnings,
    )
    agent_cfg["classifier_use_primary_model"] = _coerce_bool(
        agent_cfg.get("classifier_use_primary_model"),
        bool(default_agent.get("classifier_use_primary_model", True)),
        path="agent.classifier_use_primary_model",
        warnings=warnings,
    )
    agent_cfg["enable_structured_classification"] = _coerce_bool(
        agent_cfg.get("enable_structured_classification"),
        bool(default_agent.get("enable_structured_classification", True)),
        path="agent.enable_structured_classification",
        warnings=warnings,
    )
    agent_cfg["max_classifier_tokens"] = _coerce_int(
        agent_cfg.get("max_classifier_tokens"),
        int(default_agent.get("max_classifier_tokens", 256)),
        path="agent.max_classifier_tokens",
        warnings=warnings,
        minimum=32,
        maximum=4096,
    )
    raw_budgets = agent_cfg.get("tool_budgets")
    if isinstance(raw_budgets, dict):
        clean_budgets: Dict[str, int] = {}
        for key, value in raw_budgets.items():
            clean_key = str(key).strip()
            if not clean_key:
                continue
            clean_budgets[clean_key] = _coerce_int(
                value,
                2,
                path=f"agent.tool_budgets.{clean_key}",
                warnings=warnings,
                minimum=1,
                maximum=20,
            )
        agent_cfg["tool_budgets"] = clean_budgets
    elif raw_budgets is not None:
        _warn(warnings, "agent.tool_budgets: expected object, dropping invalid value")
        agent_cfg.pop("tool_budgets", None)
    merged["agent"] = agent_cfg

    workspace_cfg = merged.get("workspace", {}) if isinstance(merged.get("workspace"), dict) else {}
    workspace_cfg["path"] = _coerce_string(
        workspace_cfg.get("path"),
        str(DEFAULT_CONFIG["workspace"]["path"]),
        path="workspace.path",
        warnings=warnings,
        allow_empty=False,
    )
    merged["workspace"] = workspace_cfg

    raw_memory_cfg = merged.get("memory", {}) if isinstance(merged.get("memory"), dict) else {}
    memory_cfg: Dict[str, Any] = {}
    memory_cfg["min_score_default"] = _coerce_float(
        raw_memory_cfg.get("min_score_default"),
        float(DEFAULT_CONFIG["memory"]["min_score_default"]),
        path="memory.min_score_default",
        warnings=warnings,
        minimum=0.0,
        maximum=1.0,
    )
    memory_cfg["recall_min_score_default"] = _coerce_float(
        raw_memory_cfg.get("recall_min_score_default"),
        float(DEFAULT_CONFIG["memory"]["recall_min_score_default"]),
        path="memory.recall_min_score_default",
        warnings=warnings,
        minimum=0.0,
        maximum=1.0,
    )
    memory_cfg["replace_min_score_default"] = _coerce_float(
        raw_memory_cfg.get("replace_min_score_default"),
        float(DEFAULT_CONFIG["memory"]["replace_min_score_default"]),
        path="memory.replace_min_score_default",
        warnings=warnings,
        minimum=0.0,
        maximum=1.0,
    )
    memory_cfg["backup_revisions"] = _coerce_int(
        raw_memory_cfg.get("backup_revisions"),
        int(DEFAULT_CONFIG["memory"]["backup_revisions"]),
        path="memory.backup_revisions",
        warnings=warnings,
        minimum=0,
        maximum=20,
    )
    merged["memory"] = memory_cfg

    context_cfg = merged.get("context", {}) if isinstance(merged.get("context"), dict) else {}
    context_cfg["context_limit"] = _coerce_int(
        context_cfg.get("context_limit"),
        int(DEFAULT_CONFIG["context"]["context_limit"]),
        path="context.context_limit",
        warnings=warnings,
        minimum=512,
        maximum=262144,
    )
    context_cfg["keep_last_n"] = _coerce_int(
        context_cfg.get("keep_last_n"),
        int(DEFAULT_CONFIG["context"]["keep_last_n"]),
        path="context.keep_last_n",
        warnings=warnings,
        minimum=1,
        maximum=100,
    )
    context_cfg["safety_margin"] = _coerce_int(
        context_cfg.get("safety_margin"),
        int(DEFAULT_CONFIG["context"]["safety_margin"]),
        path="context.safety_margin",
        warnings=warnings,
        minimum=0,
        maximum=100000,
    )
    if context_cfg["safety_margin"] >= context_cfg["context_limit"]:
        adjusted = max(0, context_cfg["context_limit"] // 4)
        _warn(warnings, f"context.safety_margin: reduced to {adjusted} because it exceeded context_limit")
        context_cfg["safety_margin"] = adjusted
    merged["context"] = context_cfg

    caps_cfg = merged.get("capabilities", {}) if isinstance(merged.get("capabilities"), dict) else {}
    caps_cfg["shell_require_confirmation"] = _coerce_bool(
        caps_cfg.get("shell_require_confirmation"),
        bool(DEFAULT_CONFIG["capabilities"]["shell_require_confirmation"]),
        path="capabilities.shell_require_confirmation",
        warnings=warnings,
    )
    caps_cfg["dangerously_skip_permissions"] = _coerce_bool(
        caps_cfg.get("dangerously_skip_permissions"),
        bool(DEFAULT_CONFIG["capabilities"]["dangerously_skip_permissions"]),
        path="capabilities.dangerously_skip_permissions",
        warnings=warnings,
    )
    merged["capabilities"] = caps_cfg

    skills_cfg = merged.get("skills", {}) if isinstance(merged.get("skills"), dict) else {}
    skills_cfg["strict_capability_policy"] = _coerce_bool(
        skills_cfg.get("strict_capability_policy"),
        bool(DEFAULT_CONFIG["skills"]["strict_capability_policy"]),
        path="skills.strict_capability_policy",
        warnings=warnings,
    )
    skills_cfg["python_executable"] = _coerce_string(
        skills_cfg.get("python_executable"),
        str(DEFAULT_CONFIG["skills"]["python_executable"]),
        path="skills.python_executable",
        warnings=warnings,
    )
    skills_cfg.pop("load", None)
    merged["skills"] = skills_cfg

    agents_cfg = merged.get("agents", {}) if isinstance(merged.get("agents"), dict) else {}
    agents_cfg["enable_skill_agents"] = _coerce_bool(
        agents_cfg.get("enable_skill_agents"),
        bool(DEFAULT_CONFIG["agents"]["enable_skill_agents"]),
        path="agents.enable_skill_agents",
        warnings=warnings,
    )
    merged["agents"] = agents_cfg

    runtime_cfg = merged.get("runtime", {}) if isinstance(merged.get("runtime"), dict) else {}
    runtime_cfg["ask_user_tool"] = _coerce_bool(
        runtime_cfg.get("ask_user_tool"),
        bool(DEFAULT_CONFIG["runtime"]["ask_user_tool"]),
        path="runtime.ask_user_tool",
        warnings=warnings,
    )
    merged["runtime"] = runtime_cfg

    tools_cfg = merged.get("tools", {}) if isinstance(merged.get("tools"), dict) else {}
    tools_cfg.pop("enabled_toolsets", None)
    tools_cfg.pop("disabled_toolsets", None)
    merged["tools"] = tools_cfg

    search_cfg = merged.get("search", {}) if isinstance(merged.get("search"), dict) else {}
    provider = _coerce_string(
        search_cfg.get("provider"),
        str(DEFAULT_CONFIG["search"]["provider"]),
        path="search.provider",
        warnings=warnings,
        allow_empty=False,
    ).lower()
    if provider not in _ALLOWED_SEARCH_PROVIDERS:
        _warn(warnings, f"search.provider: unsupported {provider!r}, using default")
        provider = str(DEFAULT_CONFIG["search"]["provider"])
    search_cfg["provider"] = provider
    merged["search"] = search_cfg

    logging_cfg = merged.get("logging", {}) if isinstance(merged.get("logging"), dict) else {}
    level = _coerce_string(
        logging_cfg.get("level"),
        str(DEFAULT_CONFIG["logging"]["level"]),
        path="logging.level",
        warnings=warnings,
        allow_empty=False,
    ).upper()
    if level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        _warn(warnings, f"logging.level: unsupported {level!r}, using default")
        level = str(DEFAULT_CONFIG["logging"]["level"])
    logging_cfg["level"] = level
    fmt = _coerce_string(
        logging_cfg.get("format"),
        str(DEFAULT_CONFIG["logging"]["format"]),
        path="logging.format",
        warnings=warnings,
        allow_empty=False,
    ).lower()
    if fmt not in {"plain", "json"}:
        _warn(warnings, f"logging.format: unsupported {fmt!r}, using default")
        fmt = str(DEFAULT_CONFIG["logging"]["format"])
    logging_cfg["format"] = fmt
    logging_cfg["path"] = _coerce_string(
        logging_cfg.get("path"),
        str(DEFAULT_CONFIG["logging"]["path"]),
        path="logging.path",
        warnings=warnings,
    )
    merged["logging"] = logging_cfg

    tui_cfg = merged.get("tui", {}) if isinstance(merged.get("tui"), dict) else {}
    raw_theme = _coerce_string(
        tui_cfg.get("theme"),
        str(DEFAULT_CONFIG["tui"]["theme"]),
        path="tui.theme",
        warnings=warnings,
        allow_empty=False,
    )
    resolved_theme, changed = normalize_theme_id(raw_theme, default=str(DEFAULT_THEME_ID))
    if changed:
        if raw_theme.strip():
            _warn(warnings, f"tui.theme: unsupported {raw_theme!r}, using {resolved_theme!r}")
        else:
            _warn(warnings, f"tui.theme: empty value, using {resolved_theme!r}")
    tui_cfg["theme"] = resolved_theme
    tui_cfg["chat_log_max_lines"] = _coerce_int(
        tui_cfg.get("chat_log_max_lines"),
        int(DEFAULT_CONFIG["tui"]["chat_log_max_lines"]),
        path="tui.chat_log_max_lines",
        warnings=warnings,
        minimum=100,
        maximum=200000,
    )
    timing_cfg = tui_cfg.get("timing", {})
    if not isinstance(timing_cfg, dict):
        _warn(warnings, "tui.timing: expected object, using defaults")
        timing_cfg = {}
    default_timing = DEFAULT_CONFIG["tui"]["timing"]
    timing_cfg["stream_drain_interval_s"] = _coerce_float(
        timing_cfg.get("stream_drain_interval_s"),
        float(default_timing["stream_drain_interval_s"]),
        path="tui.timing.stream_drain_interval_s",
        warnings=warnings,
        minimum=0.001,
        maximum=1.0,
    )
    timing_cfg["scroll_interval_s"] = _coerce_float(
        timing_cfg.get("scroll_interval_s"),
        float(default_timing["scroll_interval_s"]),
        path="tui.timing.scroll_interval_s",
        warnings=warnings,
        minimum=0.001,
        maximum=1.0,
    )
    timing_cfg["model_refresh_interval_s"] = _coerce_float(
        timing_cfg.get("model_refresh_interval_s"),
        float(default_timing["model_refresh_interval_s"]),
        path="tui.timing.model_refresh_interval_s",
        warnings=warnings,
        minimum=0.1,
        maximum=60.0,
    )
    timing_cfg["model_refresh_fast_interval_s"] = _coerce_float(
        timing_cfg.get("model_refresh_fast_interval_s"),
        float(default_timing["model_refresh_fast_interval_s"]),
        path="tui.timing.model_refresh_fast_interval_s",
        warnings=warnings,
        minimum=0.1,
        maximum=60.0,
    )
    timing_cfg["model_refresh_fast_window_s"] = _coerce_float(
        timing_cfg.get("model_refresh_fast_window_s"),
        float(default_timing["model_refresh_fast_window_s"]),
        path="tui.timing.model_refresh_fast_window_s",
        warnings=warnings,
        minimum=0.0,
        maximum=300.0,
    )
    timing_cfg["shell_confirm_timeout_s"] = _coerce_float(
        timing_cfg.get("shell_confirm_timeout_s"),
        float(default_timing["shell_confirm_timeout_s"]),
        path="tui.timing.shell_confirm_timeout_s",
        warnings=warnings,
        minimum=1.0,
        maximum=600.0,
    )
    tui_cfg["timing"] = timing_cfg
    tree_cfg = tui_cfg.get("tree_compaction", {})
    if not isinstance(tree_cfg, dict):
        _warn(warnings, "tui.tree_compaction: expected object, using defaults")
        tree_cfg = {}
    default_tree = DEFAULT_CONFIG["tui"]["tree_compaction"]
    tree_cfg["enabled"] = _coerce_bool(
        tree_cfg.get("enabled"),
        bool(default_tree["enabled"]),
        path="tui.tree_compaction.enabled",
        warnings=warnings,
    )
    tree_cfg["inactive_assistant_char_limit"] = _coerce_int(
        tree_cfg.get("inactive_assistant_char_limit"),
        int(default_tree["inactive_assistant_char_limit"]),
        path="tui.tree_compaction.inactive_assistant_char_limit",
        warnings=warnings,
        minimum=1000,
        maximum=200000,
    )
    tree_cfg["inactive_tool_argument_char_limit"] = _coerce_int(
        tree_cfg.get("inactive_tool_argument_char_limit"),
        int(default_tree["inactive_tool_argument_char_limit"]),
        path="tui.tree_compaction.inactive_tool_argument_char_limit",
        warnings=warnings,
        minimum=500,
        maximum=100000,
    )
    tree_cfg["inactive_tool_content_char_limit"] = _coerce_int(
        tree_cfg.get("inactive_tool_content_char_limit"),
        int(default_tree["inactive_tool_content_char_limit"]),
        path="tui.tree_compaction.inactive_tool_content_char_limit",
        warnings=warnings,
        minimum=1000,
        maximum=200000,
    )
    tui_cfg["tree_compaction"] = tree_cfg
    merged["tui"] = tui_cfg

    return merged, warnings


def validate_endpoint_policy(config: Dict[str, Any]) -> None:
    agent_cfg = config.get("agent", {}) if isinstance(config.get("agent"), dict) else {}
    model_endpoint = str(agent_cfg.get("model_endpoint", "")).strip()
    models_endpoint = str(agent_cfg.get("models_endpoint", "")).strip()
    allow_cross = bool(agent_cfg.get("allow_cross_host_endpoints", False))

    parsed_model = urlparse(model_endpoint)
    parsed_models = urlparse(models_endpoint)
    if parsed_model.scheme.lower() not in {"http", "https"} or not parsed_model.hostname:
        raise ValueError("agent.model_endpoint must be a valid http(s) URL")
    if parsed_models.scheme.lower() not in {"http", "https"} or not parsed_models.hostname:
        raise ValueError("agent.models_endpoint must be a valid http(s) URL")

    host_a = (parsed_model.hostname or "").lower()
    host_b = (parsed_models.hostname or "").lower()
    if host_a != host_b and not allow_cross:
        raise ValueError("agent.model_endpoint and agent.models_endpoint must share host")


def _load_existing_global_config(path: Path, *, warnings: List[str] | None = None) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Global config not found at {path}")

    size = path.stat().st_size
    if size > MAX_CONFIG_BYTES:
        raise ValueError(f"Global config is too large ({size} bytes); limit is {MAX_CONFIG_BYTES} bytes")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid global config JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Global config must be a JSON object")

    sanitized_raw, stripped = strip_secret_fields(raw)
    if stripped:
        path.write_text(json.dumps(sanitized_raw, indent=2) + "\n", encoding="utf-8")
        raw = sanitized_raw
        if warnings is not None:
            warnings.append("Removed secret-like fields from config/global_config.json on disk.")

    normalized, local_warnings = normalize_config(raw)
    if warnings is not None:
        warnings.extend(local_warnings)
    return normalized


def load_global_config(path: Path, *, warnings: List[str] | None = None) -> Dict[str, Any]:
    return _load_existing_global_config(path, warnings=warnings)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not _VALID_ENV_NAME_RE.match(key):
            continue
        value = value.strip()
        if value and ((value[0] == value[-1]) and value[0] in {"'", '"'}):
            value = value[1:-1]
        if "\x00" in value:
            continue
        if key not in os.environ:
            os.environ[key] = value


def resolve_path(path_str: str, root: Path) -> str:
    path = Path(os.path.expanduser(path_str))
    if not path.is_absolute():
        path = (root / path).resolve()
    return str(path)
