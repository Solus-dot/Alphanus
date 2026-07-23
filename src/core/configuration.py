from __future__ import annotations

import copy
import re
import tomllib
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import tomli_w

from core.backend_profiles import AUTO_BACKEND_PROFILE, VALID_BACKEND_PROFILES
from core.coercion import parse_bool
from core.endpoint_modes import (
    ENDPOINT_MODE_AUTO,
    ENDPOINT_MODES,
    OPENAI_CHAT_COMPLETIONS_PATH,
    OPENAI_MODELS_PATH,
    OPENAI_RESPONSES_PATH,
)
from core.errors import ConfigurationError
from core.search_providers import SEARCH_FALLBACK_NONE, SEARCH_FALLBACK_PROVIDERS, SEARCH_PROVIDERS
from core.theme_catalog import DEFAULT_THEME_ID, normalize_theme_id

MAX_CONFIG_BYTES = 512 * 1024
CONFIG_VERSION = 1

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
PROJECT_ROOT_STRATEGIES = {"git-or-cwd"}
PERMISSION_MODES = {"read-only", "project-write", "danger-full-access"}
APPROVAL_MODES = {"on-boundary"}
SANDBOX_BACKENDS = {"auto", "macos-seatbelt", "linux-bubblewrap", "windows-native"}
_VALID_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _LazyDefaults(Mapping[str, Any]):
    _data: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._data is None:
            from core.config_model import default_config

            self._data = default_config()
        return self._data

    def __getitem__(self, key: str) -> Any:
        return self._load()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        return copy.deepcopy(self._load(), memo)


DEFAULT_CONFIG: Mapping[str, Any] = _LazyDefaults()


def deep_merge(base: Mapping[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = {key: copy.deepcopy(value) for key, value in base.items()}
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _warn(warnings: list[str], message: str) -> None:
    warnings.append(message)


def _coerce_bool(value: Any, default: bool, *, path: str, warnings: list[str]) -> bool:
    parsed = parse_bool(value)
    if parsed is not None:
        return parsed
    _warn(warnings, f"{path}: expected boolean, using default")
    return default


def _coerce_number(
    value: Any,
    default: int | float,
    converter: Callable[[Any], int | float],
    label: str,
    *,
    path: str,
    warnings: list[str],
    minimum: int | float | None = None,
    maximum: int | float | None = None,
) -> int | float:
    original = value
    if isinstance(value, bool):
        _warn(warnings, f"{path}: expected {label}, using default")
        parsed = default
    else:
        try:
            parsed = converter(value.strip() if isinstance(value, str) else value)
        except (TypeError, ValueError, OverflowError):
            _warn(warnings, f"{path}: expected {label}, using default")
            parsed = default
    if minimum is not None and parsed < minimum:
        _warn(warnings, f"{path}: clamped {original!r} -> {minimum}")
        parsed = minimum
    if maximum is not None and parsed > maximum:
        _warn(warnings, f"{path}: clamped {original!r} -> {maximum}")
        parsed = maximum
    return parsed


def _coerce_int(value: Any, default: int, *, path: str, warnings: list[str], minimum: int | None = None, maximum: int | None = None) -> int:
    return int(_coerce_number(value, default, int, "integer", path=path, warnings=warnings, minimum=minimum, maximum=maximum))


def _coerce_float(
    value: Any,
    default: float,
    *,
    path: str,
    warnings: list[str],
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    return float(_coerce_number(value, default, float, "number", path=path, warnings=warnings, minimum=minimum, maximum=maximum))


def _coerce_string(
    value: Any,
    default: str,
    *,
    path: str,
    warnings: list[str],
    allow_empty: bool = True,
) -> str:
    if isinstance(value, str):
        out = value.strip()
        if out or allow_empty:
            return out
    _warn(warnings, f"{path}: expected string, using default")
    return default


def _coerce_string_list(value: Any, default: Iterable[str], *, path: str, warnings: list[str]) -> list[str]:
    if not isinstance(value, list):
        if value is not None:
            _warn(warnings, f"{path}: expected list[string], using default")
        return list(default)
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _normalize_model(
    values: dict[str, Any], model_type: Any, prefix: str, warnings: list[str], *, exclude: Iterable[str] = ()
) -> dict[str, Any]:
    defaults = model_type().model_dump()
    normalized = dict(values) if model_type.model_config.get("extra") == "allow" else {}
    excluded = set(exclude)
    for key, field in model_type.model_fields.items():
        if key in excluded or key not in defaults:
            continue
        default = defaults[key]
        options = {"path": f"{prefix}.{key}", "warnings": warnings}
        minimum = maximum = None
        for constraint in field.metadata:
            minimum = getattr(constraint, "ge", minimum)
            maximum = getattr(constraint, "le", maximum)
        value = values.get(key)
        kind = field.annotation if field.annotation in {bool, int, float, str, list} else type(default)
        if kind is bool:
            normalized[key] = _coerce_bool(value, default, **options)
        elif kind is int:
            normalized[key] = _coerce_int(value, default, minimum=minimum, maximum=maximum, **options)
        elif kind is float:
            normalized[key] = _coerce_float(value, default, minimum=minimum, maximum=maximum, **options)
        elif kind is list:
            normalized[key] = _coerce_string_list(value, default, **options)
        elif kind is str:
            normalized[key] = _coerce_string(value, default, **options)
    return normalized


def _normalize_choice(
    value: Any,
    default: str,
    allowed: Iterable[str],
    *,
    path: str,
    warnings: list[str],
    upper: bool = False,
) -> str:
    choice = _coerce_string(value, default, path=path, warnings=warnings, allow_empty=False)
    choice = choice.upper() if upper else choice.lower()
    if choice not in allowed:
        _warn(warnings, f"{path}: unsupported {choice!r}, using {default!r}")
        return default
    return choice


class ConfigMigrationError(ConfigurationError):
    pass


def _legacy_config_errors(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if isinstance(config.get("workspace"), dict) or "workspace" in config:
        errors.append(
            "`workspace.path` was removed. Launch from the project directory or pass `--project-root`; configure `project.root_strategy` instead."
        )
    caps = config.get("capabilities")
    if isinstance(caps, dict):
        for key in ("permission_profile", "shell_require_confirmation", "dangerously_skip_permissions"):
            if key in caps:
                replacement = {
                    "permission_profile": "`permissions.mode`",
                    "shell_require_confirmation": "`permissions.approvals`",
                    "dangerously_skip_permissions": '`permissions.mode = "danger-full-access"`',
                }[key]
                errors.append(f"`capabilities.{key}` was removed. Use {replacement}.")
    elif "capabilities" in config:
        errors.append("`capabilities` was removed. Use `permissions` and `sandbox`.")
    runtime = config.get("runtime")
    if isinstance(runtime, dict) and "profile" in runtime:
        errors.append("`runtime.profile` was removed. Use `permissions.mode`.")
    return errors


def strip_secret_fields(config: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    changed = False

    def _strip(obj: Any) -> Any:
        nonlocal changed
        if isinstance(obj, Mapping):
            cleaned: dict[str, Any] = {}
            for key, value in obj.items():
                key_text = str(key)
                lowered_key = key_text.strip().lower()
                if lowered_key and (lowered_key in _SECRET_KEYS or lowered_key.endswith(_SECRET_SUFFIXES)):
                    # Keep explicit environment references so config can point to a secret
                    # without storing the plaintext value on disk.
                    if lowered_key in {"api_key", "apikey"} and isinstance(value, str) and value.strip().lower().startswith("env:"):
                        cleaned[key_text] = value.strip()
                        continue
                    changed = True
                    continue
                cleaned[key_text] = _strip(value)
            return cleaned
        if isinstance(obj, list):
            return [_strip(item) for item in obj]
        return obj

    stripped = _strip(copy.deepcopy(config))
    return (stripped if isinstance(stripped, dict) else {}, changed)


def config_for_editor_view(config: Mapping[str, Any]) -> dict[str, Any]:
    cleaned, _ = strip_secret_fields(config)
    agent_cfg = cleaned.get("agent")
    if isinstance(agent_cfg, dict):
        agent_cfg.pop("context_budget_max_tokens", None)
    context_cfg = cleaned.get("context")
    if isinstance(context_cfg, dict):
        context_cfg.pop("context_limit", None)
        context_cfg.pop("safety_margin", None)
    return cleaned


def _normalize_url(url: Any, *, default: str, path: str, warnings: list[str], base_only: bool) -> str:
    endpoint = _coerce_string(url, default, path=path, warnings=warnings, allow_empty=False)
    parsed = urlparse(endpoint)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        _warn(warnings, f"{path}: invalid URL {endpoint!r}, using default")
        return default
    if parsed.username or parsed.password:
        _warn(warnings, f"{path}: credentials in URL are not allowed, using default")
        return default
    if not base_only:
        return endpoint
    if parsed.path.rstrip("/"):
        _warn(warnings, f"{path}: dropped path component from {endpoint!r}")
    return parsed._replace(path="", params="", query="", fragment="").geturl().rstrip("/")


def _normalize_endpoint(url: Any, *, default: str, path: str, warnings: list[str]) -> str:
    return _normalize_url(url, default=default, path=path, warnings=warnings, base_only=False)


def _normalize_base_url(url: Any, *, default: str, path: str, warnings: list[str]) -> str:
    return _normalize_url(url, default=default, path=path, warnings=warnings, base_only=True)


def _endpoint_from_base_url(base_url: str, endpoint_path: str) -> str:
    suffix = endpoint_path if endpoint_path.startswith("/") else f"/{endpoint_path}"
    return f"{base_url.rstrip('/')}{suffix}"


def _normalize_env_name(value: Any, *, default: str, path: str, warnings: list[str]) -> str:
    env_name = _coerce_string(value, default, path=path, warnings=warnings, allow_empty=False)
    if not _VALID_ENV_NAME_RE.match(env_name):
        _warn(warnings, f"{path}: invalid env name {env_name!r}, using default")
        return default
    return env_name


def normalize_config(raw_config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(raw_config, dict):
        raise ValueError("Global config must be a JSON object")

    from core.config_model import (
        AgentConfig,
        AgentsConfig,
        ContextConfig,
        EmbeddingsConfig,
        LoggingConfig,
        MemoryConfig,
        PermissionsConfig,
        ProjectConfig,
        RetrievalConfig,
        RuntimeConfig,
        SandboxConfig,
        SearchConfig,
        SkillsConfig,
        TreeCompactionConfig,
        UiConfig,
        UiTimingConfig,
        validated_config,
    )

    warnings: list[str] = []
    sanitized_input, stripped = strip_secret_fields(raw_config)
    legacy_errors = _legacy_config_errors(sanitized_input)
    if legacy_errors:
        detail = "\n".join(f"- {item}" for item in legacy_errors)
        raise ConfigMigrationError(
            "Config uses removed workspace-era keys.\n"
            "Run `uv run alphanus init` again to write the new project/permissions config, then retry.\n"
            "Removed keys:\n" + detail
        )
    if stripped:
        _warn(warnings, "Removed secret-like fields from config; use environment variables for secrets.")

    merged = deep_merge(DEFAULT_CONFIG, sanitized_input)
    input_agent_cfg = sanitized_input.get("agent", {}) if isinstance(sanitized_input.get("agent"), dict) else {}

    default_agent = DEFAULT_CONFIG["agent"]
    agent_cfg = merged.get("agent", {}) if isinstance(merged.get("agent"), dict) else {}
    agent_cfg["provider"] = _coerce_string(
        agent_cfg.get("provider"),
        str(default_agent["provider"]),
        path="agent.provider",
        warnings=warnings,
        allow_empty=False,
    )
    base_url_input = input_agent_cfg.get("base_url")
    inferred_base_url = ""
    if base_url_input in (None, ""):
        for endpoint_key in ("model_endpoint", "responses_endpoint", "models_endpoint"):
            if endpoint_key in input_agent_cfg:
                endpoint_text = str(input_agent_cfg.get(endpoint_key) or "").strip()
                if endpoint_text:
                    parsed_endpoint = urlparse(endpoint_text)
                    if (
                        parsed_endpoint.scheme.lower() in {"http", "https"}
                        and parsed_endpoint.hostname
                        and not parsed_endpoint.username
                        and not parsed_endpoint.password
                    ):
                        inferred_base_url = parsed_endpoint._replace(path="", params="", query="", fragment="").geturl().rstrip("/")
                if inferred_base_url:
                    break
    base_url_candidate = base_url_input if base_url_input not in (None, "") else (inferred_base_url or str(default_agent["base_url"]))
    agent_cfg["base_url"] = _normalize_base_url(
        base_url_candidate,
        default=str(default_agent["base_url"]),
        path="agent.base_url",
        warnings=warnings,
    )
    model_endpoint_default = _endpoint_from_base_url(str(agent_cfg["base_url"]), OPENAI_CHAT_COMPLETIONS_PATH)
    responses_endpoint_default = _endpoint_from_base_url(str(agent_cfg["base_url"]), OPENAI_RESPONSES_PATH)
    models_endpoint_default = _endpoint_from_base_url(str(agent_cfg["base_url"]), OPENAI_MODELS_PATH)

    model_input = input_agent_cfg.get("model_endpoint") if "model_endpoint" in input_agent_cfg else model_endpoint_default
    responses_input = input_agent_cfg.get("responses_endpoint") if "responses_endpoint" in input_agent_cfg else responses_endpoint_default
    models_input = input_agent_cfg.get("models_endpoint") if "models_endpoint" in input_agent_cfg else models_endpoint_default

    agent_cfg["model_endpoint"] = _normalize_endpoint(
        model_input,
        default=model_endpoint_default,
        path="agent.model_endpoint",
        warnings=warnings,
    )
    agent_cfg["responses_endpoint"] = _normalize_endpoint(
        responses_input,
        default=responses_endpoint_default,
        path="agent.responses_endpoint",
        warnings=warnings,
    )
    agent_cfg["models_endpoint"] = _normalize_endpoint(
        models_input,
        default=models_endpoint_default,
        path="agent.models_endpoint",
        warnings=warnings,
    )
    agent_cfg["endpoint_mode"] = _normalize_choice(
        agent_cfg.get("endpoint_mode"),
        str(default_agent.get("endpoint_mode", ENDPOINT_MODE_AUTO)),
        ENDPOINT_MODES,
        path="agent.endpoint_mode",
        warnings=warnings,
    )
    agent_cfg["backend_profile"] = _normalize_choice(
        agent_cfg.get("backend_profile"),
        AUTO_BACKEND_PROFILE,
        VALID_BACKEND_PROFILES,
        path="agent.backend_profile",
        warnings=warnings,
    )
    api_key_value = _coerce_string(
        agent_cfg.get("api_key"),
        str(default_agent.get("api_key", "env:ALPHANUS_API_KEY")),
        path="agent.api_key",
        warnings=warnings,
    )
    if api_key_value and not api_key_value.lower().startswith("env:"):
        _warn(warnings, "agent.api_key: plaintext values are discouraged; use env:VARIABLE")
    api_key_env = _coerce_string(
        agent_cfg.get("api_key_env"),
        str(default_agent.get("api_key_env", "ALPHANUS_API_KEY")),
        path="agent.api_key_env",
        warnings=warnings,
        allow_empty=False,
    )
    if not _VALID_ENV_NAME_RE.match(api_key_env):
        _warn(warnings, f"agent.api_key_env: invalid env name {api_key_env!r}, using default")
        api_key_env = str(default_agent.get("api_key_env", "ALPHANUS_API_KEY"))
    agent_cfg["api_key_env"] = api_key_env
    if api_key_value.lower().startswith("env:"):
        ref_name = api_key_value[4:].strip()
        if not _VALID_ENV_NAME_RE.match(ref_name):
            _warn(warnings, f"agent.api_key: invalid env reference {api_key_value!r}, using env:{api_key_env}")
            api_key_value = f"env:{api_key_env}"
        elif ref_name != api_key_env and not _VALID_ENV_NAME_RE.match(str(input_agent_cfg.get("api_key_env", "")).strip()):
            # Keep api_key aligned when api_key_env had to be corrected.
            api_key_value = f"env:{api_key_env}"
    agent_cfg["api_key"] = api_key_value
    auth_template = _coerce_string(
        agent_cfg.get("auth_header_template"),
        str(default_agent.get("auth_header_template", "Authorization: Bearer {api_key}")),
        path="agent.auth_header_template",
        warnings=warnings,
        allow_empty=False,
    )
    if "{api_key}" not in auth_template:
        _warn(warnings, "agent.auth_header_template: missing '{api_key}' placeholder, using default")
        auth_template = str(default_agent.get("auth_header_template", "Authorization: Bearer {api_key}"))
    agent_cfg["auth_header_template"] = auth_template
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
    agent_cfg = _normalize_model(agent_cfg, AgentConfig, "agent", warnings)
    raw_budgets = agent_cfg.get("tool_budgets")
    if isinstance(raw_budgets, dict):
        clean_budgets: dict[str, int] = {}
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

    simple_sections = {
        "project": (ProjectConfig, (("root_strategy", PROJECT_ROOT_STRATEGIES, False),)),
        "memory": (MemoryConfig, ()),
        "context": (ContextConfig, ()),
        "skills": (SkillsConfig, ()),
        "agents": (AgentsConfig, ()),
        "runtime": (RuntimeConfig, ()),
        "permissions": (PermissionsConfig, (("mode", PERMISSION_MODES, False), ("approvals", APPROVAL_MODES, False))),
        "sandbox": (SandboxConfig, (("backend", SANDBOX_BACKENDS, False),)),
        "logging": (
            LoggingConfig,
            (("level", {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}, True), ("format", {"plain", "json"}, False)),
        ),
    }
    for name, (model_type, choices) in simple_sections.items():
        values = merged.get(name, {}) if isinstance(merged.get(name), dict) else {}
        normalized = _normalize_model(values, model_type, name, warnings)
        for key, allowed, upper in choices:
            normalized[key] = _normalize_choice(
                values.get(key), str(DEFAULT_CONFIG[name][key]), allowed, path=f"{name}.{key}", warnings=warnings, upper=upper
            )
        merged[name] = normalized

    context_cfg = merged["context"]
    if context_cfg["safety_margin"] >= context_cfg["context_limit"]:
        adjusted = max(0, context_cfg["context_limit"] // 4)
        _warn(warnings, f"context.safety_margin: reduced to {adjusted} because it exceeded context_limit")
        context_cfg["safety_margin"] = adjusted

    tools_cfg = merged.get("tools", {}) if isinstance(merged.get("tools"), dict) else {}
    merged["tools"] = tools_cfg

    raw_search_cfg = merged.get("search", {}) if isinstance(merged.get("search"), dict) else {}
    search_cfg: dict[str, Any] = {}
    provider = _normalize_choice(
        raw_search_cfg.get("provider"),
        str(DEFAULT_CONFIG["search"]["provider"]),
        SEARCH_PROVIDERS,
        path="search.provider",
        warnings=warnings,
    )
    search_cfg["provider"] = provider
    fallback_provider = _coerce_string(
        raw_search_cfg.get("fallback_provider"),
        str(DEFAULT_CONFIG["search"]["fallback_provider"]),
        path="search.fallback_provider",
        warnings=warnings,
    ).lower()
    if fallback_provider not in {"", *SEARCH_FALLBACK_PROVIDERS}:
        _warn(warnings, f"search.fallback_provider: unsupported {fallback_provider!r}, using default")
        fallback_provider = str(DEFAULT_CONFIG["search"]["fallback_provider"])
    search_cfg["fallback_provider"] = "" if fallback_provider == SEARCH_FALLBACK_NONE else fallback_provider
    raw_provider_chain = raw_search_cfg.get("provider_chain", DEFAULT_CONFIG["search"]["provider_chain"])
    provider_chain: list[str] = []
    if isinstance(raw_provider_chain, list):
        provider_chain = [str(item).strip().lower() for item in raw_provider_chain if str(item).strip()]
    elif isinstance(raw_provider_chain, str) and raw_provider_chain.strip():
        provider_chain = [item.strip().lower() for item in raw_provider_chain.split(",") if item.strip()]
    cleaned_provider_chain: list[str] = []
    for item in provider_chain:
        if item == SEARCH_FALLBACK_NONE:
            continue
        if item not in SEARCH_PROVIDERS:
            _warn(warnings, f"search.provider_chain: unsupported {item!r}, skipping")
            continue
        if item not in cleaned_provider_chain:
            cleaned_provider_chain.append(item)
    search_cfg["provider_chain"] = cleaned_provider_chain
    raw_searxng_url = raw_search_cfg.get("searxng_base_url")
    base_url = (
        str(DEFAULT_CONFIG["search"]["searxng_base_url"])
        if raw_searxng_url is None
        else _coerce_string(raw_searxng_url, "", path="search.searxng_base_url", warnings=warnings)
    )
    if base_url:
        parsed = urlparse(base_url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            _warn(warnings, f"search.searxng_base_url: invalid URL {base_url!r}, using empty value")
            base_url = ""
        else:
            base_url = parsed._replace(params="", query="", fragment="").geturl().rstrip("/")
    search_cfg["searxng_base_url"] = base_url
    search_cfg["tavily_api_key_env"] = _normalize_env_name(
        raw_search_cfg.get("tavily_api_key_env"),
        default=str(DEFAULT_CONFIG["search"]["tavily_api_key_env"]),
        path="search.tavily_api_key_env",
        warnings=warnings,
    )
    search_cfg.update(
        _normalize_model(
            raw_search_cfg,
            SearchConfig,
            "search",
            warnings,
            exclude=("provider", "fallback_provider", "provider_chain", "searxng_base_url", "tavily_api_key_env"),
        )
    )
    merged["search"] = search_cfg

    retrieval_cfg = merged.get("retrieval", {}) if isinstance(merged.get("retrieval"), dict) else {}
    retrieval_default = DEFAULT_CONFIG["retrieval"]
    retrieval_cfg = _normalize_model(retrieval_cfg, RetrievalConfig, "retrieval", warnings, exclude=("store_path", "embeddings"))
    raw_store_path = retrieval_cfg.get("store_path")
    retrieval_cfg["store_path"] = (
        ""
        if raw_store_path is None
        else _coerce_string(
            raw_store_path,
            str(retrieval_default["store_path"]),
            path="retrieval.store_path",
            warnings=warnings,
        )
    )
    embeddings_cfg = retrieval_cfg.get("embeddings", {}) if isinstance(retrieval_cfg.get("embeddings"), dict) else {}
    embeddings_default = retrieval_default["embeddings"]
    embeddings_cfg = _normalize_model(embeddings_cfg, EmbeddingsConfig, "retrieval.embeddings", warnings)
    if embeddings_cfg["base_url"]:
        embeddings_cfg["base_url"] = _normalize_base_url(
            embeddings_cfg["base_url"],
            default="",
            path="retrieval.embeddings.base_url",
            warnings=warnings,
        )
    embeddings_cfg["api_key_env"] = _normalize_env_name(
        embeddings_cfg.get("api_key_env"),
        default=str(embeddings_default["api_key_env"]),
        path="retrieval.embeddings.api_key_env",
        warnings=warnings,
    )
    retrieval_cfg["embeddings"] = embeddings_cfg
    merged["retrieval"] = retrieval_cfg

    tui_cfg = merged.get("tui", {}) if isinstance(merged.get("tui"), dict) else {}
    raw_theme = _coerce_string(
        tui_cfg.get("theme"),
        str(DEFAULT_CONFIG["tui"]["theme"]),
        path="tui.theme",
        warnings=warnings,
        allow_empty=False,
    )
    try:
        from core.themes import available_theme_ids

        theme_ids = available_theme_ids()
    except Exception:
        theme_ids = None
    resolved_theme, changed = normalize_theme_id(raw_theme, default=str(DEFAULT_THEME_ID), available=theme_ids)
    if changed:
        if raw_theme.strip():
            _warn(warnings, f"tui.theme: unsupported {raw_theme!r}, using {resolved_theme!r}")
        else:
            _warn(warnings, f"tui.theme: empty value, using {resolved_theme!r}")
    tui_cfg["theme"] = resolved_theme
    tui_cfg = _normalize_model(tui_cfg, UiConfig, "tui", warnings, exclude=("theme", "timing", "tree_compaction"))
    tui_cfg["theme"] = resolved_theme
    timing_cfg = tui_cfg.get("timing", {})
    if not isinstance(timing_cfg, dict):
        _warn(warnings, "tui.timing: expected object, using defaults")
        timing_cfg = {}
    timing_cfg = _normalize_model(timing_cfg, UiTimingConfig, "tui.timing", warnings)
    tui_cfg["timing"] = timing_cfg
    tree_cfg = tui_cfg.get("tree_compaction", {})
    if not isinstance(tree_cfg, dict):
        _warn(warnings, "tui.tree_compaction: expected object, using defaults")
        tree_cfg = {}
    tree_cfg = _normalize_model(tree_cfg, TreeCompactionConfig, "tui.tree_compaction", warnings)
    tui_cfg["tree_compaction"] = tree_cfg
    merged["tui"] = tui_cfg

    return validated_config(merged), warnings


def validate_endpoint_policy(config: dict[str, Any]) -> None:
    agent_cfg = config.get("agent", {}) if isinstance(config.get("agent"), dict) else {}
    model_endpoint = str(agent_cfg.get("model_endpoint", "")).strip()
    models_endpoint = str(agent_cfg.get("models_endpoint", "")).strip()
    parsed_model = urlparse(model_endpoint)
    inferred_base = ""
    if parsed_model.scheme and parsed_model.hostname:
        inferred_base = parsed_model._replace(path="", params="", query="", fragment="").geturl().rstrip("/")
    base_url = str(agent_cfg.get("base_url", "")).strip() or inferred_base
    responses_endpoint = str(agent_cfg.get("responses_endpoint", "")).strip() or _endpoint_from_base_url(
        base_url or str(DEFAULT_CONFIG["agent"]["base_url"]),
        OPENAI_RESPONSES_PATH,
    )
    allow_cross = bool(agent_cfg.get("allow_cross_host_endpoints", False))

    parsed_base = urlparse(base_url)
    parsed_responses = urlparse(responses_endpoint)
    parsed_models = urlparse(models_endpoint)
    if parsed_base.scheme.lower() not in {"http", "https"} or not parsed_base.hostname:
        raise ValueError("agent.base_url must be a valid http(s) URL")
    if parsed_model.scheme.lower() not in {"http", "https"} or not parsed_model.hostname:
        raise ValueError("agent.model_endpoint must be a valid http(s) URL")
    if parsed_responses.scheme.lower() not in {"http", "https"} or not parsed_responses.hostname:
        raise ValueError("agent.responses_endpoint must be a valid http(s) URL")
    if parsed_models.scheme.lower() not in {"http", "https"} or not parsed_models.hostname:
        raise ValueError("agent.models_endpoint must be a valid http(s) URL")

    hosts = {
        (parsed_base.hostname or "").lower(),
        (parsed_model.hostname or "").lower(),
        (parsed_responses.hostname or "").lower(),
        (parsed_models.hostname or "").lower(),
    }
    if len(hosts) > 1 and not allow_cross:
        raise ValueError("agent.base_url, agent.model_endpoint, agent.responses_endpoint, and agent.models_endpoint must share host")


def load_global_config(path: Path, *, warnings: list[str] | None = None) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Global config not found at {path}")

    size = path.stat().st_size
    if size > MAX_CONFIG_BYTES:
        raise ValueError(f"Global config is too large ({size} bytes); limit is {MAX_CONFIG_BYTES} bytes")

    if path.suffix.lower() != ".toml":
        raise ValueError(
            f"Unsupported legacy configuration at {path}. Alphanus v1 requires config/config.toml; run `alphanus init --reset`."
        )
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid global config TOML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Global config must be a TOML document")

    version = raw.get("config_version")
    if version != CONFIG_VERSION:
        raise ValueError(f"Unsupported config_version {version!r}; expected {CONFIG_VERSION}")
    unknown = sorted(set(raw) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {', '.join(unknown)}")

    sanitized_raw, stripped = strip_secret_fields(raw)
    if stripped:
        raise ValueError(
            "Secret-like values are forbidden in config.toml. Remove them and provide credentials through environment variables."
        )

    normalized, local_warnings = normalize_config(raw)
    if warnings is not None:
        warnings.extend(local_warnings)
    return normalized


def _toml_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _toml_ready(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_toml_ready(item) for item in value]
    return value


def config_to_toml(config: dict[str, Any]) -> str:
    """Serialize the normalized, JSON-shaped configuration used by Alphanus."""
    return tomli_w.dumps(_toml_ready(config))


def save_global_config(path: Path, config: Mapping[str, Any]) -> None:
    from core.secure_io import atomic_write_text

    cleaned = config_for_editor_view(config)
    cleaned["config_version"] = CONFIG_VERSION
    sanitized, stripped = strip_secret_fields(cleaned)
    if stripped:
        raise ValueError("Refusing to persist secret-like configuration fields")
    atomic_write_text(path, config_to_toml(sanitized), mode=0o600)
