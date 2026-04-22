from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.configuration import (
    DEFAULT_CONFIG,
    load_dotenv,
    load_global_config,
    normalize_config,
    validate_endpoint_policy,
)
from core.runtime_config import ProviderConfig, SkillsRuntimeConfig, UiRuntimeConfig


def test_normalize_config_strips_secret_like_fields() -> None:
    raw = {
        "schema_version": "1.0.0",
        "agent": {"auth_header": "Authorization: Bearer secret"},
        "search": {"provider": "tavily", "tavily_api_key": "tvly-secret"},
        "nested": {"api_key": "abc", "keep": True},
    }

    normalized, warnings = normalize_config(raw)

    assert "auth_header" not in normalized["agent"]
    assert "tavily_api_key" not in normalized["search"]
    assert "api_key" not in normalized["nested"]
    assert normalized["nested"]["keep"] is True
    assert any("secret-like fields" in item for item in warnings)


def test_normalize_config_clamps_and_falls_back_invalid_values() -> None:
    raw = {
        "schema_version": "1.0.0",
        "agent": {
            "model_endpoint": "ftp://bad-endpoint",
            "request_timeout_s": "bad",
            "readiness_poll_s": 0,
            "max_action_depth": -8,
            "max_tokens": "0",
        },
        "context": {"context_limit": 200, "safety_margin": 5000},
        "skills": {"strict_capability_policy": "bad"},
        "search": {"provider": "bing"},
        "tui": {"chat_log_max_lines": -1},
    }

    normalized, _warnings = normalize_config(raw)

    assert normalized["agent"]["model_endpoint"] == DEFAULT_CONFIG["agent"]["model_endpoint"]
    assert normalized["agent"]["request_timeout_s"] == float(DEFAULT_CONFIG["agent"]["request_timeout_s"])
    assert normalized["agent"]["readiness_poll_s"] == 0.05
    assert normalized["agent"]["max_action_depth"] == 1
    assert normalized["agent"]["max_tokens"] is None
    assert normalized["context"]["context_limit"] == 512
    assert normalized["context"]["safety_margin"] < normalized["context"]["context_limit"]
    assert normalized["skills"]["strict_capability_policy"] == DEFAULT_CONFIG["skills"]["strict_capability_policy"]
    assert normalized["search"]["provider"] == DEFAULT_CONFIG["search"]["provider"]
    assert normalized["tui"]["chat_log_max_lines"] == 100


def test_normalize_config_theme_alias_and_invalid_values() -> None:
    aliased, alias_warnings = normalize_config({"schema_version": "1.0.0", "tui": {"theme": "catppuccin"}})
    invalid, invalid_warnings = normalize_config({"schema_version": "1.0.0", "tui": {"theme": "unknown-theme"}})

    assert aliased["tui"]["theme"] == "catppuccin-mocha"
    assert invalid["tui"]["theme"] == DEFAULT_CONFIG["tui"]["theme"]
    assert any("tui.theme" in warning for warning in invalid_warnings)
    assert not any("unsupported 'catppuccin'" in warning for warning in alias_warnings)

def test_load_global_config_reports_and_rejects_bad_json(tmp_path: Path) -> None:
    cfg = tmp_path / "config" / "global_config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("{broken-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid global config JSON"):
        load_global_config(cfg)


def test_load_global_config_scrubs_secret_fields_on_disk(tmp_path: Path) -> None:
    cfg = tmp_path / "config" / "global_config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        '{"schema_version":"1.0.0","search":{"provider":"tavily","tavily_api_key":"secret"}}',
        encoding="utf-8",
    )

    warnings: list[str] = []
    loaded = load_global_config(cfg, warnings=warnings)
    disk = cfg.read_text(encoding="utf-8")

    assert "tavily_api_key" not in loaded["search"]
    assert "tavily_api_key" not in disk
    assert any("on disk" in warning for warning in warnings)


def test_load_global_config_fails_for_missing_file(tmp_path: Path) -> None:
    cfg = tmp_path / "config" / "global_config.json"

    with pytest.raises(FileNotFoundError, match="Global config not found"):
        load_global_config(cfg)


def test_load_dotenv_supports_export_and_ignores_invalid_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "EXISTING=from-file",
                "export TAVILY_API_KEY='tvly-test'",
                "INVALID-NAME=bad",
                "EMPTY_LINE_ONLY",
                'QUOTED_VALUE="hello world"',
                "NULLY=bad\x00value",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EXISTING", "from-env")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("INVALID-NAME", raising=False)
    monkeypatch.delenv("QUOTED_VALUE", raising=False)
    monkeypatch.delenv("NULLY", raising=False)

    load_dotenv(env_file)

    assert "from-env" == os.environ["EXISTING"]
    assert "tvly-test" == os.environ["TAVILY_API_KEY"]
    assert "INVALID-NAME" not in os.environ
    assert "hello world" == os.environ["QUOTED_VALUE"]
    assert "NULLY" not in os.environ


def test_validate_endpoint_policy_allows_same_host_different_ports() -> None:
    config = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://127.0.0.1:9000/v1/models",
            "allow_cross_host_endpoints": False,
        }
    }
    validate_endpoint_policy(config)


def test_validate_endpoint_policy_rejects_cross_host_when_disallowed() -> None:
    config = {
        "agent": {
            "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
            "models_endpoint": "http://example.com/v1/models",
            "allow_cross_host_endpoints": False,
        }
    }
    with pytest.raises(ValueError, match="must share host"):
        validate_endpoint_policy(config)


def test_normalize_config_preserves_new_runtime_boundary_fields() -> None:
    raw = {
        "schema_version": "1.0.0",
        "agent": {
            "provider": "openai-compatible",
            "connect_timeout_s": "2.5",
            "per_turn_retries": "2",
            "retry_backoff_s": "0.75",
        },
        "skills": {"python_executable": "/usr/bin/python3"},
        "tui": {"timing": {"stream_drain_interval_s": "0.02", "shell_confirm_timeout_s": "90"}},
    }

    normalized, _warnings = normalize_config(raw)

    assert normalized["agent"]["provider"] == "openai-compatible"
    assert normalized["agent"]["connect_timeout_s"] == 2.5
    assert normalized["agent"]["per_turn_retries"] == 2
    assert normalized["agent"]["retry_backoff_s"] == 0.75
    assert normalized["skills"]["python_executable"] == "/usr/bin/python3"
    assert normalized["tui"]["timing"]["stream_drain_interval_s"] == 0.02
    assert normalized["tui"]["timing"]["shell_confirm_timeout_s"] == 90.0


def test_normalize_config_clamps_memory_robustness_fields() -> None:
    raw = {
        "schema_version": "1.0.0",
        "memory": {
            "path": "./legacy/memory.pkl",
            "min_score_default": 2.0,
            "recall_min_score_default": "-1",
            "replace_min_score_default": "bad",
            "backup_revisions": -4,
            "allow_schema_migration": "false",
            "model_name": "legacy-model",
            "allow_model_download": True,
        },
    }

    normalized, warnings = normalize_config(raw)

    assert normalized["memory"]["min_score_default"] == 1.0
    assert normalized["memory"]["recall_min_score_default"] == 0.0
    assert normalized["memory"]["replace_min_score_default"] == DEFAULT_CONFIG["memory"]["replace_min_score_default"]
    assert normalized["memory"]["backup_revisions"] == 0
    assert "path" not in normalized["memory"]
    assert "allow_schema_migration" not in normalized["memory"]
    assert "model_name" not in normalized["memory"]
    assert "allow_model_download" not in normalized["memory"]


def test_typed_runtime_configs_parse_normalized_config() -> None:
    normalized, _warnings = normalize_config(
        {
            "schema_version": "1.0.0",
            "agent": {"connect_timeout_s": 3, "per_turn_retries": 2},
            "skills": {"python_executable": "/usr/bin/python3"},
            "tui": {"theme": "gruvbox-dark-soft", "chat_log_max_lines": 1234, "timing": {"model_refresh_interval_s": 9}},
        }
    )

    provider = ProviderConfig.from_config(normalized, auth_header="Authorization: Bearer demo")
    skills = SkillsRuntimeConfig.from_config(normalized)
    ui = UiRuntimeConfig.from_config(normalized)

    assert provider.connect_timeout_s == 3.0
    assert provider.per_turn_retries == 2
    assert provider.auth_header == "Authorization: Bearer demo"
    assert skills.python_executable == "/usr/bin/python3"
    assert ui.theme == "gruvbox-dark-soft"
    assert ui.chat_log_max_lines == 1234
    assert ui.timing.model_refresh_interval_s == 9.0
