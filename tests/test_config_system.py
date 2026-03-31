from __future__ import annotations

import os
import json
from pathlib import Path

import pytest

from core.configuration import (
    DEFAULT_CONFIG,
    load_dotenv,
    load_or_create_global_config,
    normalize_config,
    validate_endpoint_policy,
)


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

def test_load_or_create_global_config_reports_and_rejects_bad_json(tmp_path: Path) -> None:
    cfg = tmp_path / "config" / "global_config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("{broken-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid global config JSON"):
        load_or_create_global_config(cfg)


def test_load_or_create_global_config_scrubs_secret_fields_on_disk(tmp_path: Path) -> None:
    cfg = tmp_path / "config" / "global_config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        '{"schema_version":"1.0.0","search":{"provider":"tavily","tavily_api_key":"secret"}}',
        encoding="utf-8",
    )

    warnings: list[str] = []
    loaded = load_or_create_global_config(cfg, warnings=warnings)
    disk = cfg.read_text(encoding="utf-8")

    assert "tavily_api_key" not in loaded["search"]
    assert "tavily_api_key" not in disk
    assert any("on disk" in warning for warning in warnings)


def test_load_or_create_global_config_hides_internal_context_budget_fields_on_disk(tmp_path: Path) -> None:
    cfg = tmp_path / "config" / "global_config.json"

    loaded = load_or_create_global_config(cfg)
    written = json.loads(cfg.read_text(encoding="utf-8"))

    assert loaded["agent"]["context_budget_max_tokens"] == DEFAULT_CONFIG["agent"]["context_budget_max_tokens"]
    assert loaded["context"]["context_limit"] == DEFAULT_CONFIG["context"]["context_limit"]
    assert loaded["context"]["safety_margin"] == DEFAULT_CONFIG["context"]["safety_margin"]
    assert "context_budget_max_tokens" not in written["agent"]
    assert "context_limit" not in written["context"]
    assert "safety_margin" not in written["context"]


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
