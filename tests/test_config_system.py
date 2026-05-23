from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.config_model import TypedConfigV2
from core.configuration import (
    DEFAULT_CONFIG,
    load_dotenv,
    load_global_config,
    normalize_config,
    validate_endpoint_policy,
)
from core.retrieval import configured_store_path
from core.runtime_config import ProviderConfig, SkillsRuntimeConfig, UiRuntimeConfig


def test_normalize_config_strips_secret_like_fields() -> None:
    raw = {
        "agent": {"auth_header": "Authorization: Bearer secret"},
        "search": {"provider": "searxng", "tavily_api_key": "tvly-secret"},
        "nested": {"api_key": "abc", "keep": True},
    }

    normalized, warnings = normalize_config(raw)

    assert "auth_header" not in normalized["agent"]
    assert "tavily_api_key" not in normalized["search"]
    assert "api_key" not in normalized["nested"]
    assert normalized["nested"]["keep"] is True
    assert any("secret-like fields" in item for item in warnings)


def test_default_retrieval_store_path_uses_app_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    monkeypatch.setenv("ALPHANUS_APP_ROOT", str(state_root))

    normalized, warnings = normalize_config({})

    assert warnings == []
    assert normalized["retrieval"]["store_path"] == ""
    assert configured_store_path(normalized) == state_root / "retrieval" / "index.sqlite"


def test_normalize_config_clamps_and_falls_back_invalid_values() -> None:
    raw = {
        "agent": {
            "model_endpoint": "ftp://bad-endpoint",
            "request_timeout_s": "bad",
            "readiness_poll_s": 0,
            "max_action_depth": -8,
            "max_tokens": "0",
            "backend_profile": "unsupported-backend",
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
    assert normalized["agent"]["backend_profile"] == "auto"
    assert normalized["context"]["context_limit"] == 512
    assert normalized["context"]["safety_margin"] < normalized["context"]["context_limit"]
    assert normalized["skills"]["strict_capability_policy"] == DEFAULT_CONFIG["skills"]["strict_capability_policy"]
    assert normalized["search"]["provider"] == DEFAULT_CONFIG["search"]["provider"]
    assert normalized["tui"]["chat_log_max_lines"] == 100


def test_normalize_config_accepts_search_architecture_knobs() -> None:
    normalized, warnings = normalize_config(
        {
            "search": {
                "provider_chain": ["searxng", "bad", "tavily", "searxng"],
                "cache_first": "false",
                "min_usable_results": 2,
                "fetch_min_chars": 80,
            }
        }
    )

    assert normalized["search"]["provider_chain"] == ["searxng", "tavily"]
    assert normalized["search"]["cache_first"] is False
    assert normalized["search"]["min_usable_results"] == 2
    assert normalized["search"]["fetch_min_chars"] == 80
    assert any("search.provider_chain: unsupported 'bad'" in warning for warning in warnings)


def test_normalize_config_preserves_api_key_env_reference() -> None:
    normalized, warnings = normalize_config(
        {
            "agent": {
                "api_key": "env:CUSTOM_KEY",
                "api_key_env": "CUSTOM_KEY",
            },
        }
    )

    assert normalized["agent"]["api_key"] == "env:CUSTOM_KEY"
    assert normalized["agent"]["api_key_env"] == "CUSTOM_KEY"
    assert not any("secret-like fields" in warning for warning in warnings)


def test_normalize_config_infers_base_url_from_existing_remote_endpoints() -> None:
    normalized, _warnings = normalize_config(
        {
            "agent": {
                "model_endpoint": "https://remote.example/v1/chat/completions",
                "models_endpoint": "https://remote.example/v1/models",
            },
        }
    )

    assert normalized["agent"]["base_url"] == "https://remote.example"
    assert normalized["agent"]["responses_endpoint"] == "https://remote.example/v1/responses"
    validate_endpoint_policy(normalized)


def test_normalize_config_aligns_api_key_reference_when_api_key_env_is_invalid() -> None:
    normalized, warnings = normalize_config(
        {
            "agent": {
                "api_key": "env:BAD-NAME",
                "api_key_env": "BAD-NAME",
            },
        }
    )

    assert normalized["agent"]["api_key_env"] == "ALPHANUS_API_KEY"
    assert normalized["agent"]["api_key"] == "env:ALPHANUS_API_KEY"
    assert any("agent.api_key_env: invalid env name" in item for item in warnings)


def test_normalize_config_theme_alias_and_invalid_values() -> None:
    aliased, alias_warnings = normalize_config({"tui": {"theme": "catppuccin"}})
    invalid, invalid_warnings = normalize_config({"tui": {"theme": "unknown-theme"}})

    assert aliased["tui"]["theme"] == "catppuccin-mocha"
    assert invalid["tui"]["theme"] == DEFAULT_CONFIG["tui"]["theme"]
    assert any("tui.theme" in warning for warning in invalid_warnings)
    assert not any("unsupported 'catppuccin'" in warning for warning in alias_warnings)


def test_normalize_config_runtime_profile_aliases() -> None:
    minimal, minimal_warnings = normalize_config({"runtime": {"profile": "safe"}})
    standard, standard_warnings = normalize_config({"runtime": {"profile": "workspace"}})
    invalid, invalid_warnings = normalize_config({"runtime": {"profile": "unknown"}})

    assert minimal["runtime"]["profile"] == "minimal"
    assert standard["runtime"]["profile"] == "standard"
    assert invalid["runtime"]["profile"] == DEFAULT_CONFIG["runtime"]["profile"]
    assert not minimal_warnings
    assert not standard_warnings
    assert any("runtime.profile" in warning for warning in invalid_warnings)


def test_normalize_config_permission_profile_aliases() -> None:
    safe, safe_warnings = normalize_config({"capabilities": {"permission_profile": "minimal"}})
    workspace, workspace_warnings = normalize_config({"capabilities": {"permission_profile": "standard"}})
    invalid, invalid_warnings = normalize_config({"capabilities": {"permission_profile": "unknown"}})

    assert safe["capabilities"]["permission_profile"] == "safe"
    assert workspace["capabilities"]["permission_profile"] == "workspace"
    assert invalid["capabilities"]["permission_profile"] == DEFAULT_CONFIG["capabilities"]["permission_profile"]
    assert not safe_warnings
    assert not workspace_warnings
    assert any("capabilities.permission_profile" in warning for warning in invalid_warnings)


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
        '{"search":{"provider":"searxng","tavily_api_key":"secret"}}',
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
                "export ALPHANUS_EMBEDDINGS_API_KEY='embed-test'",
                "INVALID-NAME=bad",
                "EMPTY_LINE_ONLY",
                'QUOTED_VALUE="hello world"',
                "NULLY=bad\x00value",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EXISTING", "from-env")
    monkeypatch.delenv("ALPHANUS_EMBEDDINGS_API_KEY", raising=False)
    monkeypatch.delenv("INVALID-NAME", raising=False)
    monkeypatch.delenv("QUOTED_VALUE", raising=False)
    monkeypatch.delenv("NULLY", raising=False)

    load_dotenv(env_file)

    assert "from-env" == os.environ["EXISTING"]
    assert "embed-test" == os.environ["ALPHANUS_EMBEDDINGS_API_KEY"]
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
        "agent": {
            "provider": "openai-compatible",
            "connect_timeout_s": "2.5",
            "per_turn_retries": "2",
            "retry_backoff_s": "0.75",
        },
        "skills": {"python_executable": "/usr/bin/python3", "paths": ["~/agent-skills"]},
        "tui": {"timing": {"stream_drain_interval_s": "0.02", "shell_confirm_timeout_s": "90"}},
    }

    normalized, _warnings = normalize_config(raw)

    assert normalized["agent"]["provider"] == "openai-compatible"
    assert normalized["agent"]["connect_timeout_s"] == 2.5
    assert normalized["agent"]["per_turn_retries"] == 2
    assert normalized["agent"]["retry_backoff_s"] == 0.75
    assert normalized["skills"]["python_executable"] == "/usr/bin/python3"
    assert normalized["skills"]["paths"] == ["~/agent-skills"]
    assert normalized["tui"]["timing"]["stream_drain_interval_s"] == 0.02
    assert normalized["tui"]["timing"]["shell_confirm_timeout_s"] == 90.0


def test_normalize_config_clamps_memory_fields() -> None:
    raw = {
        "memory": {
            "min_score_default": 2.0,
            "recall_min_score_default": "-1",
            "replace_min_score_default": "bad",
            "backup_revisions": -4,
        },
    }

    normalized, warnings = normalize_config(raw)

    assert normalized["memory"]["min_score_default"] == 1.0
    assert normalized["memory"]["recall_min_score_default"] == 0.0
    assert normalized["memory"]["replace_min_score_default"] == DEFAULT_CONFIG["memory"]["replace_min_score_default"]
    assert normalized["memory"]["backup_revisions"] == 0


def test_normalize_config_does_not_accept_old_section_aliases() -> None:
    normalized, warnings = normalize_config(
        {
            "search": {"base_url": "http://127.0.0.1:8888"},
            "skills": {"load": ["search-ops"]},
            "memory": {"path": "./memory.pkl", "model_name": "unused"},
        }
    )

    assert normalized["search"]["searxng_base_url"] == ""
    assert "base_url" not in normalized["search"]
    assert "load" not in normalized["skills"]
    assert "path" not in normalized["memory"]
    assert "model_name" not in normalized["memory"]
    assert warnings == []


def test_typed_runtime_configs_parse_normalized_config() -> None:
    normalized, _warnings = normalize_config(
        {
            "agent": {"connect_timeout_s": 3, "per_turn_retries": 2},
            "skills": {"python_executable": "/usr/bin/python3", "paths": ["~/agent-skills"]},
            "tui": {"theme": "gruvbox-dark-soft", "chat_log_max_lines": 1234, "timing": {"model_refresh_interval_s": 9}},
        }
    )

    provider = ProviderConfig.from_config(normalized, auth_header="Authorization: Bearer demo")
    skills = SkillsRuntimeConfig.from_config(normalized)
    ui = UiRuntimeConfig.from_config(normalized)

    assert provider.connect_timeout_s == 3.0
    assert provider.per_turn_retries == 2
    assert provider.backend_profile == "auto"
    assert provider.auth_header == "Authorization: Bearer demo"
    assert skills.python_executable == "/usr/bin/python3"
    assert skills.paths == ["~/agent-skills"]
    assert ui.theme == "gruvbox-dark-soft"
    assert ui.chat_log_max_lines == 1234
    assert ui.timing.model_refresh_interval_s == 9.0


def test_typed_config_v2_groups_runtime_sections() -> None:
    normalized, _warnings = normalize_config(
        {
            "agent": {"connect_timeout_s": 3, "per_turn_retries": 2},
            "workspace": {"path": "~/code"},
            "memory": {"backup_revisions": 4},
            "capabilities": {"permission_profile": "workspace"},
            "runtime": {"profile": "minimal", "ask_user_tool": False},
            "search": {"provider": "searxng", "fallback_provider": "tavily", "searxng_base_url": "http://127.0.0.1:8888"},
            "skills": {"python_executable": "/usr/bin/python3", "paths": ["~/agent-skills"]},
            "tui": {"theme": "gruvbox-dark-soft"},
        }
    )

    typed = TypedConfigV2.from_normalized_config(normalized, auth_header="Authorization: Bearer demo")

    assert typed.provider.connect_timeout_s == 3.0
    assert typed.provider.auth_header == "Authorization: Bearer demo"
    assert typed.workspace.path == "~/code"
    assert typed.memory.backup_revisions == 4
    assert typed.runtime_policy.runtime_profile == "minimal"
    assert typed.runtime_policy.permission_profile == "workspace"
    assert typed.runtime_policy.ask_user_tool is False
    assert typed.search.provider == "searxng"
    assert typed.search.fallback_provider == "tavily"
    assert typed.search.searxng_base_url == "http://127.0.0.1:8888"
    assert typed.retrieval.enabled is True
    assert typed.skills.python_executable == "/usr/bin/python3"
    assert typed.skills.paths == ["~/agent-skills"]
    assert typed.ui.theme == "gruvbox-dark-soft"


def test_normalize_config_accepts_loadable_custom_theme(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    theme_dir = tmp_path / "themes"
    theme_dir.mkdir()
    (theme_dir / "custom-oxide.json").write_text(
        """
{
  "id": "custom-oxide",
  "title": "Custom Oxide",
  "description": "Custom config theme",
  "theme": {
    "primary": "#111111",
    "secondary": "#222222",
    "accent": "#333333",
    "foreground": "#eeeeee",
    "background": "#000000",
    "surface": "#111111",
    "panel": "#050505",
    "success": "#00ff00",
    "warning": "#ffff00",
    "error": "#ff0000",
    "dark": true
  },
  "colors": {
    "accent": "#333333",
    "text": "#eeeeee",
    "muted": "#999999",
    "subtle": "#777777",
    "success": "#00ff00",
    "warning": "#ffff00",
    "error": "#ff0000",
    "user_bar": "#00ff00",
    "assistant_bar": "#333333",
    "chip_bg": "#111111",
    "chip_text": "#eeeeee",
    "panel_bg": "#050505",
    "panel_border": "#555555"
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHANUS_THEME_PATHS", str(theme_dir))
    from tui import themes

    themes.reload_theme_specs()
    try:
        normalized, warnings = normalize_config({"tui": {"theme": "custom-oxide"}})
        ui = UiRuntimeConfig.from_config(normalized)
    finally:
        themes.reload_theme_specs()

    assert normalized["tui"]["theme"] == "custom-oxide"
    assert ui.theme == "custom-oxide"
    assert warnings == []
