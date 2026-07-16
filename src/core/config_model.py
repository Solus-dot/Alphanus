from __future__ import annotations

import sys
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from core.configuration import DEFAULT_CONFIG


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    return value if isinstance(value, dict) else {}


class RuntimeModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)


class UiTimingConfig(RuntimeModel):
    stream_drain_interval_s: float = float(DEFAULT_CONFIG["tui"]["timing"]["stream_drain_interval_s"])
    scroll_interval_s: float = float(DEFAULT_CONFIG["tui"]["timing"]["scroll_interval_s"])
    action_approval_timeout_s: float = float(DEFAULT_CONFIG["tui"]["timing"]["action_approval_timeout_s"])


class UiRuntimeConfig(RuntimeModel):
    theme: str = str(DEFAULT_CONFIG["tui"]["theme"])
    chat_log_max_lines: int | None = int(DEFAULT_CONFIG["tui"]["chat_log_max_lines"])
    tree_compaction_enabled: bool = bool(DEFAULT_CONFIG["tui"]["tree_compaction"]["enabled"])
    inactive_assistant_char_limit: int = int(DEFAULT_CONFIG["tui"]["tree_compaction"]["inactive_assistant_char_limit"])
    inactive_tool_argument_char_limit: int = int(DEFAULT_CONFIG["tui"]["tree_compaction"]["inactive_tool_argument_char_limit"])
    inactive_tool_content_char_limit: int = int(DEFAULT_CONFIG["tui"]["tree_compaction"]["inactive_tool_content_char_limit"])
    timing: UiTimingConfig = Field(default_factory=UiTimingConfig)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        section = _section(config, "tui")
        tree = _section(section, "tree_compaction")
        return cls.model_validate(
            {
                **section,
                "tree_compaction_enabled": tree.get("enabled", cls.model_fields["tree_compaction_enabled"].default),
                "inactive_assistant_char_limit": tree.get(
                    "inactive_assistant_char_limit", cls.model_fields["inactive_assistant_char_limit"].default
                ),
                "inactive_tool_argument_char_limit": tree.get(
                    "inactive_tool_argument_char_limit", cls.model_fields["inactive_tool_argument_char_limit"].default
                ),
                "inactive_tool_content_char_limit": tree.get(
                    "inactive_tool_content_char_limit", cls.model_fields["inactive_tool_content_char_limit"].default
                ),
            }
        )


class ProviderConfig(RuntimeModel):
    provider_name: str = Field(default=str(DEFAULT_CONFIG["agent"]["provider"]), alias="provider")
    base_url: str = str(DEFAULT_CONFIG["agent"]["base_url"])
    model_endpoint: str = str(DEFAULT_CONFIG["agent"]["model_endpoint"])
    responses_endpoint: str = str(DEFAULT_CONFIG["agent"]["responses_endpoint"])
    models_endpoint: str = str(DEFAULT_CONFIG["agent"]["models_endpoint"])
    endpoint_mode: str = str(DEFAULT_CONFIG["agent"]["endpoint_mode"])
    backend_profile: str = str(DEFAULT_CONFIG["agent"]["backend_profile"])
    tls_verify: bool = bool(DEFAULT_CONFIG["agent"]["tls_verify"])
    ca_bundle_path: str = str(DEFAULT_CONFIG["agent"]["ca_bundle_path"])
    allow_cross_host: bool = Field(default=bool(DEFAULT_CONFIG["agent"]["allow_cross_host_endpoints"]), alias="allow_cross_host_endpoints")
    request_timeout_s: float = float(DEFAULT_CONFIG["agent"]["request_timeout_s"])
    readiness_timeout_s: float = float(DEFAULT_CONFIG["agent"]["readiness_timeout_s"])
    readiness_poll_s: float = float(DEFAULT_CONFIG["agent"]["readiness_poll_s"])
    connect_timeout_s: float = float(DEFAULT_CONFIG["agent"]["connect_timeout_s"])
    per_turn_retries: int = int(DEFAULT_CONFIG["agent"]["per_turn_retries"])
    retry_backoff_s: float = float(DEFAULT_CONFIG["agent"]["retry_backoff_s"])
    default_max_tokens: int | None = Field(default=None, alias="max_tokens")
    api_key: str = str(DEFAULT_CONFIG["agent"]["api_key"])
    api_key_env: str = str(DEFAULT_CONFIG["agent"]["api_key_env"])
    auth_header_template: str = str(DEFAULT_CONFIG["agent"]["auth_header_template"])
    auth_header: str | None = Field(default=None, exclude=True)

    @classmethod
    def from_config(cls, config: dict[str, Any], *, auth_header: str | None = None) -> Self:
        value = (auth_header or "").strip() or None
        return cls.model_validate({**_section(config, "agent"), "auth_header": value})


class SkillsRuntimeConfig(RuntimeModel):
    python_executable: str = sys.executable
    paths: list[str] = Field(default_factory=list)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        section = _section(config, "skills")
        return cls.model_validate({**section, "python_executable": str(section.get("python_executable") or sys.executable)})
