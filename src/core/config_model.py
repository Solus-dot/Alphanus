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


class ProjectConfig(RuntimeModel):
    root_strategy: str = str(DEFAULT_CONFIG["project"]["root_strategy"])

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        return cls.model_validate(_section(config, "project"))


class MemoryConfig(RuntimeModel):
    min_score_default: float = float(DEFAULT_CONFIG["memory"]["min_score_default"])
    recall_min_score_default: float = float(DEFAULT_CONFIG["memory"]["recall_min_score_default"])
    replace_min_score_default: float = float(DEFAULT_CONFIG["memory"]["replace_min_score_default"])
    backup_revisions: int = int(DEFAULT_CONFIG["memory"]["backup_revisions"])
    auto_capture: bool = bool(DEFAULT_CONFIG["memory"]["auto_capture"])

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        return cls.model_validate(_section(config, "memory"))


class RuntimePolicyConfig(RuntimeModel):
    permission_mode: str = str(DEFAULT_CONFIG["permissions"]["mode"])
    approvals: str = str(DEFAULT_CONFIG["permissions"]["approvals"])
    network: bool = bool(DEFAULT_CONFIG["permissions"]["network"])
    sandbox_backend: str = str(DEFAULT_CONFIG["sandbox"]["backend"])
    sandbox_fail_closed: bool = bool(DEFAULT_CONFIG["sandbox"]["fail_closed"])
    ask_user_tool: bool = bool(DEFAULT_CONFIG["runtime"]["ask_user_tool"])

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        permissions = _section(config, "permissions")
        sandbox = _section(config, "sandbox")
        runtime = _section(config, "runtime")
        return cls.model_validate(
            {
                "permission_mode": permissions.get("mode", cls.model_fields["permission_mode"].default),
                "approvals": permissions.get("approvals", cls.model_fields["approvals"].default),
                "network": permissions.get("network", cls.model_fields["network"].default),
                "sandbox_backend": sandbox.get("backend", cls.model_fields["sandbox_backend"].default),
                "sandbox_fail_closed": sandbox.get("fail_closed", cls.model_fields["sandbox_fail_closed"].default),
                "ask_user_tool": runtime.get("ask_user_tool", cls.model_fields["ask_user_tool"].default),
            }
        )


class SearchConfig(RuntimeModel):
    provider: str = str(DEFAULT_CONFIG["search"]["provider"])
    fallback_provider: str = str(DEFAULT_CONFIG["search"]["fallback_provider"])
    searxng_base_url: str = str(DEFAULT_CONFIG["search"]["searxng_base_url"])
    tavily_api_key_env: str = str(DEFAULT_CONFIG["search"]["tavily_api_key_env"])
    provider_chain: tuple[str, ...] = tuple(DEFAULT_CONFIG["search"]["provider_chain"])
    cache_first: bool = bool(DEFAULT_CONFIG["search"]["cache_first"])
    min_usable_results: int = int(DEFAULT_CONFIG["search"]["min_usable_results"])
    fetch_min_chars: int = int(DEFAULT_CONFIG["search"]["fetch_min_chars"])

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        return cls.model_validate(_section(config, "search"))


class RetrievalConfig(RuntimeModel):
    enabled: bool = bool(DEFAULT_CONFIG["retrieval"]["enabled"])
    store_path: str = str(DEFAULT_CONFIG["retrieval"]["store_path"])
    web_ttl_hours: float = float(DEFAULT_CONFIG["retrieval"]["web_ttl_hours"])
    embeddings_enabled: bool = bool(DEFAULT_CONFIG["retrieval"]["embeddings"]["enabled"])

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        section = _section(config, "retrieval")
        embeddings = _section(section, "embeddings")
        return cls.model_validate({**section, "embeddings_enabled": embeddings.get("enabled", cls.model_fields["embeddings_enabled"].default)})


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


class TypedConfigV2(RuntimeModel):
    provider: ProviderConfig
    project: ProjectConfig
    memory: MemoryConfig
    runtime_policy: RuntimePolicyConfig
    skills: SkillsRuntimeConfig
    search: SearchConfig
    retrieval: RetrievalConfig
    ui: UiRuntimeConfig
    raw: dict[str, Any]

    @classmethod
    def from_normalized_config(cls, config: dict[str, Any], *, auth_header: str | None = None) -> Self:
        return cls(
            provider=ProviderConfig.from_config(config, auth_header=auth_header),
            project=ProjectConfig.from_config(config),
            memory=MemoryConfig.from_config(config),
            runtime_policy=RuntimePolicyConfig.from_config(config),
            skills=SkillsRuntimeConfig.from_config(config),
            search=SearchConfig.from_config(config),
            retrieval=RetrievalConfig.from_config(config),
            ui=UiRuntimeConfig.from_config(config),
            raw=config,
        )
