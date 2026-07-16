from __future__ import annotations

import sys
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field

from core.endpoint_modes import ENDPOINT_MODE_CHAT
from core.search_providers import DEFAULT_TAVILY_API_KEY_ENV, SEARCH_PROVIDER_SEARXNG, SEARCH_PROVIDER_TAVILY
from core.theme_catalog import DEFAULT_THEME_ID


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    return value if isinstance(value, dict) else {}


class ConfigSection(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, populate_by_name=True)


class ClosedConfigSection(ConfigSection):
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)


class AgentConfig(ConfigSection):
    _transport_fields: ClassVar[frozenset[str]] = frozenset(
        "provider base_url model_endpoint responses_endpoint models_endpoint endpoint_mode backend_profile tls_verify ca_bundle_path "
        "allow_cross_host_endpoints request_timeout_s readiness_timeout_s readiness_poll_s connect_timeout_s per_turn_retries "
        "retry_backoff_s max_tokens api_key api_key_env auth_header_template".split()
    )
    provider: str = "openai-compatible"
    base_url: str = "http://127.0.0.1:8080"
    model_endpoint: str = "http://127.0.0.1:8080/v1/chat/completions"
    responses_endpoint: str = "http://127.0.0.1:8080/v1/responses"
    models_endpoint: str = "http://127.0.0.1:8080/v1/models"
    endpoint_mode: str = ENDPOINT_MODE_CHAT
    backend_profile: str = "auto"
    api_key: str = "env:ALPHANUS_API_KEY"
    api_key_env: str = "ALPHANUS_API_KEY"
    auth_header_template: str = "Authorization: Bearer {api_key}"
    connect_timeout_s: float = 10
    request_timeout_s: float = 180
    readiness_timeout_s: float = 30
    readiness_poll_s: float = 0.5
    per_turn_retries: int = 1
    retry_backoff_s: float = 0.5
    enable_thinking: bool = True
    tls_verify: bool = True
    ca_bundle_path: str = ""
    allow_cross_host_endpoints: bool = False
    max_tokens: int | None = None
    context_budget_max_tokens: int = 2048
    max_action_depth: int = 10
    max_tool_result_chars: int = 12000
    max_reasoning_chars: int = 20000
    compact_tool_results_in_history: bool = True
    compact_tool_result_tools: list[str] = Field(default_factory=list)
    classifier_model: str = ""
    classifier_use_primary_model: bool = True
    enable_structured_classification: bool = True
    max_classifier_tokens: int = 256
    tool_budgets: dict[str, int] | None = None
    auth_header: str | None = Field(default=None, exclude=True)

    @property
    def provider_name(self) -> str:
        return self.provider

    @property
    def allow_cross_host(self) -> bool:
        return self.allow_cross_host_endpoints

    @property
    def default_max_tokens(self) -> int | None:
        return self.max_tokens

    @classmethod
    def from_config(cls, config: dict[str, Any], *, auth_header: str | None = None) -> Self:
        section = _section(config, "agent")
        values = {key: value for key, value in section.items() if key in cls._transport_fields}
        return cls.model_validate({**values, "auth_header": (auth_header or "").strip() or None})


class ProjectConfig(ClosedConfigSection):
    root_strategy: str = "git-or-cwd"


class MemoryConfig(ClosedConfigSection):
    min_score_default: float = 0.3
    recall_min_score_default: float = 0.18
    replace_min_score_default: float = 0.72
    backup_revisions: int = 2
    auto_capture: bool = True


class ContextConfig(ConfigSection):
    context_limit: int = 8192
    keep_last_n: int = 10
    safety_margin: int = 500


class PermissionsConfig(ClosedConfigSection):
    mode: str = "project-write"
    approvals: str = "on-boundary"
    network: bool = False


class SandboxConfig(ClosedConfigSection):
    backend: str = "auto"
    fail_closed: bool = True


class SkillsConfig(ClosedConfigSection):
    strict_capability_policy: bool = False
    python_executable: str = ""
    paths: list[str] = Field(default_factory=list)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        section = _section(config, "skills")
        return cls.model_validate({**section, "python_executable": str(section.get("python_executable") or sys.executable)})


class AgentsConfig(ConfigSection):
    enable_skill_agents: bool = True


class RuntimeConfig(ConfigSection):
    ask_user_tool: bool = True


class SearchConfig(ClosedConfigSection):
    provider: str = SEARCH_PROVIDER_SEARXNG
    fallback_provider: str = SEARCH_PROVIDER_TAVILY
    searxng_base_url: str = ""
    tavily_api_key_env: str = DEFAULT_TAVILY_API_KEY_ENV
    request_timeout_s: float = 20
    request_retries: int = 1
    request_retry_backoff_s: float = 0.5
    fetch_max_redirects: int = 5
    provider_chain: list[str] = Field(default_factory=list)
    cache_first: bool = True
    min_usable_results: int = 1
    fetch_min_chars: int = 20


class EmbeddingsConfig(ConfigSection):
    enabled: bool = False
    base_url: str = ""
    model: str = ""
    api_key_env: str = "ALPHANUS_EMBEDDINGS_API_KEY"
    dimensions: int = 0
    batch_size: int = 32


class RetrievalConfig(ConfigSection):
    enabled: bool = True
    store_path: str = ""
    web_ttl_hours: int = 72
    max_chunks_per_record: int = 64
    pre_context_top_k: int = 3
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)


class LoggingConfig(ConfigSection):
    level: str = "INFO"
    format: str = "json"
    path: str = "./logs/runtime.jsonl"


class UiTimingConfig(ConfigSection):
    stream_drain_interval_s: float = 0.033
    scroll_interval_s: float = 0.05
    action_approval_timeout_s: float = 60.0


class TreeCompactionConfig(ConfigSection):
    enabled: bool = True
    inactive_assistant_char_limit: int = 12000
    inactive_tool_argument_char_limit: int = 5000
    inactive_tool_content_char_limit: int = 8000


class UiConfig(ConfigSection):
    theme: str = DEFAULT_THEME_ID
    chat_log_max_lines: int | None = 10000
    timing: UiTimingConfig = Field(default_factory=UiTimingConfig)
    tree_compaction: TreeCompactionConfig = Field(default_factory=TreeCompactionConfig)

    @property
    def tree_compaction_enabled(self) -> bool:
        return self.tree_compaction.enabled

    @property
    def inactive_assistant_char_limit(self) -> int:
        return self.tree_compaction.inactive_assistant_char_limit

    @property
    def inactive_tool_argument_char_limit(self) -> int:
        return self.tree_compaction.inactive_tool_argument_char_limit

    @property
    def inactive_tool_content_char_limit(self) -> int:
        return self.tree_compaction.inactive_tool_content_char_limit

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        return cls.model_validate(_section(config, "tui"))


class ConfigSchema(ConfigSection):
    config_version: int = 1
    agent: AgentConfig = Field(default_factory=AgentConfig)
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    tools: dict[str, Any] = Field(default_factory=dict)
    search: SearchConfig = Field(default_factory=SearchConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    tui: UiConfig = Field(default_factory=UiConfig)


ProviderConfig = AgentConfig
SkillsRuntimeConfig = SkillsConfig
UiRuntimeConfig = UiConfig


def validated_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    model = ConfigSchema.model_validate(config) if config is not None else ConfigSchema()
    output = model.model_dump(exclude={"agent": {"auth_header"}})
    if model.agent.tool_budgets is None:
        output["agent"].pop("tool_budgets", None)
    return output


def default_config() -> dict[str, Any]:
    return validated_config()
