from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.endpoint_modes import ENDPOINT_MODE_CHAT
from core.search_providers import DEFAULT_TAVILY_API_KEY_ENV, SEARCH_PROVIDER_SEARXNG, SEARCH_PROVIDER_TAVILY
from core.theme_catalog import DEFAULT_THEME_ID


class ConfigSection(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, populate_by_name=True)


class ClosedConfigSection(ConfigSection):
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)


class AgentConfig(ConfigSection):
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
    connect_timeout_s: float = Field(default=10, ge=0.1, le=60)
    request_timeout_s: float = Field(default=180, ge=5, le=600)
    readiness_timeout_s: float = Field(default=30, ge=1, le=300)
    readiness_poll_s: float = Field(default=0.5, ge=0.05, le=10)
    per_turn_retries: int = Field(default=1, ge=0, le=5)
    retry_backoff_s: float = Field(default=0.5, ge=0, le=30)
    enable_thinking: bool = True
    tls_verify: bool = True
    ca_bundle_path: str = ""
    allow_cross_host_endpoints: bool = False
    max_tokens: int | None = None
    context_budget_max_tokens: int = Field(default=2048, ge=256, le=262144)
    max_action_depth: int = Field(default=10, ge=1, le=100)
    max_tool_result_chars: int = Field(default=12000, ge=500, le=200000)
    max_reasoning_chars: int = Field(default=20000, ge=0, le=200000)
    recent_tool_detail_limit: int = Field(default=12, ge=1, le=100)
    compact_tool_results_in_history: bool = True
    compact_tool_result_tools: list[str] = Field(default_factory=list)
    classifier_model: str = ""
    classifier_use_primary_model: bool = True
    enable_structured_classification: bool = True
    max_classifier_tokens: int = Field(default=256, ge=32, le=4096)
    tool_budgets: dict[str, int] | None = None
    auth_header: str | None = Field(default=None, exclude=True)

class ProjectConfig(ClosedConfigSection):
    root_strategy: str = "git-or-cwd"


class MemoryConfig(ClosedConfigSection):
    min_score_default: float = Field(default=0.3, ge=0, le=1)
    recall_min_score_default: float = Field(default=0.18, ge=0, le=1)
    replace_min_score_default: float = Field(default=0.72, ge=0, le=1)
    backup_revisions: int = Field(default=2, ge=0, le=20)
    auto_capture: bool = True
    auto_capture_importance: float = Field(default=0.55, ge=0, le=1)


class ContextConfig(ConfigSection):
    context_limit: int = Field(default=8192, ge=512, le=262144)
    keep_last_n: int = Field(default=10, ge=1, le=100)
    safety_margin: int = Field(default=500, ge=0, le=100000)


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


class AgentsConfig(ConfigSection):
    enable_skill_agents: bool = True


class RuntimeConfig(ConfigSection):
    ask_user_tool: bool = True


class SearchConfig(ClosedConfigSection):
    provider: str = SEARCH_PROVIDER_SEARXNG
    fallback_provider: str = SEARCH_PROVIDER_TAVILY
    searxng_base_url: str = ""
    tavily_api_key_env: str = DEFAULT_TAVILY_API_KEY_ENV
    request_timeout_s: float = Field(default=20, ge=1)
    request_retries: int = Field(default=1, ge=0)
    request_retry_backoff_s: float = Field(default=0.5, ge=0)
    fetch_max_redirects: int = Field(default=5, ge=0)
    provider_chain: list[str] = Field(default_factory=list)
    cache_first: bool = True
    min_usable_results: int = Field(default=1, ge=1)
    fetch_min_chars: int = Field(default=20, ge=1)


class EmbeddingsConfig(ConfigSection):
    enabled: bool = False
    base_url: str = ""
    model: str = ""
    api_key_env: str = "ALPHANUS_EMBEDDINGS_API_KEY"
    dimensions: int = Field(default=0, ge=0)
    batch_size: int = Field(default=32, ge=1)


class RetrievalConfig(ConfigSection):
    enabled: bool = True
    store_path: str = ""
    web_ttl_hours: float = Field(default=72, ge=0)
    max_chunks_per_record: int = Field(default=64, ge=1)
    pre_context_top_k: int = Field(default=3, ge=0, le=10)
    embeddings: EmbeddingsConfig = Field(default_factory=lambda: EmbeddingsConfig())


class LoggingConfig(ConfigSection):
    level: str = "INFO"
    format: str = "json"
    path: str = "./logs/runtime.jsonl"


class UiTimingConfig(ConfigSection):
    stream_drain_interval_s: float = Field(default=0.033, ge=0.001, le=1)
    scroll_interval_s: float = Field(default=0.05, ge=0.001, le=1)
    action_approval_timeout_s: float = Field(default=60, ge=1, le=600)


class TreeCompactionConfig(ConfigSection):
    enabled: bool = True
    inactive_assistant_char_limit: int = Field(default=12000, ge=1000, le=200000)
    inactive_tool_argument_char_limit: int = Field(default=5000, ge=500, le=100000)
    inactive_tool_content_char_limit: int = Field(default=8000, ge=1000, le=200000)


class UiConfig(ConfigSection):
    theme: str = DEFAULT_THEME_ID
    chat_log_max_lines: int | None = Field(default=10000, ge=1000, le=200000)
    timing: UiTimingConfig = Field(default_factory=lambda: UiTimingConfig())
    tree_compaction: TreeCompactionConfig = Field(default_factory=lambda: TreeCompactionConfig())


class ConfigSchema(ConfigSection):
    config_version: int = 1
    agent: AgentConfig = Field(default_factory=lambda: AgentConfig())
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    memory: MemoryConfig = Field(default_factory=lambda: MemoryConfig())
    context: ContextConfig = Field(default_factory=lambda: ContextConfig())
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    tools: dict[str, Any] = Field(default_factory=dict)
    search: SearchConfig = Field(default_factory=lambda: SearchConfig())
    retrieval: RetrievalConfig = Field(default_factory=lambda: RetrievalConfig())
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    tui: UiConfig = Field(default_factory=lambda: UiConfig())


def config_schema(config: ConfigSchema | Mapping[str, Any] | None = None) -> ConfigSchema:
    if isinstance(config, ConfigSchema):
        return config
    return ConfigSchema.model_validate(dict(config or {}))


def validated_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    model = config_schema(config)
    output = model.model_dump(exclude={"agent": {"auth_header"}})
    if model.agent.tool_budgets is None:
        output["agent"].pop("tool_budgets", None)
    return output


def default_config() -> dict[str, Any]:
    return validated_config()
