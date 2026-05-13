from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.configuration import DEFAULT_CONFIG
from core.runtime_config import ProviderConfig, SkillsRuntimeConfig, UiRuntimeConfig


@dataclass(frozen=True, slots=True)
class WorkspaceConfig:
    path: str

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> WorkspaceConfig:
        section = config.get("workspace", {}) if isinstance(config.get("workspace"), dict) else {}
        default_section = DEFAULT_CONFIG["workspace"]
        return cls(path=str(section.get("path", default_section["path"])).strip())


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    min_score_default: float
    recall_min_score_default: float
    replace_min_score_default: float
    backup_revisions: int
    auto_capture: bool

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MemoryConfig:
        section = config.get("memory", {}) if isinstance(config.get("memory"), dict) else {}
        default_section = DEFAULT_CONFIG["memory"]
        return cls(
            min_score_default=float(section.get("min_score_default", default_section["min_score_default"])),
            recall_min_score_default=float(section.get("recall_min_score_default", default_section["recall_min_score_default"])),
            replace_min_score_default=float(section.get("replace_min_score_default", default_section["replace_min_score_default"])),
            backup_revisions=int(section.get("backup_revisions", default_section["backup_revisions"])),
            auto_capture=bool(section.get("auto_capture", default_section["auto_capture"])),
        )


@dataclass(frozen=True, slots=True)
class RuntimePolicyConfig:
    runtime_profile: str
    permission_profile: str
    ask_user_tool: bool
    shell_require_confirmation: bool

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RuntimePolicyConfig:
        runtime = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
        capabilities = config.get("capabilities", {}) if isinstance(config.get("capabilities"), dict) else {}
        default_runtime = DEFAULT_CONFIG["runtime"]
        default_capabilities = DEFAULT_CONFIG["capabilities"]
        return cls(
            runtime_profile=str(runtime.get("profile", default_runtime["profile"])),
            permission_profile=str(capabilities.get("permission_profile", default_capabilities["permission_profile"])),
            ask_user_tool=bool(runtime.get("ask_user_tool", default_runtime["ask_user_tool"])),
            shell_require_confirmation=bool(
                capabilities.get("shell_require_confirmation", default_capabilities["shell_require_confirmation"])
            ),
        )


@dataclass(frozen=True, slots=True)
class SearchConfig:
    provider: str
    fallback_provider: str
    searxng_base_url: str
    tavily_api_key_env: str

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SearchConfig:
        section = config.get("search", {}) if isinstance(config.get("search"), dict) else {}
        default_section = DEFAULT_CONFIG["search"]
        return cls(
            provider=str(section.get("provider", default_section["provider"])),
            fallback_provider=str(section.get("fallback_provider", default_section["fallback_provider"])),
            searxng_base_url=str(section.get("searxng_base_url", default_section["searxng_base_url"])),
            tavily_api_key_env=str(section.get("tavily_api_key_env", default_section["tavily_api_key_env"])),
        )


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    enabled: bool
    store_path: str
    web_ttl_hours: float
    embeddings_enabled: bool

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RetrievalConfig:
        section = config.get("retrieval", {}) if isinstance(config.get("retrieval"), dict) else {}
        default_section = DEFAULT_CONFIG["retrieval"]
        embeddings = section.get("embeddings", {}) if isinstance(section.get("embeddings"), dict) else {}
        default_embeddings = default_section["embeddings"]
        return cls(
            enabled=bool(section.get("enabled", default_section["enabled"])),
            store_path=str(section.get("store_path", default_section["store_path"])),
            web_ttl_hours=float(section.get("web_ttl_hours", default_section["web_ttl_hours"])),
            embeddings_enabled=bool(embeddings.get("enabled", default_embeddings["enabled"])),
        )


@dataclass(frozen=True, slots=True)
class TypedConfigV2:
    provider: ProviderConfig
    workspace: WorkspaceConfig
    memory: MemoryConfig
    runtime_policy: RuntimePolicyConfig
    skills: SkillsRuntimeConfig
    search: SearchConfig
    retrieval: RetrievalConfig
    ui: UiRuntimeConfig
    raw: dict[str, Any]

    @classmethod
    def from_normalized_config(cls, config: dict[str, Any], *, auth_header: str | None = None) -> TypedConfigV2:
        return cls(
            provider=ProviderConfig.from_config(config, auth_header=auth_header),
            workspace=WorkspaceConfig.from_config(config),
            memory=MemoryConfig.from_config(config),
            runtime_policy=RuntimePolicyConfig.from_config(config),
            skills=SkillsRuntimeConfig.from_config(config),
            search=SearchConfig.from_config(config),
            retrieval=RetrievalConfig.from_config(config),
            ui=UiRuntimeConfig.from_config(config),
            raw=config,
        )
