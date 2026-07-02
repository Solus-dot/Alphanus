from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.configuration import DEFAULT_CONFIG
from core.runtime_config import ProviderConfig, SkillsRuntimeConfig, UiRuntimeConfig


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    root_strategy: str

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ProjectConfig:
        section = config.get("project", {}) if isinstance(config.get("project"), dict) else {}
        default_section = DEFAULT_CONFIG["project"]
        return cls(root_strategy=str(section.get("root_strategy", default_section["root_strategy"])).strip())


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
    permission_mode: str
    approvals: str
    network: bool
    sandbox_backend: str
    sandbox_fail_closed: bool
    ask_user_tool: bool

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RuntimePolicyConfig:
        runtime = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
        permissions = config.get("permissions", {}) if isinstance(config.get("permissions"), dict) else {}
        sandbox = config.get("sandbox", {}) if isinstance(config.get("sandbox"), dict) else {}
        default_runtime = DEFAULT_CONFIG["runtime"]
        default_permissions = DEFAULT_CONFIG["permissions"]
        default_sandbox = DEFAULT_CONFIG["sandbox"]
        return cls(
            permission_mode=str(permissions.get("mode", default_permissions["mode"])),
            approvals=str(permissions.get("approvals", default_permissions["approvals"])),
            network=bool(permissions.get("network", default_permissions["network"])),
            sandbox_backend=str(sandbox.get("backend", default_sandbox["backend"])),
            sandbox_fail_closed=bool(sandbox.get("fail_closed", default_sandbox["fail_closed"])),
            ask_user_tool=bool(runtime.get("ask_user_tool", default_runtime["ask_user_tool"])),
        )


@dataclass(frozen=True, slots=True)
class SearchConfig:
    provider: str
    fallback_provider: str
    searxng_base_url: str
    tavily_api_key_env: str
    provider_chain: tuple[str, ...]
    cache_first: bool
    min_usable_results: int
    fetch_min_chars: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SearchConfig:
        section = config.get("search", {}) if isinstance(config.get("search"), dict) else {}
        default_section = DEFAULT_CONFIG["search"]
        return cls(
            provider=str(section.get("provider", default_section["provider"])),
            fallback_provider=str(section.get("fallback_provider", default_section["fallback_provider"])),
            searxng_base_url=str(section.get("searxng_base_url", default_section["searxng_base_url"])),
            tavily_api_key_env=str(section.get("tavily_api_key_env", default_section["tavily_api_key_env"])),
            provider_chain=tuple(str(item) for item in section.get("provider_chain", default_section["provider_chain"])),
            cache_first=bool(section.get("cache_first", default_section["cache_first"])),
            min_usable_results=int(section.get("min_usable_results", default_section["min_usable_results"])),
            fetch_min_chars=int(section.get("fetch_min_chars", default_section["fetch_min_chars"])),
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
    project: ProjectConfig
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
            project=ProjectConfig.from_config(config),
            memory=MemoryConfig.from_config(config),
            runtime_policy=RuntimePolicyConfig.from_config(config),
            skills=SkillsRuntimeConfig.from_config(config),
            search=SearchConfig.from_config(config),
            retrieval=RetrievalConfig.from_config(config),
            ui=UiRuntimeConfig.from_config(config),
            raw=config,
        )
