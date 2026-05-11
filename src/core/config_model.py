from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.configuration import DEFAULT_CONFIG
from core.runtime_config import ProviderConfig, SkillsRuntimeConfig, UiRuntimeConfig

CONFIG_MODEL_VERSION = "2.0.0"


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

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MemoryConfig:
        section = config.get("memory", {}) if isinstance(config.get("memory"), dict) else {}
        default_section = DEFAULT_CONFIG["memory"]
        return cls(
            min_score_default=float(section.get("min_score_default", default_section["min_score_default"])),
            recall_min_score_default=float(section.get("recall_min_score_default", default_section["recall_min_score_default"])),
            replace_min_score_default=float(section.get("replace_min_score_default", default_section["replace_min_score_default"])),
            backup_revisions=int(section.get("backup_revisions", default_section["backup_revisions"])),
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

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SearchConfig:
        section = config.get("search", {}) if isinstance(config.get("search"), dict) else {}
        default_section = DEFAULT_CONFIG["search"]
        return cls(provider=str(section.get("provider", default_section["provider"])))


@dataclass(frozen=True, slots=True)
class TypedConfigV2:
    provider: ProviderConfig
    workspace: WorkspaceConfig
    memory: MemoryConfig
    runtime_policy: RuntimePolicyConfig
    skills: SkillsRuntimeConfig
    search: SearchConfig
    ui: UiRuntimeConfig
    raw: dict[str, Any]
    model_version: str = CONFIG_MODEL_VERSION

    @classmethod
    def from_normalized_config(cls, config: dict[str, Any], *, auth_header: str | None = None) -> TypedConfigV2:
        return cls(
            provider=ProviderConfig.from_config(config, auth_header=auth_header),
            workspace=WorkspaceConfig.from_config(config),
            memory=MemoryConfig.from_config(config),
            runtime_policy=RuntimePolicyConfig.from_config(config),
            skills=SkillsRuntimeConfig.from_config(config),
            search=SearchConfig.from_config(config),
            ui=UiRuntimeConfig.from_config(config),
            raw=config,
        )
