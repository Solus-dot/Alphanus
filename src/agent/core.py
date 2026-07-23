from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.classifier import TurnClassifier
from agent.context import ContextWindowManager
from agent.harness_metrics import HarnessMetrics
from agent.orchestrator import TurnOrchestrator, request_user_input_passthrough
from agent.policies import PromptPolicyRenderer
from agent.prompts import build_system_prompt
from agent.provider import LLMClient
from agent.telemetry import TelemetryEmitter
from core.config_model import ConfigSchema, config_schema
from core.configuration import normalize_config, validate_endpoint_policy
from core.message_types import ChatMessage, JsonObject
from core.retrieval import SQLiteRetrievalStore, configured_store_path
from core.search_providers import DEFAULT_TAVILY_API_KEY_ENV, SEARCH_PROVIDER_SEARXNG, SEARCH_PROVIDER_TAVILY
from core.types import AgentTurnResult, ApprovalRequestFn, ModelStatus
from skills.runtime import SkillRuntime


class Agent:
    def __init__(self, config: ConfigSchema | Mapping[str, Any], skill_runtime: SkillRuntime, debug: bool = False) -> None:
        config = config if isinstance(config, ConfigSchema) else config_schema(normalize_config(dict(config))[0])
        self.skill_runtime = skill_runtime
        self.debug = debug
        self.telemetry = TelemetryEmitter()
        self.harness_metrics = HarnessMetrics()
        self.system_prompt = build_system_prompt(str(self.skill_runtime.project.project_root))
        self.context_mgr = ContextWindowManager()
        self.llm_client = LLMClient(config, debug=debug, telemetry=self.telemetry)
        self.classifier = TurnClassifier(config, skill_runtime, self.llm_client, telemetry=self.telemetry)
        self.prompt_renderer = PromptPolicyRenderer(
            self.system_prompt,
            self.skill_runtime,
            context_limit=self.context_mgr.context_limit,
        )
        self.orchestrator = TurnOrchestrator(
            skill_runtime=self.skill_runtime,
            context_mgr=self.context_mgr,
            llm_client=self.llm_client,
            classifier=self.classifier,
            prompt_renderer=self.prompt_renderer,
            telemetry=self.telemetry,
        )
        self._apply_config(config)

    def reload_config(self, config: ConfigSchema | Mapping[str, Any]) -> None:
        model = config if isinstance(config, ConfigSchema) else config_schema(normalize_config(dict(config))[0])
        self._apply_config(model)

    def _apply_config(self, config: ConfigSchema) -> None:
        self.config = config_schema(config)
        self.skill_runtime.reload_config(self.config)
        self.context_mgr.context_limit = self.config.context.context_limit
        self.context_mgr.keep_last_n = self.config.context.keep_last_n
        self.context_mgr.safety_margin = self.config.context.safety_margin
        self.system_prompt = build_system_prompt(str(self.skill_runtime.project.project_root))
        self.llm_client.reload_config(self.config)
        self.prompt_renderer.system_prompt = self.system_prompt
        self.prompt_renderer.context_limit = self.context_mgr.context_limit
        self.orchestrator.reload_config(self.config)

    def ensure_ready(
        self, stop_event=None, on_event: Callable[[dict[str, Any]], None] | None = None, timeout_s: float | None = None
    ) -> bool | None:
        return self.llm_client.ensure_ready(stop_event=stop_event, on_event=on_event, timeout_s=timeout_s)

    def get_model_status(self) -> ModelStatus:
        return self.llm_client.get_model_status()

    def refresh_model_status(self, timeout_s: float | None = None, force: bool = False) -> ModelStatus:
        return self.llm_client.refresh_model_status(timeout_s=timeout_s, force=force)

    def _offline_status_detail(self, status: ModelStatus) -> str:
        return f": {self.llm_client.friendly_endpoint_error(status.last_error)}" if status.last_error else ""

    @staticmethod
    def _empty_result(status: str, error: str = "") -> AgentTurnResult:
        return AgentTurnResult(status=status, content="", reasoning="", skill_exchanges=[], error=error)

    def _not_ready_result(self, status: ModelStatus) -> AgentTurnResult:
        error = (
            f"Model endpoint offline{self._offline_status_detail(status)}"
            if status.state == "offline"
            else f"Model endpoint not ready: {self.llm_client.models_endpoint}"
        )
        return self._empty_result("error", error)

    def _validate_endpoints(self) -> str | None:
        try:
            validate_endpoint_policy(
                {
                    "agent": {
                        "base_url": self.llm_client.base_url,
                        "model_endpoint": self.llm_client.model_endpoint,
                        "responses_endpoint": self.llm_client.responses_endpoint,
                        "models_endpoint": self.llm_client.models_endpoint,
                        "allow_cross_host_endpoints": self.llm_client.allow_cross_host,
                    }
                }
            )
        except ValueError as exc:
            return str(exc)
        return None

    def doctor_report(self, *, probe_ready: bool = True) -> dict[str, object]:
        config_obj: ConfigSchema = self.config
        endpoint_error = self._validate_endpoints()
        project_root = Path(self.skill_runtime.project.project_root)
        memory_stats = self.skill_runtime.memory.stats()
        permissions_cfg = config_obj.permissions
        sandbox_cfg = config_obj.sandbox
        search_cfg = config_obj.search
        provider = search_cfg.provider.strip().lower() or SEARCH_PROVIDER_SEARXNG
        fallback_provider = search_cfg.fallback_provider.strip().lower()
        searxng_base_url = search_cfg.searxng_base_url.strip()
        tavily_api_key_env = search_cfg.tavily_api_key_env.strip() or DEFAULT_TAVILY_API_KEY_ENV
        tavily_ready = bool(os.environ.get(tavily_api_key_env, "").strip())
        search_ready = (
            (provider == SEARCH_PROVIDER_SEARXNG and bool(searxng_base_url)) or provider == SEARCH_PROVIDER_TAVILY and tavily_ready
        )
        if provider == SEARCH_PROVIDER_SEARXNG and fallback_provider == SEARCH_PROVIDER_TAVILY:
            search_ready = bool(searxng_base_url) or tavily_ready
        search_reason = ""
        if not search_ready:
            search_reason = (
                f"missing env: {tavily_api_key_env}" if provider == SEARCH_PROVIDER_TAVILY else "missing search.searxng_base_url"
            )
            if provider == SEARCH_PROVIDER_SEARXNG and fallback_provider == SEARCH_PROVIDER_TAVILY:
                search_reason = f"missing search.searxng_base_url and env: {tavily_api_key_env}"
        retrieval_enabled = config_obj.retrieval.enabled
        try:
            retrieval_stats = SQLiteRetrievalStore(configured_store_path(config_obj)).stats() if retrieval_enabled else {}
            retrieval_ready = retrieval_enabled
            retrieval_reason = ""
        except Exception as exc:
            retrieval_stats = {}
            retrieval_ready = False
            retrieval_reason = str(exc)
        if probe_ready:
            ready = self.ensure_ready(timeout_s=min(self.llm_client.readiness_timeout_s, 3.0))
        else:
            ready = self.get_model_status().state == "online"
        backend_info = self.llm_client.backend_profile_info()
        sandbox_preflight = self.skill_runtime.project.sandbox_preflight()
        return {
            "agent": {
                "base_url": self.llm_client.base_url,
                "model_endpoint": self.llm_client.model_endpoint,
                "responses_endpoint": self.llm_client.responses_endpoint,
                "models_endpoint": self.llm_client.models_endpoint,
                "ready": bool(ready),
                "endpoint_policy_error": endpoint_error or "",
                "auth_header_source": self.llm_client.auth_source,
                "endpoint_mode": self.llm_client.endpoint_mode,
                "backend_profile_requested": str(backend_info.get("requested", "auto")),
                "backend_profile_detected": str(backend_info.get("detected", "unknown")),
                "backend_profile_selected": str(backend_info.get("selected", "unknown")),
                "backend_profile_reason": str(backend_info.get("reason", "")),
                "backend_capabilities": backend_info.get("capabilities", {}),
                "backend_model_integrity": str(backend_info.get("model_integrity", "unknown")),
                "backend_incompatibility_last": str(backend_info.get("incompatibility_last", "")),
                "compatibility_profile": self.llm_client.compatibility_profile(),
                "fallback_events": self.llm_client.fallback_events(),
                "permission_mode": permissions_cfg.mode,
                "approvals": permissions_cfg.approvals,
                "network": permissions_cfg.network,
                "sandbox_backend": sandbox_cfg.backend,
            },
            "project": {
                "path": str(project_root),
                "exists": project_root.exists(),
                "writable": os.access(project_root, os.W_OK),
            },
            "sandbox": sandbox_preflight,
            "memory": {
                "backend": memory_stats.get("backend"),
                "mode": memory_stats.get("mode_label"),
                "min_score_default": memory_stats.get("min_score_default"),
                "count": memory_stats.get("count"),
                "load_recovery_count": memory_stats.get("load_recovery_count"),
                "backup_revisions": memory_stats.get("backup_revisions"),
            },
            "search": {
                "provider": provider,
                "fallback_provider": fallback_provider,
                "ready": search_ready,
                "searxng_base_url": searxng_base_url,
                "tavily_api_key_env": tavily_api_key_env,
                "tavily_ready": tavily_ready,
                "reason": search_reason,
            },
            "retrieval": {
                "enabled": retrieval_enabled,
                "ready": retrieval_ready,
                "reason": retrieval_reason,
                **retrieval_stats,
            },
            "harness_metrics": self.harness_metrics.snapshot(),
            "skills": self.skill_runtime.skill_health_report(),
        }

    def build_support_bundle(self, tree_payload: dict[str, object]) -> dict[str, object]:
        return {
            "created_at": datetime.now(UTC).isoformat(),
            "doctor": self.doctor_report(),
            "tree": tree_payload,
        }

    def reload_skills(self) -> int:
        return self.classifier.reload_skills()

    def _record_and_return(self, result: AgentTurnResult) -> AgentTurnResult:
        try:
            self.harness_metrics.record(result)
        except Exception as exc:
            logging.debug("Harness metric recording failed: %s", exc)
        return result

    def run_turn(
        self,
        history_messages: list[ChatMessage],
        user_input: str,
        thinking: bool,
        branch_labels: list[str] | None = None,
        attachments: list[str] | None = None,
        loaded_skill_ids: list[str] | None = None,
        context_summary: str = "",
        collaboration_mode: str = "execute",
        session_id: str = "",
        stop_event=None,
        on_event: Callable[[JsonObject], None] | None = None,
        request_approval: ApprovalRequestFn | None = None,
    ) -> AgentTurnResult:
        endpoint_err = self._validate_endpoints()
        if endpoint_err:
            return self._record_and_return(self._empty_result("error", endpoint_err))
        if self.llm_client.stop_requested(stop_event):
            return self._record_and_return(self._empty_result("cancelled"))
        status = self.get_model_status()
        if status.state == "offline" and self.llm_client.is_model_status_fresh(status):
            if self.llm_client.should_fail_fast_on_offline_status(status):
                return self._record_and_return(self._not_ready_result(status))
            ready = self.ensure_ready(stop_event=stop_event, on_event=on_event)
            if ready is None:
                return self._record_and_return(self._empty_result("cancelled"))
            if not ready:
                return self._record_and_return(self._not_ready_result(self.get_model_status()))
        if not self.llm_client.is_model_status_fresh(status):
            if status.state != "unknown":
                status = self.refresh_model_status(timeout_s=min(self.llm_client.connect_timeout_s, 1.0), force=True)
            if status.state != "online":
                ready = self.ensure_ready(stop_event=stop_event, on_event=on_event)
                if ready is None:
                    return self._record_and_return(self._empty_result("cancelled"))
                if not ready:
                    return self._record_and_return(self._not_ready_result(self.get_model_status()))
        return self._record_and_return(
            self.orchestrator.run_turn(
                history_messages=history_messages,
                user_input=user_input,
                thinking=thinking,
                branch_labels=branch_labels,
                attachments=attachments,
                loaded_skill_ids=loaded_skill_ids,
                context_summary=context_summary,
                collaboration_mode=collaboration_mode,
                session_id=session_id,
                stop_event=stop_event,
                on_event=on_event,
                request_approval=request_approval,
                request_user_input=request_user_input_passthrough,
            )
        )
