from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.classifier import TurnClassifier
from agent.config_values import coerce_int, get_json_object
from agent.context import DEFAULT_CONTEXT_LIMIT, DEFAULT_KEEP_LAST_N, DEFAULT_SAFETY_MARGIN, ContextWindowManager
from agent.harness_metrics import HarnessMetrics
from agent.llm_client import LLMClient
from agent.orchestrator import TurnOrchestrator, request_user_input_passthrough
from agent.policies import PromptPolicyRenderer
from agent.prompts import build_system_prompt
from agent.runtime_hooks import AgentTurnRuntimeHooks
from agent.telemetry import TelemetryEmitter
from core.config_model import TypedConfigV2
from core.configuration import validate_endpoint_policy
from core.message_types import ChatMessage, JsonObject
from core.retrieval import SQLiteRetrievalStore, configured_store_path
from core.search_providers import DEFAULT_TAVILY_API_KEY_ENV, SEARCH_PROVIDER_SEARXNG, SEARCH_PROVIDER_TAVILY
from core.types import AgentTurnResult, ModelStatus, ShellConfirmationFn
from skills.runtime import SkillRuntime


class Agent:
    def __init__(self, config: dict[str, Any], skill_runtime: SkillRuntime, debug: bool = False) -> None:
        self.skill_runtime = skill_runtime
        self.debug = debug
        self.telemetry = TelemetryEmitter()
        self.harness_metrics = HarnessMetrics()
        self.system_prompt = build_system_prompt(str(self.skill_runtime.workspace.workspace_root))
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
        self._runtime_hooks = AgentTurnRuntimeHooks(self)
        self.classifier.bind_runtime_hooks(self._runtime_hooks)
        self.orchestrator.bind_runtime_hooks(self._runtime_hooks)
        self.reload_config(config)

    @property
    def connect_timeout_s(self) -> float:
        return float(self.llm_client.connect_timeout_s)

    @property
    def request_timeout_s(self) -> float:
        return float(self.llm_client.request_timeout_s)

    @property
    def classifier_model(self) -> str:
        return str(self.llm_client.classifier_model)

    @property
    def classifier_use_primary_model(self) -> bool:
        return bool(self.llm_client.classifier_use_primary_model)

    @property
    def max_classifier_tokens(self) -> int:
        return int(self.llm_client.max_classifier_tokens)

    @property
    def auth_header(self) -> str | None:
        return self.llm_client.auth_header

    def reload_config(self, config: dict[str, Any]) -> None:
        self.config = config
        self.skill_runtime.reload_config(config)
        self.skill_runtime.refresh_process_env()
        self.skill_runtime.load_skills()
        context_cfg = get_json_object(config, "context")
        self.context_mgr.context_limit = coerce_int(context_cfg.get("context_limit"), DEFAULT_CONTEXT_LIMIT, minimum=1)
        self.context_mgr.keep_last_n = coerce_int(context_cfg.get("keep_last_n"), DEFAULT_KEEP_LAST_N, minimum=1)
        self.context_mgr.safety_margin = coerce_int(context_cfg.get("safety_margin"), DEFAULT_SAFETY_MARGIN, minimum=0)
        self.system_prompt = build_system_prompt(str(self.skill_runtime.workspace.workspace_root))
        self.llm_client.reload_config(config)
        self.typed_config = TypedConfigV2.from_normalized_config(config, auth_header=self.llm_client.auth_header)
        self.classifier.reload_config(config)
        self.classifier.bind_runtime_hooks(self._runtime_hooks)
        self.prompt_renderer.system_prompt = self.system_prompt
        self.prompt_renderer.skill_runtime = self.skill_runtime
        self.prompt_renderer.context_limit = self.context_mgr.context_limit
        self.orchestrator.skill_runtime = self.skill_runtime
        self.orchestrator.context_mgr = self.context_mgr
        self.orchestrator.llm_client = self.llm_client
        self.orchestrator.classifier = self.classifier
        self.orchestrator.prompt_renderer = self.prompt_renderer
        self.orchestrator.bind_runtime_hooks(self._runtime_hooks)
        self.orchestrator.reload_config(config)
        self.model_endpoint = self.llm_client.model_endpoint
        self.models_endpoint = self.llm_client.models_endpoint
        self.allow_cross_host = self.llm_client.allow_cross_host
        self.readiness_timeout_s = self.llm_client.readiness_timeout_s

    def ensure_ready(
        self, stop_event=None, on_event: Callable[[dict[str, Any]], None] | None = None, timeout_s: float | None = None
    ) -> bool | None:
        return self.llm_client.ensure_ready(stop_event=stop_event, on_event=on_event, timeout_s=timeout_s)

    def fetch_model_metadata(self, timeout_s: float | None = None) -> tuple[str | None, int | None]:
        return self.llm_client.fetch_model_metadata(timeout_s=timeout_s)

    def get_model_status(self) -> ModelStatus:
        return self.llm_client.get_model_status()

    def refresh_model_status(self, timeout_s: float | None = None, force: bool = False) -> ModelStatus:
        return self.llm_client.refresh_model_status(timeout_s=timeout_s, force=force)

    def mark_model_transport_failure(self, exc: Exception) -> None:
        self.llm_client.mark_model_transport_failure(exc)

    def _offline_status_detail(self, status: ModelStatus) -> str:
        return f": {self.llm_client.friendly_endpoint_error(status.last_error)}" if status.last_error else ""

    def fetch_model_name(self, timeout_s: float | None = None) -> str | None:
        model_name, _context_window = self.fetch_model_metadata(timeout_s=timeout_s)
        return model_name

    def _validate_endpoints(self) -> str | None:
        try:
            validate_endpoint_policy(
                {
                    "agent": {
                        "base_url": self.llm_client.base_url,
                        "model_endpoint": self.model_endpoint,
                        "responses_endpoint": self.llm_client.responses_endpoint,
                        "models_endpoint": self.models_endpoint,
                        "allow_cross_host_endpoints": self.allow_cross_host,
                    }
                }
            )
        except ValueError as exc:
            return str(exc)
        return None

    def doctor_report(self, *, probe_ready: bool = True) -> dict[str, object]:
        config_obj = self.config if isinstance(self.config, dict) else {}
        endpoint_error = self._validate_endpoints()
        workspace_root = Path(self.skill_runtime.workspace.workspace_root)
        memory_stats = self.skill_runtime.memory.stats()
        runtime_cfg = get_json_object(config_obj, "runtime")
        capabilities_cfg = get_json_object(config_obj, "capabilities")
        search_cfg = get_json_object(config_obj, "search")
        provider = str(search_cfg.get("provider", SEARCH_PROVIDER_SEARXNG)).strip().lower() or SEARCH_PROVIDER_SEARXNG
        fallback_provider = str(search_cfg.get("fallback_provider", SEARCH_PROVIDER_TAVILY)).strip().lower()
        searxng_base_url = str(search_cfg.get("searxng_base_url", "")).strip()
        tavily_api_key_env = str(search_cfg.get("tavily_api_key_env", DEFAULT_TAVILY_API_KEY_ENV)).strip() or DEFAULT_TAVILY_API_KEY_ENV
        tavily_ready = bool(os.environ.get(tavily_api_key_env, "").strip())
        search_ready = (provider == SEARCH_PROVIDER_SEARXNG and bool(searxng_base_url)) or provider == SEARCH_PROVIDER_TAVILY and tavily_ready
        if provider == SEARCH_PROVIDER_SEARXNG and fallback_provider == SEARCH_PROVIDER_TAVILY:
            search_ready = bool(searxng_base_url) or tavily_ready
        search_reason = ""
        if not search_ready:
            search_reason = f"missing env: {tavily_api_key_env}" if provider == SEARCH_PROVIDER_TAVILY else "missing search.searxng_base_url"
            if provider == SEARCH_PROVIDER_SEARXNG and fallback_provider == SEARCH_PROVIDER_TAVILY:
                search_reason = f"missing search.searxng_base_url and env: {tavily_api_key_env}"
        retrieval_cfg = get_json_object(config_obj, "retrieval")
        retrieval_enabled = bool(retrieval_cfg.get("enabled", True))
        try:
            retrieval_stats = SQLiteRetrievalStore(configured_store_path(config_obj)).stats() if retrieval_enabled else {}
            retrieval_ready = retrieval_enabled
            retrieval_reason = ""
        except Exception as exc:
            retrieval_stats = {}
            retrieval_ready = False
            retrieval_reason = str(exc)
        if probe_ready:
            ready = self.ensure_ready(timeout_s=min(self.readiness_timeout_s, 3.0))
        else:
            ready = self.get_model_status().state == "online"
        backend_info = self.llm_client.backend_profile_info()
        return {
            "agent": {
                "base_url": self.llm_client.base_url,
                "model_endpoint": self.model_endpoint,
                "responses_endpoint": self.llm_client.responses_endpoint,
                "models_endpoint": self.models_endpoint,
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
                "runtime_profile": str(runtime_cfg.get("profile", "standard")),
                "permission_profile": str(capabilities_cfg.get("permission_profile", "full")),
            },
            "workspace": {
                "path": str(workspace_root),
                "exists": workspace_root.exists(),
                "writable": os.access(workspace_root, os.W_OK),
            },
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
        stop_event=None,
        on_event: Callable[[JsonObject], None] | None = None,
        confirm_shell: ShellConfirmationFn | None = None,
    ) -> AgentTurnResult:
        endpoint_err = self._validate_endpoints()
        if endpoint_err:
            return self._record_and_return(
                AgentTurnResult(status="error", content="", reasoning="", skill_exchanges=[], error=endpoint_err)
            )
        if self.llm_client.stop_requested(stop_event):
            return self._record_and_return(AgentTurnResult(status="cancelled", content="", reasoning="", skill_exchanges=[]))
        status = self.get_model_status()
        if status.state == "offline" and self.llm_client.is_model_status_fresh(status):
            if self.llm_client.should_fail_fast_on_offline_status(status):
                detail = self._offline_status_detail(status)
                return self._record_and_return(
                    AgentTurnResult(
                        status="error",
                        content="",
                        reasoning="",
                        skill_exchanges=[],
                        error=f"Model endpoint offline{detail}",
                    )
                )
            ready = self.ensure_ready(stop_event=stop_event, on_event=on_event)
            if ready is None:
                return self._record_and_return(AgentTurnResult(status="cancelled", content="", reasoning="", skill_exchanges=[]))
            if not ready:
                refreshed = self.get_model_status()
                detail = self._offline_status_detail(refreshed)
                return self._record_and_return(
                    AgentTurnResult(
                        status="error",
                        content="",
                        reasoning="",
                        skill_exchanges=[],
                        error=(
                            f"Model endpoint offline{detail}"
                            if refreshed.state == "offline"
                            else f"Model endpoint not ready: {self.models_endpoint}"
                        ),
                    )
                )
        if not self.llm_client.is_model_status_fresh(status):
            if status.state != "unknown":
                status = self.refresh_model_status(timeout_s=min(self.connect_timeout_s, 1.0), force=True)
                if status.state == "online":
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
                            stop_event=stop_event,
                            on_event=on_event,
                            confirm_shell=confirm_shell,
                            request_user_input=request_user_input_passthrough,
                        )
                    )
            ready = self.ensure_ready(stop_event=stop_event, on_event=on_event)
            if ready is None:
                return self._record_and_return(AgentTurnResult(status="cancelled", content="", reasoning="", skill_exchanges=[]))
            if not ready:
                status = self.get_model_status()
                detail = self._offline_status_detail(status)
                return self._record_and_return(
                    AgentTurnResult(
                        status="error",
                        content="",
                        reasoning="",
                        skill_exchanges=[],
                        error=(
                            f"Model endpoint offline{detail}"
                            if status.state == "offline"
                            else f"Model endpoint not ready: {self.models_endpoint}"
                        ),
                    )
                )
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
                stop_event=stop_event,
                on_event=on_event,
                confirm_shell=confirm_shell,
                request_user_input=request_user_input_passthrough,
            )
        )
