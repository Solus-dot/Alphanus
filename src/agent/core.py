from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

from core.message_types import ChatMessage
from agent.classifier import TurnClassifier
from agent.context import ContextWindowManager
from agent.llm_client import LLMClient
from agent.orchestrator import TurnOrchestrator, request_user_input_passthrough
from agent.policies import PromptPolicyRenderer
from agent.prompts import build_system_prompt
from agent.telemetry import TelemetryEmitter
from core.configuration import validate_endpoint_policy
from core.types import AgentTurnResult, JsonObject, ModelStatus, ShellConfirmationFn
from core.skills import SkillRuntime


class Agent:
    def __init__(self, config: JsonObject, skill_runtime: SkillRuntime, debug: bool = False) -> None:
        self.skill_runtime = skill_runtime
        self.debug = debug
        self.telemetry = TelemetryEmitter()
        self.system_prompt = build_system_prompt(self.skill_runtime.workspace.workspace_root)
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
        self.reload_config(config)

    @property
    def _ready_checked(self) -> bool:
        return bool(self.llm_client._ready_checked)

    @_ready_checked.setter
    def _ready_checked(self, value: bool) -> None:
        self.llm_client._ready_checked = bool(value)

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
    def auth_header(self) -> Optional[str]:
        return self.llm_client.auth_header

    def reload_config(self, config: JsonObject) -> None:
        self.config = config
        self.skill_runtime.reload_config(config)
        self.skill_runtime._proc_env_base = self.skill_runtime._build_proc_env_base()
        self.skill_runtime.load_skills()
        context_cfg = config.get("context", {}) if isinstance(config.get("context"), dict) else {}
        self.context_mgr = ContextWindowManager(
            context_limit=int(context_cfg.get("context_limit", 8192)),
            keep_last_n=int(context_cfg.get("keep_last_n", 10)),
            safety_margin=int(context_cfg.get("safety_margin", 500)),
        )
        self.system_prompt = build_system_prompt(self.skill_runtime.workspace.workspace_root)
        self.llm_client.reload_config(config)
        self.classifier.reload_config(config)
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
        self.classifier.call_with_retry = lambda payload, stop_event, on_event, pass_id: self._call_with_retry(payload, stop_event, on_event, pass_id)
        self.orchestrator.call_with_retry = lambda payload, stop_event, on_event, pass_id: self._call_with_retry(payload, stop_event, on_event, pass_id)
        self.orchestrator.build_skill_context = lambda user_input, branch_labels, attachments, history_messages, loaded_skill_ids: self._build_skill_context(
            user_input,
            branch_labels,
            attachments,
            history_messages,
            loaded_skill_ids,
        )
        self.orchestrator.classify_context = lambda ctx, stop_event=None: self._classify_turn(ctx, stop_event)
        self.orchestrator.select_skills = lambda ctx, stop_event: self._select_turn(ctx, stop_event)
        self.model_endpoint = self.llm_client.model_endpoint
        self.models_endpoint = self.llm_client.models_endpoint
        self.allow_cross_host = self.llm_client.allow_cross_host
        self.readiness_timeout_s = self.llm_client.readiness_timeout_s

    def ensure_ready(self, stop_event=None, on_event: Optional[Callable[[JsonObject], None]] = None, timeout_s: Optional[float] = None) -> Optional[bool]:
        return self.llm_client.ensure_ready(stop_event=stop_event, on_event=on_event, timeout_s=timeout_s)

    def fetch_model_metadata(self, timeout_s: Optional[float] = None) -> tuple[Optional[str], Optional[int]]:
        return self.llm_client.fetch_model_metadata(timeout_s=timeout_s)

    def get_model_status(self) -> ModelStatus:
        return self.llm_client.get_model_status()

    def refresh_model_status(self, timeout_s: Optional[float] = None, force: bool = False) -> ModelStatus:
        return self.llm_client.refresh_model_status(timeout_s=timeout_s, force=force)

    def mark_model_transport_failure(self, exc: Exception) -> None:
        self.llm_client.mark_model_transport_failure(exc)

    def fetch_model_name(self, timeout_s: Optional[float] = None) -> Optional[str]:
        model_name, _context_window = self.fetch_model_metadata(timeout_s=timeout_s)
        return model_name

    def _validate_endpoints(self) -> Optional[str]:
        try:
            validate_endpoint_policy(
                {
                    "agent": {
                        "model_endpoint": self.model_endpoint,
                        "models_endpoint": self.models_endpoint,
                        "allow_cross_host_endpoints": self.allow_cross_host,
                    }
                }
            )
        except ValueError as exc:
            return str(exc)
        return None

    def doctor_report(self) -> dict[str, object]:
        endpoint_error = self._validate_endpoints()
        workspace_root = Path(self.skill_runtime.workspace.workspace_root)
        memory_stats = self.skill_runtime.memory.stats()
        search_cfg = self.config.get("search", {}) if isinstance(self.config, dict) else {}
        provider = str(search_cfg.get("provider", "tavily")).strip().lower() or "tavily"
        provider_env = {"tavily": "TAVILY_API_KEY", "brave": "BRAVE_SEARCH_API_KEY"}
        required_env = provider_env.get(provider, "")
        search_ready = bool(os.environ.get(required_env, "").strip()) if required_env else False
        ready = self.ensure_ready(timeout_s=min(self.readiness_timeout_s, 3.0))
        return {
            "agent": {
                "model_endpoint": self.model_endpoint,
                "models_endpoint": self.models_endpoint,
                "ready": bool(ready),
                "endpoint_policy_error": endpoint_error or "",
                "auth_header_source": "env" if self.llm_client.auth_header else "none",
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
                "ready": search_ready,
                "reason": "" if search_ready or not required_env else f"missing env: {required_env}",
            },
            "skills": self.skill_runtime.skill_health_report(),
        }

    def build_support_bundle(self, tree_payload: dict[str, object]) -> dict[str, object]:
        return {
            "schema_version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "doctor": self.doctor_report(),
            "tree": tree_payload,
        }

    def reload_skills(self) -> int:
        return self.classifier.reload_skills()

    @staticmethod
    def _extract_model_name(payload: object) -> Optional[str]:
        return LLMClient.extract_model_name(payload)

    @staticmethod
    def _extract_model_context_window(payload: object) -> Optional[int]:
        return LLMClient.extract_model_context_window(payload)

    def _build_skill_context(
        self,
        user_input: str,
        branch_labels: List[str],
        attachments: List[str],
        history_messages: Optional[list[ChatMessage]] = None,
        loaded_skill_ids: Optional[list[str]] = None,
    ):
        return self.classifier.build_skill_context(user_input, branch_labels, attachments, history_messages, loaded_skill_ids)

    def _classify_turn(self, ctx, stop_event=None):
        return self.classifier.classify(ctx, stop_event=stop_event)

    def _select_turn(self, ctx, stop_event):
        classification = self._classify_turn(ctx, stop_event=stop_event)
        selected = self._select_skills(ctx, stop_event, classification=classification)
        return classification, selected

    def _select_skills(self, ctx, stop_event, classification=None):
        return self.skill_runtime.select_skills(ctx)

    def _explicit_path_outside_workspace(self, text: str) -> str:
        return self.classifier._explicit_path_outside_workspace(text)

    def _prefers_local_workspace_tools(self, ctx, selected) -> bool:
        classification = self._classify_turn(ctx)
        return classification.prefer_local_workspace_tools

    def _call_with_retry(self, payload: JsonObject, stop_event, on_event, pass_id: str):
        return self.llm_client.call_with_retry(payload, stop_event, on_event, pass_id)

    def _build_turn_state(self, ctx, selected, history_messages, user_input):
        classification = self._classify_turn(ctx)
        return self.orchestrator.build_turn_state(ctx, selected, history_messages, classification)

    def _record_tool_effects(self, state, call, result) -> None:
        self.orchestrator.record_tool_effects(state, call, result)

    def _run_finalization_pass(self, system_content, state, stop_event, on_event, pass_id, extra_rules: str = ""):
        return self.orchestrator.finalize_turn(system_content, state, stop_event, on_event, pass_id, extra_rules)

    def _needs_fetch_evidence(self, state) -> bool:
        return self.orchestrator.needs_fetch_evidence(state)

    def _tool_call_args_for_history(self, args: dict[str, object]) -> dict[str, object]:
        return self.orchestrator.tool_call_args_for_history(args)

    def run_turn(
        self,
        history_messages: list[ChatMessage],
        user_input: str,
        thinking: bool,
        branch_labels: Optional[List[str]] = None,
        attachments: Optional[List[str]] = None,
        loaded_skill_ids: Optional[List[str]] = None,
        stop_event=None,
        on_event: Optional[Callable[[JsonObject], None]] = None,
        confirm_shell: Optional[ShellConfirmationFn] = None,
    ) -> AgentTurnResult:
        endpoint_err = self._validate_endpoints()
        if endpoint_err:
            return AgentTurnResult(status="error", content="", reasoning="", skill_exchanges=[], error=endpoint_err)
        if self.llm_client.stop_requested(stop_event):
            return AgentTurnResult(status="cancelled", content="", reasoning="", skill_exchanges=[])
        status = self.get_model_status()
        if status.state == "offline" and self.llm_client.is_model_status_fresh(status):
            if self.llm_client.should_fail_fast_on_offline_status(status):
                detail = f": {status.last_error}" if status.last_error else ""
                return AgentTurnResult(
                    status="error",
                    content="",
                    reasoning="",
                    skill_exchanges=[],
                    error=f"Model endpoint offline{detail}",
                )
            ready = self.ensure_ready(stop_event=stop_event, on_event=on_event)
            if ready is None:
                return AgentTurnResult(status="cancelled", content="", reasoning="", skill_exchanges=[])
            if not ready:
                refreshed = self.get_model_status()
                detail = f": {refreshed.last_error}" if refreshed.last_error else ""
                return AgentTurnResult(
                    status="error",
                    content="",
                    reasoning="",
                    skill_exchanges=[],
                    error=f"Model endpoint offline{detail}" if refreshed.state == "offline" else f"Model endpoint not ready: {self.models_endpoint}",
                )
        if not self.llm_client.is_model_status_fresh(status):
            if status.state != "unknown":
                status = self.refresh_model_status(timeout_s=min(self.connect_timeout_s, 1.0), force=True)
                if status.state == "online":
                    return self.orchestrator.run_turn(
                        history_messages=history_messages,
                        user_input=user_input,
                        thinking=thinking,
                        branch_labels=branch_labels,
                        attachments=attachments,
                        loaded_skill_ids=loaded_skill_ids,
                        stop_event=stop_event,
                        on_event=on_event,
                        confirm_shell=confirm_shell,
                        request_user_input=request_user_input_passthrough,
                    )
            ready = self.ensure_ready(stop_event=stop_event, on_event=on_event)
            if ready is None:
                return AgentTurnResult(status="cancelled", content="", reasoning="", skill_exchanges=[])
            if not ready:
                status = self.get_model_status()
                detail = f": {status.last_error}" if status.last_error else ""
                return AgentTurnResult(
                    status="error",
                    content="",
                    reasoning="",
                    skill_exchanges=[],
                    error=f"Model endpoint offline{detail}" if status.state == "offline" else f"Model endpoint not ready: {self.models_endpoint}",
                )
        return self.orchestrator.run_turn(
            history_messages=history_messages,
            user_input=user_input,
            thinking=thinking,
            branch_labels=branch_labels,
            attachments=attachments,
            loaded_skill_ids=loaded_skill_ids,
            stop_event=stop_event,
            on_event=on_event,
            confirm_shell=confirm_shell,
            request_user_input=request_user_input_passthrough,
        )
