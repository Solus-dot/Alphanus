from __future__ import annotations

import copy
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.classifier import TurnClassifier
from agent.context import ContextWindowManager
from agent.llm_client import LLMClient
from agent.orchestrator import TurnOrchestrator, new_stop_event, request_user_input_passthrough
from agent.policies import PromptPolicyRenderer
from agent.prompts import build_system_prompt
from agent.telemetry import TelemetryEmitter
from agent.types import AgentTurnResult, BackgroundSkillAgentTask
from core.configuration import validate_endpoint_policy
from core.skills import SkillRuntime


class Agent:
    def __init__(self, config: Dict[str, Any], skill_runtime: SkillRuntime, debug: bool = False) -> None:
        self.skill_runtime = skill_runtime
        self.debug = debug
        self.telemetry = TelemetryEmitter()
        self._bg_skill_agent_tasks: Dict[str, BackgroundSkillAgentTask] = {}
        self._bg_skill_agent_lock = threading.Lock()
        self.system_prompt = build_system_prompt(self.skill_runtime.workspace.workspace_root)
        self.context_mgr = ContextWindowManager()
        self.llm_client = LLMClient(config, debug=debug, telemetry=self.telemetry)
        self.classifier = TurnClassifier(config, skill_runtime, self.llm_client, telemetry=self.telemetry)
        self.prompt_renderer = PromptPolicyRenderer(self.system_prompt, self.skill_runtime)
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

    def reload_config(self, config: Dict[str, Any]) -> None:
        self.config = config
        context_cfg = config.get("context", {}) if isinstance(config.get("context"), dict) else {}
        self.context_mgr = ContextWindowManager(
            context_limit=int(context_cfg.get("context_limit", 8192)),
            keep_last_n=int(context_cfg.get("keep_last_n", 10)),
            safety_margin=int(context_cfg.get("safety_margin", 500)),
        )
        self.system_prompt = build_system_prompt(self.skill_runtime.workspace.workspace_root)
        self.llm_client.reload_config(config)
        self.classifier.reload_config(config)
        self.prompt_renderer = PromptPolicyRenderer(self.system_prompt, self.skill_runtime)
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
        self.orchestrator.build_skill_context = lambda user_input, branch_labels, attachments, history_messages: self._build_skill_context(
            user_input,
            branch_labels,
            attachments,
            history_messages,
        )
        self.orchestrator.classify_context = lambda ctx, stop_event=None: self._classify_turn(ctx, stop_event)
        self.orchestrator.select_skills = lambda ctx, stop_event: self._select_turn(ctx, stop_event)
        self.model_endpoint = self.llm_client.model_endpoint
        self.models_endpoint = self.llm_client.models_endpoint
        self.allow_cross_host = self.llm_client.allow_cross_host
        self.readiness_timeout_s = self.llm_client.readiness_timeout_s

    def ensure_ready(self, stop_event=None, on_event: Optional[Callable[[Dict[str, Any]], None]] = None, timeout_s: Optional[float] = None) -> Optional[bool]:
        return self.llm_client.ensure_ready(stop_event=stop_event, on_event=on_event, timeout_s=timeout_s)

    def fetch_model_metadata(self, timeout_s: Optional[float] = None) -> tuple[Optional[str], Optional[int]]:
        return self.llm_client.fetch_model_metadata(timeout_s=timeout_s)

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

    def doctor_report(self) -> Dict[str, Any]:
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
                "backend": memory_stats.get("embedding_backend"),
                "configured_backend": memory_stats.get("configured_embedding_backend"),
                "allow_model_download": memory_stats.get("allow_model_download"),
                "encoder_status": memory_stats.get("encoder_status"),
                "encoder_source": memory_stats.get("encoder_source"),
                "encoder_detail": memory_stats.get("encoder_detail"),
                "mode": memory_stats.get("mode_label"),
                "model_name": memory_stats.get("model_name"),
                "recommended_model_name": memory_stats.get("recommended_model_name"),
                "dimension": memory_stats.get("dimension"),
                "count": memory_stats.get("count"),
            },
            "search": {
                "provider": provider,
                "ready": search_ready,
                "reason": "" if search_ready or not required_env else f"missing env: {required_env}",
            },
            "skills": self.skill_runtime.skill_health_report(),
        }

    def build_support_bundle(self, tree_payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "doctor": self.doctor_report(),
            "tree": tree_payload,
        }

    def reload_skills(self) -> int:
        return self.classifier.reload_skills()

    @staticmethod
    def _extract_model_name(payload: Any) -> Optional[str]:
        return LLMClient.extract_model_name(payload)

    @staticmethod
    def _extract_model_context_window(payload: Any) -> Optional[int]:
        return LLMClient.extract_model_context_window(payload)

    def _build_skill_context(
        self,
        user_input: str,
        branch_labels: List[str],
        attachments: List[str],
        history_messages: Optional[List[Dict[str, Any]]] = None,
    ):
        return self.classifier.build_skill_context(user_input, branch_labels, attachments, history_messages)

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

    def _call_with_retry(self, payload: Dict[str, Any], stop_event, on_event, pass_id: str):
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

    def _tool_call_args_for_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.orchestrator.tool_call_args_for_history(args)

    def _run_nested_skill_agent(self, agent_name: str, prompt: str, skill_id: str = "") -> AgentTurnResult:
        try:
            agent_contract = self.skill_runtime.load_agent_contract(agent_name, skill_id=skill_id)
        except Exception as exc:
            return AgentTurnResult(status="error", content="", reasoning="", skill_exchanges=[], error=str(exc))
        agent_record = self.skill_runtime.get_agent(agent_name)
        if agent_record is None:
            return AgentTurnResult(
                status="error",
                content="",
                reasoning="",
                skill_exchanges=[],
                error=f"Unknown companion agent: {agent_name}",
            )
        nested_config = copy.deepcopy(self.config) if isinstance(self.config, dict) else {}
        nested_agents_cfg = nested_config.get("agents", {}) if isinstance(nested_config.get("agents"), dict) else {}
        nested_agents_cfg["enable_skill_agents"] = False
        nested_config["agents"] = nested_agents_cfg
        nested = Agent(nested_config, self.skill_runtime, debug=self.debug)
        nested.system_prompt = (
            f"{self.system_prompt}\n\n"
            f"Companion agent profile: {agent_record.name}\n"
            f"{agent_record.description}\n\n"
            f"{agent_contract.prompt}"
        ).strip()
        nested.prompt_renderer = PromptPolicyRenderer(nested.system_prompt, nested.skill_runtime)
        nested.orchestrator = TurnOrchestrator(
            skill_runtime=nested.skill_runtime,
            context_mgr=nested.context_mgr,
            llm_client=nested.llm_client,
            classifier=nested.classifier,
            prompt_renderer=nested.prompt_renderer,
            telemetry=nested.telemetry,
        )
        return nested.run_turn(
            history_messages=[],
            user_input=prompt,
            thinking=False,
            branch_labels=[],
            attachments=[],
            stop_event=new_stop_event(),
            on_event=None,
            confirm_shell=None,
        )

    def _background_skill_agent_worker(self, task_id: str, agent_name: str, prompt: str, skill_id: str) -> None:
        result = self._run_nested_skill_agent(agent_name, prompt, skill_id=skill_id)
        with self._bg_skill_agent_lock:
            task = self._bg_skill_agent_tasks.get(task_id)
            if not task:
                return
            task.completed_at = time.time()
            if result.status == "done":
                task.status = "done"
                task.output = result.content
            elif result.status == "cancelled":
                task.status = "cancelled"
                task.error = "cancelled"
            else:
                task.status = "error"
                task.error = result.error or "background agent failed"

    def _handle_spawn_skill_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        agents_cfg = self.config.get("agents", {}) if isinstance(self.config.get("agents"), dict) else {}
        if not bool(agents_cfg.get("enable_skill_agents", True)):
            raise PermissionError("Skill agents are disabled in configuration")
        action = str(args.get("action", "start")).strip().lower() or "start"
        if action in {"status", "wait"}:
            task_id = str(args.get("task_id", "")).strip()
            if not task_id:
                raise ValueError("Missing required argument: task_id")
            deadline = time.time() + max(1, int(args.get("timeout_s", 30))) if action == "wait" else time.time()
            while True:
                with self._bg_skill_agent_lock:
                    task = self._bg_skill_agent_tasks.get(task_id)
                    snapshot = None if task is None else {
                        "task_id": task.task_id,
                        "agent_name": task.agent_name,
                        "skill_id": task.skill_id,
                        "status": task.status,
                        "output": task.output,
                        "error": task.error,
                    }
                if snapshot is None:
                    raise FileNotFoundError(f"Unknown skill agent task: {task_id}")
                if action == "status" or snapshot["status"] != "running" or time.time() >= deadline:
                    return snapshot
                time.sleep(0.05)
        agent_name = str(args.get("agent_name", "")).strip()
        skill_id = str(args.get("skill_id", "")).strip()
        prompt = str(args.get("prompt", "")).strip()
        if not agent_name:
            raise ValueError("Missing required argument: agent_name")
        if not prompt:
            raise ValueError("Missing required argument: prompt")
        background = bool(args.get("background", True))
        if not background:
            result = self._run_nested_skill_agent(agent_name, prompt, skill_id=skill_id)
            return {
                "task_id": "",
                "agent_name": agent_name,
                "skill_id": skill_id,
                "status": result.status,
                "output": result.content,
                "error": result.error or "",
            }
        task_id = f"skill-agent-{uuid.uuid4().hex[:10]}"
        with self._bg_skill_agent_lock:
            self._bg_skill_agent_tasks[task_id] = BackgroundSkillAgentTask(
                task_id=task_id,
                agent_name=agent_name,
                skill_id=skill_id,
                prompt=prompt,
            )
        thread = threading.Thread(target=self._background_skill_agent_worker, args=(task_id, agent_name, prompt, skill_id), daemon=True)
        thread.start()
        return {
            "task_id": task_id,
            "agent_name": agent_name,
            "skill_id": skill_id,
            "status": "running",
            "output": "",
            "error": "",
        }

    def run_turn(
        self,
        history_messages: List[Dict[str, Any]],
        user_input: str,
        thinking: bool,
        branch_labels: Optional[List[str]] = None,
        attachments: Optional[List[str]] = None,
        stop_event=None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        confirm_shell: Optional[Callable[[str], bool]] = None,
    ) -> AgentTurnResult:
        endpoint_err = self._validate_endpoints()
        if endpoint_err:
            return AgentTurnResult(status="error", content="", reasoning="", skill_exchanges=[], error=endpoint_err)
        if self.llm_client.stop_requested(stop_event):
            return AgentTurnResult(status="cancelled", content="", reasoning="", skill_exchanges=[])
        if not self.llm_client._ready_checked:
            ready = self.ensure_ready(stop_event=stop_event, on_event=on_event)
            if ready is None:
                return AgentTurnResult(status="cancelled", content="", reasoning="", skill_exchanges=[])
            if not ready:
                return AgentTurnResult(status="error", content="", reasoning="", skill_exchanges=[], error=f"Model endpoint not ready: {self.models_endpoint}")
        return self.orchestrator.run_turn(
            history_messages=history_messages,
            user_input=user_input,
            thinking=thinking,
            branch_labels=branch_labels,
            attachments=attachments,
            stop_event=stop_event,
            on_event=on_event,
            confirm_shell=confirm_shell,
            spawn_skill_agent=self._handle_spawn_skill_agent,
            request_user_input=request_user_input_passthrough,
        )
