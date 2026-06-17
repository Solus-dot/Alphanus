from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any, cast

from agent.classifier import TurnClassifier
from agent.config_values import coerce_int, get_json_object
from agent.evidence_guard import EvidenceGuard
from agent.finalization_engine import FinalizationEngine
from agent.llm_client import LLMClient
from agent.policies import OutputSanitizer, PromptPolicyRenderer
from agent.response_finalizer import ResponseFinalizer
from agent.runtime_hooks import TurnRuntimeHooks
from agent.telemetry import TelemetryEmitter
from agent.tool_execution_engine import ToolExecutionEngine
from agent.tool_loop_engine import ToolLoopEngine
from agent.turn_journal import TurnJournalBuilder
from agent.turn_policy_engine import TurnPolicyEngine
from core.message_types import ChatMessage, JSONValue
from core.retrieval import SQLiteRetrievalStore, configured_store_path
from core.types import (
    AgentTurnResult,
    JsonObject,
    ShellConfirmationFn,
    ToolCall,
    TurnPolicySnapshot,
    TurnState,
    UserInputRequestFn,
)
from skills.runtime import SkillRuntime
from skills.skill_parser import SkillManifest

_AUTO_MEMORY_PATTERNS = (
    re.compile(r"\b(?:i|we)\s+(?:prefer|like|use|work with)\s+([^.\n]{3,160})", re.IGNORECASE),
    re.compile(r"\bmy\s+(?:preferred|favorite|go-to)\s+([^.\n]{3,160})", re.IGNORECASE),
    re.compile(r"\b(?:this|the)\s+project\s+(?:uses|is|runs on|depends on)\s+([^.\n]{3,180})", re.IGNORECASE),
)
_AUTO_MEMORY_SECRET_RE = re.compile(r"\b(?:password|secret|token|api[_ -]?key|credential|private key)\b", re.IGNORECASE)
_TOOL_OUTCOME_SKIP = {
    "web_search",
    "fetch_url",
    "retrieve_knowledge",
    "retrieval_stats",
    "forget_retrieval_record",
    "recall_memory",
    "list_memories",
}


class TurnOrchestrator:
    def __init__(
        self,
        skill_runtime: SkillRuntime,
        context_mgr,
        llm_client: LLMClient,
        classifier: TurnClassifier,
        prompt_renderer: PromptPolicyRenderer,
        telemetry: TelemetryEmitter | None = None,
        runtime_hooks: TurnRuntimeHooks | None = None,
    ) -> None:
        self.skill_runtime = skill_runtime
        self.context_mgr = context_mgr
        self.llm_client = llm_client
        self.classifier = classifier
        self.prompt_renderer = prompt_renderer
        self.telemetry = telemetry or TelemetryEmitter()
        self._runtime_hooks = runtime_hooks
        self.reload_config(llm_client.config)

    def bind_runtime_hooks(self, runtime_hooks: TurnRuntimeHooks | None) -> None:
        self._runtime_hooks = runtime_hooks

    def call_with_retry(self, payload: JsonObject, stop_event, on_event, pass_id: str):
        hooks = self._runtime_hooks
        if hooks is not None:
            return hooks.call_with_retry(payload, stop_event, on_event, pass_id)
        return self.llm_client.call_with_retry(payload, stop_event, on_event, pass_id)

    def build_skill_context(
        self,
        user_input: str,
        branch_labels: list[str],
        attachments: list[str],
        history_messages: list[ChatMessage] | None = None,
        loaded_skill_ids: list[str] | None = None,
    ):
        hooks = self._runtime_hooks
        if hooks is not None:
            return hooks.build_skill_context(user_input, branch_labels, attachments, history_messages, loaded_skill_ids)
        return self.classifier.build_skill_context(user_input, branch_labels, attachments, history_messages, loaded_skill_ids)

    def classify_context(self, ctx, stop_event=None):
        hooks = self._runtime_hooks
        if hooks is not None:
            return hooks.classify_context(ctx, stop_event=stop_event)
        return self.classifier.classify(ctx, stop_event=stop_event)

    def select_skills(self, ctx, stop_event):
        hooks = self._runtime_hooks
        if hooks is not None:
            return hooks.select_skills(ctx, stop_event)
        classification = self.classify_context(ctx, stop_event=stop_event)
        selected = self.skill_runtime.select_skills(ctx)
        return classification, selected

    def reload_config(self, config: dict[str, Any]) -> None:
        self.config = config
        agent_cfg = get_json_object(config, "agent")
        self.max_action_depth = coerce_int(agent_cfg.get("max_action_depth", 10), 10, minimum=1)
        self.max_tool_result_chars = coerce_int(agent_cfg.get("max_tool_result_chars", 12000), 12000, minimum=0)
        self.max_reasoning_chars = coerce_int(agent_cfg.get("max_reasoning_chars", 20000), 20000, minimum=0)
        self.compact_tool_results_in_history = bool(agent_cfg.get("compact_tool_results_in_history", True))
        compact_tools = agent_cfg.get("compact_tool_result_tools", [])
        if isinstance(compact_tools, list):
            self.compact_tool_result_tools = {str(name).strip() for name in compact_tools if str(name).strip()}
        else:
            self.compact_tool_result_tools = set()
        context_budget_default = coerce_int(self.llm_client.default_max_tokens, 1024, minimum=128)
        self.context_budget_max_tokens = coerce_int(
            agent_cfg.get("context_budget_max_tokens", context_budget_default),
            context_budget_default,
            minimum=128,
        )
        self.default_tool_budgets = {"web_search": 2, "fetch_url": 2, "recall_memory": 2}
        budgets = agent_cfg.get("tool_budgets", {})
        if isinstance(budgets, dict):
            for key, value in budgets.items():
                tool_name = str(key)
                fallback = self.default_tool_budgets.get(tool_name, 1)
                self.default_tool_budgets[tool_name] = coerce_int(value, fallback, minimum=1)
        self.sanitizer = OutputSanitizer(self.max_reasoning_chars)
        self.policy_engine = TurnPolicyEngine(self.skill_runtime, self.default_tool_budgets)
        self.evidence_guard = EvidenceGuard(self.skill_runtime)
        self.tool_execution_engine = ToolExecutionEngine()
        self.tool_loop = ToolLoopEngine(self)
        self.turn_journal = TurnJournalBuilder()
        self.finalization_engine = FinalizationEngine(self)
        self.response_finalizer = ResponseFinalizer(self)

    def inject_policy_retrieval_context(self, state: TurnState, on_event: Callable[[JsonObject], None] | None = None) -> None:
        retrieval_cfg = get_json_object(self.config, "retrieval")
        if not bool(retrieval_cfg.get("enabled", True)) or not state.time_sensitive_query:
            return
        top_k = coerce_int(retrieval_cfg.get("pre_context_top_k", 3), 3, minimum=0, maximum=10)
        if top_k <= 0:
            return
        try:
            store = SQLiteRetrievalStore(configured_store_path(self.config))
            hits = store.search(state.ctx.user_input, top_k=top_k, sources=["web_page", "memory_fact", "workspace_document"])
        except Exception as exc:
            self._trace_add(state, "retrieval", {"status": "error", "error": str(exc), "query": state.ctx.user_input})
            self.emit(on_event, {"type": "info", "text": f"Retrieval pre-context unavailable: {exc}"})
            return
        state.ctx.retrieval_hits = cast(list[dict[str, JSONValue]], hits)
        self._trace_add(
            state,
            "retrieval",
            {"status": "ok", "query": state.ctx.user_input, "count": len(hits), "record_ids": [hit.get("record_id", 0) for hit in hits]},
        )
        if hits:
            self.emit(on_event, {"type": "info", "text": f"Retrieved {len(hits)} local context record(s)."})

    @staticmethod
    def _auto_memory_text(user_input: str) -> str:
        text = " ".join(str(user_input or "").split())
        if len(text) < 8 or len(text) > 600 or _AUTO_MEMORY_SECRET_RE.search(text):
            return ""
        for pattern in _AUTO_MEMORY_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            fact = match.group(0).strip(" .")
            if 8 <= len(fact) <= 220:
                return fact
        return ""

    def maybe_auto_capture_memory(self, state: TurnState, result: AgentTurnResult) -> None:
        if result.status != "done":
            return
        memory_cfg = get_json_object(self.config, "memory")
        if not bool(memory_cfg.get("auto_capture", True)):
            return
        text = self._auto_memory_text(state.ctx.user_input)
        if not text:
            return
        if any(str(item.text).strip().lower() == text.lower() for item in self.skill_runtime.memory.memories):
            return
        item = self.skill_runtime.memory.add_memory(
            text,
            memory_type="preference" if re.search(r"\b(?:prefer|preferred|favorite|go-to|like)\b", text, re.IGNORECASE) else "project",
            metadata={"source": "auto_capture", "turn_id": state.telemetry.turn_id},
            importance=0.55,
        )
        self.skill_runtime.memory.flush()
        retrieval_cfg = get_json_object(self.config, "retrieval")
        if bool(retrieval_cfg.get("enabled", True)):
            SQLiteRetrievalStore(configured_store_path(self.config)).upsert_record(
                record_type="memory_fact",
                source=f"memory:{item['id']}",
                canonical_source=f"memory:{item['id']}",
                title=str(item.get("type", "memory")),
                text=text,
                metadata={"memory_id": item["id"], "source": "auto_capture", "turn_id": state.telemetry.turn_id},
            )
        self._trace_add(state, "memory", {"status": "auto_captured", "memory_id": item["id"], "text": text})

    def maybe_index_tool_outcome(self, state: TurnState, call: ToolCall, result: dict[str, object]) -> None:
        retrieval_cfg = get_json_object(self.config, "retrieval")
        if not bool(retrieval_cfg.get("enabled", True)) or call.name in _TOOL_OUTCOME_SKIP or not result.get("ok"):
            return
        text = f"Tool {call.name} succeeded.\nArguments: {self.safe_json_dumps(call.arguments)}\nResult: {self.safe_json_dumps(result)}"
        text = self.truncate_text(text, 2400)
        if _AUTO_MEMORY_SECRET_RE.search(text):
            return
        try:
            record = SQLiteRetrievalStore(configured_store_path(self.config)).upsert_record(
                record_type="tool_outcome",
                source=f"tool:{state.telemetry.turn_id}:{len(state.evidence)}:{call.name}",
                title=f"{call.name} outcome",
                text=text,
                metadata={"tool": call.name, "turn_id": state.telemetry.turn_id},
            )
        except Exception as exc:
            self._trace_add(state, "retrieval", {"status": "tool_outcome_error", "tool": call.name, "error": str(exc)})
            return
        if record:
            self._trace_add(state, "retrieval", {"status": "tool_outcome_indexed", "tool": call.name, "record_id": record.id})

    @staticmethod
    def emit(on_event: Callable[[JsonObject], None] | None, event: JsonObject) -> None:
        if not on_event:
            return
        try:
            on_event(event)
        except Exception as exc:
            logging.debug("Event emission failed: %s", exc)
            return

    def _trace_add(self, state: TurnState, key: str, row: dict[str, object]) -> None:
        self.turn_journal.trace_add(state, key, row)

    def _is_stop_requested(self, stop_event) -> bool:
        return self.llm_client.stop_requested(stop_event)

    @staticmethod
    def safe_json_dumps(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def truncate_text(text: str, limit: int) -> str:
        if limit <= 0 or len(text) <= limit:
            return text
        clipped = text[:limit]
        remaining = len(text) - limit
        return f"{clipped}\n...[truncated {remaining} chars]"

    def compact_jsonish(self, value: object, depth: int = 0) -> object:
        if depth >= 4:
            return "[truncated]"
        if isinstance(value, str):
            return self.truncate_text(value, self.max_tool_result_chars)
        if isinstance(value, list):
            max_items = 80
            list_out: list[object] = [self.compact_jsonish(item, depth + 1) for item in value[:max_items]]
            if len(value) > max_items:
                list_out.append(f"... [{len(value) - max_items} more items truncated]")
            return list_out
        if isinstance(value, dict):
            max_keys = 120
            dict_out: dict[str, object] = {}
            items = list(value.items())
            for key, item in items[:max_keys]:
                dict_out[str(key)] = self.compact_jsonish(item, depth + 1)
            if len(items) > max_keys:
                dict_out["__truncated_keys__"] = len(items) - max_keys
            return dict_out
        return value

    def compact_tool_result(self, result: JsonObject) -> JsonObject:
        if self.max_tool_result_chars <= 0:
            return result
        compacted = self.compact_jsonish(result)
        if isinstance(compacted, dict):
            return cast(JsonObject, compacted)
        return {"value": cast(JSONValue, compacted)}

    def tool_result_for_history(self, tool_name: str, result: JsonObject) -> JsonObject:
        if not self.compact_tool_results_in_history:
            return result
        if self.compact_tool_result_tools and tool_name not in self.compact_tool_result_tools:
            return result
        return self.compact_tool_result(result)

    def tool_call_args_for_history(self, args: JsonObject) -> JsonObject:
        if not isinstance(args, dict):
            return {}
        out: JsonObject = {}
        for key, value in args.items():
            if isinstance(value, str):
                if len(value) <= 1200:
                    out[key] = value
                elif key in {"content", "old_string", "new_string"}:
                    omitted = len(value) - 1200
                    out[key] = value[:1200] + f"\n...[history excerpt; {omitted} chars omitted]"
                else:
                    out[key] = value[:1200] + f"...[truncated {len(value) - 1200} chars]"
            else:
                out[key] = cast(JSONValue, self.compact_jsonish(value))
        return out

    @staticmethod
    def _normalize_collaboration_mode(value: str) -> str:
        return "plan" if str(value or "").strip().lower() == "plan" else "execute"

    def _is_plan_mode(self, state: TurnState) -> bool:
        return self._normalize_collaboration_mode(getattr(state, "collaboration_mode", "execute")) == "plan"

    def _tool_allowed_in_plan_mode(self, tool_name: str) -> bool:
        normalized = str(tool_name or "").strip()
        if not normalized:
            return False
        reg = self.skill_runtime.tool_registration(normalized)
        if reg is None:
            return False
        capability = str(getattr(reg, "capability", "") or "").strip().lower()
        if normalized == "request_user_input" or capability == "user_input_requester":
            return True
        if normalized == "shell_command" or capability in {"run_shell_command", "workspace_execute"}:
            return False
        if capability in {"workspace_read", "workspace_tree"}:
            return True
        return not self.skill_runtime.tool_is_mutating(normalized)

    def _policy_block_tool(
        self,
        *,
        state: TurnState,
        call: ToolCall,
        pass_id: str,
        message: str,
        on_event: Callable[[JsonObject], None] | None = None,
    ) -> None:
        result = {
            "ok": False,
            "data": None,
            "error": {
                "code": "E_POLICY",
                "message": message,
            },
            "meta": {},
        }
        self.emit(on_event, {"type": "tool_result", "name": call.name, "id": call.id, "result": result})
        tool_message = {
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.name,
            "content": self.safe_json_dumps(self.tool_result_for_history(call.name, result)),
        }
        tool_chat_message = cast(ChatMessage, tool_message)
        state.dynamic_history.append(tool_chat_message)
        state.skill_exchanges.append(tool_chat_message)
        self.record_tool_effects(state, call, result, policy_blocked=True)
        self._trace_add(
            state,
            "tool_results",
            {
                "pass_id": pass_id,
                "id": call.id,
                "name": call.name,
                "result": result,
                "policy_blocked": True,
                "finished_at": time.time(),
            },
        )

    def build_turn_state(
        self,
        ctx,
        selected: list[SkillManifest],
        history_messages: list[ChatMessage],
        classification,
        *,
        collaboration_mode: str = "execute",
        context_summary: str = "",
    ) -> TurnState:
        return self.policy_engine.build_turn_state(
            ctx,
            selected,
            history_messages,
            classification,
            collaboration_mode=self._normalize_collaboration_mode(collaboration_mode),
            context_summary=context_summary,
        )

    def refresh_search_tools_enabled(self, state: TurnState) -> None:
        self.policy_engine.refresh_search_tools_enabled(state)

    @staticmethod
    def _message_contains_vision_content(message: ChatMessage) -> bool:
        content = message.get("content")
        if not isinstance(content, list):
            return False
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip().lower()
            if item_type in {"image", "image_url", "video"}:
                return True
            if "image" in item or "image_url" in item or "video" in item:
                return True
        return False

    @classmethod
    def _latest_user_message_contains_vision_content(cls, messages: list[ChatMessage]) -> bool:
        for message in reversed(messages):
            if str(message.get("role", "")).strip().lower() != "user":
                continue
            return cls._message_contains_vision_content(message)
        return False

    @staticmethod
    def _latest_user_message(messages: list[ChatMessage]) -> ChatMessage | None:
        for message in reversed(messages):
            if str(message.get("role", "")).strip().lower() == "user":
                return message
        return None

    @staticmethod
    def _leading_system_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
        kept: list[ChatMessage] = []
        for message in messages:
            if str(message.get("role", "")).strip().lower() != "system":
                break
            kept.append(message)
        return kept

    def _retry_simplified_vision_payload(
        self,
        *,
        model_messages: list[ChatMessage],
        thinking: bool,
        stop_event=None,
        on_event: Callable[[JsonObject], None] | None = None,
        pass_id: str,
    ):
        latest_user = self._latest_user_message(model_messages)
        if latest_user is None or not self._message_contains_vision_content(latest_user):
            return None
        simplified_messages = self._leading_system_messages(model_messages) + [latest_user]
        payload = self.llm_client.build_payload(simplified_messages, thinking=thinking, tools=None)
        self.emit(on_event, {"type": "info", "text": "Retrying image request with simplified multimodal payload..."})
        return self.call_with_retry(payload, stop_event, on_event, pass_id=f"{pass_id}_vision_retry")

    @classmethod
    def _friendly_vision_request_error(cls, messages: list[ChatMessage], exc: Exception) -> str:
        if not cls._latest_user_message_contains_vision_content(messages):
            return str(exc)
        raw = str(exc or "").strip()
        lowered = raw.lower()
        if "failed to tokenize prompt" in lowered:
            return (
                "The current model endpoint rejected this image attachment while tokenizing the prompt. "
                "Use a vision-capable model/template for image inputs, or remove the image attachment."
            )
        if "no user query found in messages" in lowered:
            return (
                "The current model endpoint rejected this image attachment because its chat template could not "
                "render the multimodal prompt. Use a vision-capable model/template for image inputs, or remove "
                "the image attachment."
            )
        if "image input is not supported" in lowered or "mmproj" in lowered:
            return (
                "The current model endpoint does not support image inputs. If you are using llama.cpp, start the "
                "server with a vision-capable model and matching --mmproj file. Otherwise remove the image "
                "attachment or switch to a vision-capable endpoint."
            )
        return raw

    def workspace_materialization_count(self, state: TurnState) -> int:
        return self.evidence_guard.workspace_materialization_count(state)

    def workspace_mutation_count(self, state: TurnState) -> int:
        return self.evidence_guard.workspace_mutation_count(state)

    def workspace_readback_count(self, state: TurnState) -> int:
        return self.evidence_guard.workspace_readback_count(state)

    @staticmethod
    def tool_result_paths(name: str, payload: dict[str, object]) -> list[str]:
        return ToolExecutionEngine.tool_result_paths(name, payload)

    def tool_budget_reason(self, state: TurnState, call: ToolCall) -> str:
        return self.policy_engine.tool_budget_reason(state, call) or ""

    def record_tool_effects(self, state: TurnState, call: ToolCall, result: dict[str, object], *, policy_blocked: bool = False) -> None:
        self.tool_execution_engine.record_tool_effects(state, call, result, policy_blocked=policy_blocked)
        if not policy_blocked:
            self.maybe_index_tool_outcome(state, call, result)

    def needs_fetch_evidence(self, state: TurnState) -> bool:
        return self.evidence_guard.needs_fetch_evidence(state)

    def workspace_action_evidence(self, state: TurnState) -> JsonObject:
        return self.evidence_guard.workspace_action_evidence(state)

    def workspace_action_outcome(self, state: TurnState, text: str, *, stop_event, pass_id: str) -> str:
        if self.workspace_mutation_count(state) > 0:
            return "completed_with_evidence"
        cleaned = self.sanitizer.sanitize_final_content(text)
        return self.classifier.classify_workspace_action_outcome(
            current_user_input=state.ctx.user_input,
            recent_routing_hint=getattr(state.ctx, "recent_routing_hint", ""),
            assistant_reply=cleaned,
            evidence=self.workspace_action_evidence(state),
            pass_id=pass_id,
            stop_event=stop_event,
        )

    def coerce_workspace_action_failure(self, state: TurnState, result: AgentTurnResult, *, stop_event, pass_id: str) -> AgentTurnResult:
        if self._is_plan_mode(state):
            return result
        if result.status != "done" or not state.requires_workspace_action or self.workspace_mutation_count(state) > 0:
            return result
        outcome = self.workspace_action_outcome(state, result.content, stop_event=stop_event, pass_id=pass_id)
        if outcome in {"completed_with_evidence", "declined_or_blocked", "needs_clarification"}:
            return result
        return AgentTurnResult(
            status="error",
            content=(
                "[agent error] Workspace action was not completed: no successful mutating workspace tool ran. "
                "The assistant draft was rejected."
            ),
            reasoning=result.reasoning,
            skill_exchanges=result.skill_exchanges,
            error="workspace_action_not_completed",
            journal=result.journal,
        )

    def build_turn_journal(self, state: TurnState, result: AgentTurnResult) -> JsonObject:
        return self.turn_journal.build(
            state,
            result,
            collaboration_mode=self._normalize_collaboration_mode(getattr(state, "collaboration_mode", "execute")),
        )

    def log_turn_summary(self, state: TurnState, result: AgentTurnResult) -> None:
        self.telemetry.emit(
            "turn_summary",
            status=result.status,
            error=result.error or "",
            turn_id=state.telemetry.turn_id,
            selected_skills=[getattr(skill, "id", "") for skill in state.selected],
            tool_counts=state.completion.tool_counts,
            evidence_count=len(state.evidence),
            search_mode=state.search_mode,
            search_failures=state.completion.search_failure_count,
            fetched_urls=len(state.completion.fetched_urls),
            blocked_domains=sorted(state.completion.blocked_fetch_domains),
            collaboration_mode=self._normalize_collaboration_mode(getattr(state, "collaboration_mode", "execute")),
            content_chars=len(result.content),
            reasoning_chars=len(result.reasoning),
            finalization_attempts=state.telemetry.finalization_attempts,
            finalization_repairs=state.telemetry.finalization_repairs,
            finalization_repair_failed=state.telemetry.finalization_repair_failed,
        )

    def build_policy_snapshot(self, state: TurnState) -> TurnPolicySnapshot:
        return self.policy_engine.build_policy_snapshot(state)

    def finalize_turn(
        self, system_content: str, state: TurnState, stop_event, on_event, pass_id: str, extra_rules: str = ""
    ) -> AgentTurnResult:
        return self.finalization_engine.finalize_turn(
            system_content=system_content,
            state=state,
            stop_event=stop_event,
            on_event=on_event,
            pass_id=pass_id,
            extra_rules=extra_rules,
        )

    def prepare_turn(
        self,
        history_messages: list[ChatMessage],
        user_input: str,
        *,
        branch_labels: list[str] | None = None,
        attachments: list[str] | None = None,
        loaded_skill_ids: list[str] | None = None,
        context_summary: str = "",
        collaboration_mode: str = "execute",
        stop_event=None,
    ) -> TurnState:
        branch_labels = branch_labels or []
        attachments = attachments or []
        ctx = self.build_skill_context(user_input, branch_labels, attachments, history_messages, loaded_skill_ids or [])
        if ctx is None:
            ctx = self.classifier.build_skill_context(
                user_input,
                branch_labels,
                attachments,
                history_messages,
                loaded_skill_ids or [],
            )
        selection = self.select_skills(ctx, stop_event)
        if (
            not isinstance(selection, tuple)
            or len(selection) != 2
            or not isinstance(selection[1], list)
        ):
            classification = self.classify_context(ctx, stop_event=stop_event)
            selected = self.skill_runtime.select_skills(ctx)
        else:
            classification, selected = selection
        ctx.context_summary = str(context_summary or "").strip()
        ctx.relevant_skill_ids = [getattr(skill, "id", "") for skill in selected if getattr(skill, "id", "")]
        return self.build_turn_state(
            ctx,
            selected,
            history_messages,
            classification,
            collaboration_mode=collaboration_mode,
            context_summary=ctx.context_summary,
        )

    def _context_budget_report(
        self,
        *,
        system_content: str,
        policy_rules: str,
        retrieval_hits: int,
        skill_count: int,
        messages_before: list[ChatMessage],
        messages_after: list[ChatMessage],
        tools: list[dict[str, Any]],
        summary_status: str,
        output_reserve_tokens: int,
    ) -> JsonObject:
        tool_schema_tokens = self.context_mgr.estimate_json_tokens(tools)
        system_tokens = self.context_mgr.estimate_text_tokens(system_content)
        history_before_tokens = self.context_mgr.estimate_tokens(messages_before[1:]) if len(messages_before) > 1 else 0
        history_after_tokens = self.context_mgr.estimate_tokens(messages_after[1:]) if len(messages_after) > 1 else 0
        final_prompt_tokens = self.context_mgr.estimate_tokens(messages_after) + tool_schema_tokens
        budget = max(1, self.context_mgr.context_limit - self.context_mgr.safety_margin)
        return {
            "context_limit": self.context_mgr.context_limit,
            "safety_margin": self.context_mgr.safety_margin,
            "budget_tokens": budget,
            "output_reserve_tokens": output_reserve_tokens,
            "tool_schema_tokens": tool_schema_tokens,
            "system_tokens": system_tokens,
            "policy_tokens": self.context_mgr.estimate_text_tokens(policy_rules),
            "history_before_tokens": history_before_tokens,
            "history_after_tokens": history_after_tokens,
            "final_prompt_tokens_estimate": final_prompt_tokens,
            "messages_before": len(messages_before),
            "messages_after": len(messages_after),
            "tool_count": len(tools),
            "retrieval_records": retrieval_hits,
            "skill_count": skill_count,
            "summary_status": summary_status,
            "pruned": len(messages_after) < len(messages_before) or history_after_tokens < history_before_tokens,
            "over_budget": final_prompt_tokens + output_reserve_tokens > budget,
        }

    def _summary_needed(self, system_messages: list[ChatMessage], tools: list[dict[str, Any]]) -> bool:
        budget = max(1, self.context_mgr.context_limit - self.context_mgr.safety_margin)
        tool_schema_tokens = self.context_mgr.estimate_json_tokens(tools)
        return self.context_mgr.estimate_tokens(system_messages) + tool_schema_tokens + self.context_budget_max_tokens > budget

    def _summarize_history_with_model(self, prior_summary: str, messages: list[ChatMessage], stop_event) -> str:
        if not messages or self._is_stop_requested(stop_event):
            return ""
        system = (
            "Summarize earlier conversation context for a coding/desktop agent. "
            "Preserve user goals, decisions, files touched, tool outcomes, loaded skills, unresolved errors, and next steps. "
            "Be concise and factual. Do not invent details."
        )
        lines = []
        prior = str(prior_summary or "").strip()
        if prior:
            lines.append("Existing summary:\n" + prior)
        lines.append("Messages to summarize:")
        for message in messages:
            lines.append(self.context_mgr._message_label(message))
        payload = self.llm_client.build_payload(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": "\n".join(lines)},
            ],
            thinking=False,
            tools=None,
        )
        result = self.call_with_retry(payload, stop_event, None, pass_id="context_summary")
        if result is None:
            return ""
        if result.finish_reason == "cancelled":
            return ""
        text = (result.content or "").strip()
        return self.truncate_text(text, 2400)

    def _maybe_summarize_history(self, state: TurnState, system_messages: list[ChatMessage], tools: list[dict[str, Any]], stop_event) -> str:
        if not self._summary_needed(system_messages + state.dynamic_history, tools):
            return "not_needed"
        summarize, retained = self.context_mgr.split_for_summary(state.dynamic_history)
        if not summarize:
            return "not_possible"
        previous_summary = str(state.context_summary or "").strip()
        try:
            summary = self._summarize_history_with_model(previous_summary, summarize, stop_event)
            status = "model"
        except Exception as exc:
            logging.debug("Context model summary failed: %s", exc)
            summary = ""
            status = "fallback"
        if not summary:
            summary = self.context_mgr.deterministic_summary(previous_summary, summarize)
            status = "fallback"
        state.context_summary = summary
        state.ctx.context_summary = summary
        state.dynamic_history = retained
        return status

    def run_model_pass(
        self,
        state: TurnState,
        thinking: bool,
        *,
        stop_event=None,
        on_event: Callable[[JsonObject], None] | None = None,
    ) -> AgentTurnResult | tuple[str, str, Any]:
        self.refresh_search_tools_enabled(state)
        state.pass_index += 1
        state.telemetry.pass_index = state.pass_index
        pass_id = f"pass_{state.pass_index}"

        if stop_event is not None and stop_event.is_set():
            return AgentTurnResult(
                status="cancelled",
                content="",
                reasoning=state.full_reasoning,
                skill_exchanges=state.skill_exchanges,
            )

        tools = self.skill_runtime.tools_for_turn(state.selected, ctx=state.ctx)
        if self._normalize_collaboration_mode(getattr(state, "collaboration_mode", "execute")) == "plan":
            tools = [
                item
                for item in tools
                if isinstance(item, dict)
                and isinstance(item.get("function"), dict)
                and self._tool_allowed_in_plan_mode(str(item["function"].get("name", "")).strip())
            ]
        policy_snapshot = self.build_policy_snapshot(state)
        system_content = self.prompt_renderer.compose_system_content(state.selected, state.ctx)
        policy_rules = self.prompt_renderer.render_policy_rules(policy_snapshot)
        if policy_rules:
            system_content += "\n\n" + policy_rules
        system_messages: list[ChatMessage] = [cast(ChatMessage, {"role": "system", "content": system_content})]
        summary_status = self._maybe_summarize_history(state, system_messages, tools, stop_event)
        if summary_status in {"model", "fallback"}:
            policy_snapshot = self.build_policy_snapshot(state)
            system_content = self.prompt_renderer.compose_system_content(state.selected, state.ctx)
            policy_rules = self.prompt_renderer.render_policy_rules(policy_snapshot)
            if policy_rules:
                system_content += "\n\n" + policy_rules
            system_messages = [cast(ChatMessage, {"role": "system", "content": system_content})]
        messages_before = system_messages + state.dynamic_history
        tool_schema_tokens = self.context_mgr.estimate_json_tokens(tools)
        model_messages = self.context_mgr.prune(messages_before, self.context_budget_max_tokens + tool_schema_tokens)
        if (
            tools
            and self._latest_user_message_contains_vision_content(model_messages)
            and not self.skill_runtime.core_tool_names_for_turn(state.selected, ctx=state.ctx)
            and not self.skill_runtime.optional_tool_names(state.selected, ctx=state.ctx)
        ):
            tools = None
        report_tools = tools or []
        state.context_report = self._context_budget_report(
            system_content=system_content,
            policy_rules=policy_rules,
            retrieval_hits=len(getattr(state.ctx, "retrieval_hits", []) or []),
            skill_count=len(state.selected),
            messages_before=messages_before,
            messages_after=model_messages,
            tools=report_tools,
            summary_status=summary_status,
            output_reserve_tokens=self.context_budget_max_tokens,
        )
        payload = self.llm_client.build_payload(model_messages, thinking=thinking, tools=tools or None)
        pass_trace: dict[str, object] = {
            "pass_id": pass_id,
            "started_at": time.time(),
            "collaboration_mode": self._normalize_collaboration_mode(getattr(state, "collaboration_mode", "execute")),
            "selected_skills": [getattr(skill, "id", "") for skill in state.selected],
            "tool_names": [
                str(fn.get("name", "")).strip()
                for item in (tools or [])
                if isinstance(item, dict)
                for fn in [item.get("function")]
                if isinstance(fn, dict)
            ],
            "system_prompt": system_content,
            "payload": payload,
        }
        self._trace_add(state, "passes", pass_trace)

        try:
            stream_result = self.call_with_retry(payload, stop_event, on_event, pass_id=pass_id)
        except Exception as exc:
            if self._latest_user_message_contains_vision_content(model_messages) and (
                "failed to tokenize prompt" in str(exc or "").strip().lower()
            ):
                try:
                    stream_result = self._retry_simplified_vision_payload(
                        model_messages=model_messages,
                        thinking=thinking,
                        stop_event=stop_event,
                        on_event=on_event,
                        pass_id=pass_id,
                    )
                except Exception as retry_exc:
                    message = self._friendly_vision_request_error(model_messages, retry_exc)
                    self.emit(on_event, {"type": "error", "text": message})
                    return AgentTurnResult(
                        status="error",
                        content="",
                        reasoning=state.full_reasoning,
                        skill_exchanges=state.skill_exchanges,
                        error=message,
                    )
                if stream_result is None:
                    message = self._friendly_vision_request_error(model_messages, exc)
                    self.emit(on_event, {"type": "error", "text": message})
                    return AgentTurnResult(
                        status="error",
                        content="",
                        reasoning=state.full_reasoning,
                        skill_exchanges=state.skill_exchanges,
                        error=message,
                    )
            else:
                message = self._friendly_vision_request_error(model_messages, exc)
                self.emit(on_event, {"type": "error", "text": message})
                return AgentTurnResult(
                    status="error",
                    content="",
                    reasoning=state.full_reasoning,
                    skill_exchanges=state.skill_exchanges,
                    error=message,
                )

        if stream_result is None:
            return AgentTurnResult(
                status="cancelled",
                content="",
                reasoning=state.full_reasoning,
                skill_exchanges=state.skill_exchanges,
            )

        pass_trace["completed_at"] = time.time()
        completed_at_raw = pass_trace.get("completed_at")
        started_at_raw = pass_trace.get("started_at")
        completed_at = float(completed_at_raw) if isinstance(completed_at_raw, (int, float)) else time.time()
        started_at = float(started_at_raw) if isinstance(started_at_raw, (int, float)) else completed_at
        pass_trace["duration_ms"] = max(0, int((completed_at - started_at) * 1000))
        pass_trace["finish_reason"] = stream_result.finish_reason
        pass_trace["usage"] = dict(getattr(stream_result, "usage", {}) or {})
        pass_trace["first_token_latency_ms"] = getattr(stream_result, "first_token_latency_ms", None)

        if stream_result.finish_reason == "cancelled":
            return AgentTurnResult(
                status="cancelled",
                content=stream_result.content,
                reasoning=self.sanitizer.append_reasoning(state.full_reasoning, stream_result.reasoning),
                skill_exchanges=state.skill_exchanges,
            )

        state.full_reasoning = self.sanitizer.append_reasoning(state.full_reasoning, stream_result.reasoning)
        stream_usage = getattr(stream_result, "usage", {}) or {}
        if isinstance(stream_usage, dict) and stream_usage:
            state.telemetry.model_usage = dict(stream_usage)
        self.emit(
            on_event,
            {
                "type": "pass_end",
                "pass_id": pass_id,
                "finish_reason": stream_result.finish_reason,
                "has_content": bool(stream_result.content.strip()),
                "has_tool_calls": bool(stream_result.tool_calls),
            },
        )
        return pass_id, system_content, stream_result

    def execute_tool_calls(
        self,
        *,
        system_content: str,
        state: TurnState,
        pass_id: str,
        stream_result,
        stop_event=None,
        on_event: Callable[[JsonObject], None] | None = None,
        confirm_shell: ShellConfirmationFn | None = None,
        request_user_input: UserInputRequestFn | None = None,
    ) -> tuple[str, AgentTurnResult | None]:
        return self.tool_loop.execute_tool_calls(
            system_content=system_content,
            state=state,
            pass_id=pass_id,
            stream_result=stream_result,
            stop_event=stop_event,
            on_event=on_event,
            confirm_shell=confirm_shell,
            request_user_input=request_user_input,
        )

    def finalize_response(
        self,
        *,
        system_content: str,
        state: TurnState,
        pass_id: str,
        stream_result,
        stop_event=None,
        on_event: Callable[[JsonObject], None] | None = None,
    ) -> tuple[str, AgentTurnResult | None]:
        return self.response_finalizer.finalize_response(
            system_content=system_content,
            state=state,
            pass_id=pass_id,
            stream_result=stream_result,
            stop_event=stop_event,
            on_event=on_event,
        )

    def run_turn(
        self,
        history_messages: list[ChatMessage],
        user_input: str,
        thinking: bool,
        *,
        branch_labels: list[str] | None = None,
        attachments: list[str] | None = None,
        loaded_skill_ids: list[str] | None = None,
        context_summary: str = "",
        collaboration_mode: str = "execute",
        stop_event=None,
        on_event: Callable[[JsonObject], None] | None = None,
        confirm_shell: ShellConfirmationFn | None = None,
        request_user_input: UserInputRequestFn | None = None,
    ) -> AgentTurnResult:
        state = self.prepare_turn(
            history_messages,
            user_input,
            branch_labels=branch_labels,
            attachments=attachments,
            loaded_skill_ids=loaded_skill_ids,
            context_summary=context_summary,
            collaboration_mode=collaboration_mode,
            stop_event=stop_event,
        )
        self.inject_policy_retrieval_context(state, on_event=on_event)

        def finish(result: AgentTurnResult) -> AgentTurnResult:
            self.maybe_auto_capture_memory(state, result)
            result.journal = self.build_turn_journal(state, result)
            self.log_turn_summary(state, result)
            return result

        def finish_finalized(result: AgentTurnResult) -> AgentTurnResult:
            return finish(result)

        while True:
            if self._is_stop_requested(stop_event):
                return finish(
                    AgentTurnResult(
                        status="cancelled",
                        content="",
                        reasoning=state.full_reasoning,
                        skill_exchanges=state.skill_exchanges,
                    )
                )
            model_phase = self.run_model_pass(state, thinking, stop_event=stop_event, on_event=on_event)
            if isinstance(model_phase, AgentTurnResult):
                return finish(model_phase)

            pass_id, system_content, stream_result = model_phase

            if stream_result.finish_reason == "tool_calls":
                action, tool_phase_result = self.execute_tool_calls(
                    system_content=system_content,
                    state=state,
                    pass_id=pass_id,
                    stream_result=stream_result,
                    stop_event=stop_event,
                    on_event=on_event,
                    confirm_shell=confirm_shell,
                    request_user_input=request_user_input,
                )
                if action == "continue":
                    continue
                if tool_phase_result is None:
                    continue
                if action == "finalized":
                    return finish_finalized(tool_phase_result)
                return finish(tool_phase_result)

            action, final_phase_result = self.finalize_response(
                system_content=system_content,
                state=state,
                pass_id=pass_id,
                stream_result=stream_result,
                stop_event=stop_event,
                on_event=on_event,
            )
            if action == "continue":
                continue
            if final_phase_result is None:
                continue
            if action == "finalized":
                return finish_finalized(final_phase_result)
            return finish(final_phase_result)


def request_user_input_passthrough(args: JsonObject) -> JsonObject:
    question = str(args.get("question", "")).strip()
    if not question:
        raise ValueError("Missing required argument: question")
    options = args.get("options")
    normalized_options = [str(item).strip() for item in options if str(item).strip()] if isinstance(options, list) else []
    return {
        "question": question,
        "options": cast(JSONValue, normalized_options),
        "header": str(args.get("header", "")).strip(),
        "awaiting_user_input": True,
    }
