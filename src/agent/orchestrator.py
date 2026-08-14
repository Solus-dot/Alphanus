from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any, cast

from agent import tool_execution_engine, turn_journal
from agent.classifier import TurnClassifier
from agent.evidence_guard import EvidenceGuard
from agent.policies import OutputSanitizer, PromptPolicyRenderer
from agent.provider import LLMClient
from agent.telemetry import TelemetryEmitter
from agent.tool_history import ToolHistoryCompactor
from agent.tool_loop_engine import ToolLoopEngine
from agent.turn_policy_engine import TurnPolicyEngine
from core.config_model import ConfigSchema, config_schema
from core.message_types import ChatMessage, JSONValue
from core.retrieval import SQLiteRetrievalStore, configured_store_path
from core.types import (
    AgentTurnResult,
    ApprovalRequestFn,
    JsonObject,
    ToolCall,
    TurnState,
    UserInputRequestFn,
    cancelled_turn_result,
)
from skills.runtime import SkillRuntime


class TurnOrchestrator:
    def __init__(
        self,
        skill_runtime: SkillRuntime,
        context_mgr,
        llm_client: LLMClient,
        classifier: TurnClassifier,
        prompt_renderer: PromptPolicyRenderer,
        telemetry: TelemetryEmitter | None = None,
    ) -> None:
        self.skill_runtime = skill_runtime
        self.context_mgr = context_mgr
        self.llm_client = llm_client
        self.classifier = classifier
        self.prompt_renderer = prompt_renderer
        self.telemetry = telemetry or TelemetryEmitter()
        self.reload_config(llm_client.config)

    def reload_config(self, config: ConfigSchema | Mapping[str, Any]) -> None:
        self.config = config_schema(config)
        agent_cfg = self.config.agent
        self.max_iterations = agent_cfg.max_iterations
        self.turn_timeout_s = agent_cfg.turn_timeout_s
        self.max_reasoning_chars = agent_cfg.max_reasoning_chars
        self.history = ToolHistoryCompactor(agent_cfg)
        self.context_budget_max_tokens = agent_cfg.context_budget_max_tokens
        self.default_tool_budgets = {"web_search": 2, "fetch_url": 2, "recall_memory": 2}
        for key, value in (agent_cfg.tool_budgets or {}).items():
            self.default_tool_budgets[str(key)] = int(value)
        self.sanitizer = OutputSanitizer(self.max_reasoning_chars)
        self.policy_engine = TurnPolicyEngine(self.skill_runtime, self.default_tool_budgets)
        self.evidence_guard = EvidenceGuard(self.skill_runtime, agent_cfg.recent_tool_detail_limit)
        self.tool_execution_engine = tool_execution_engine
        self.tool_loop = ToolLoopEngine(self)
        self.turn_journal = turn_journal

    def inject_policy_retrieval_context(self, state: TurnState, on_event: Callable[[JsonObject], None] | None = None) -> None:
        retrieval_cfg = self.config.retrieval
        if not retrieval_cfg.enabled or not state.classification.time_sensitive:
            return
        top_k = retrieval_cfg.pre_context_top_k
        if top_k <= 0:
            return
        try:
            store = SQLiteRetrievalStore(configured_store_path(self.config))
            hits = store.search(state.ctx.user_input, top_k=top_k, sources=["web_page", "memory_fact", "project_document"])
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
        if normalized == "shell_command" or capability in {"run_shell_command", "project_execute"}:
            return False
        if capability in {"project_read", "project_tree"}:
            return True
        return not self.skill_runtime.tool_is_mutating(normalized)

    def _policy_block_tool(
        self,
        *,
        state: TurnState,
        call: ToolCall,
        pass_id: str,
        message: str,
        code: str = "E_POLICY",
        on_event: Callable[[JsonObject], None] | None = None,
    ) -> None:
        result = {
            "ok": False,
            "data": None,
            "error": {
                "code": code,
                "message": message,
            },
            "meta": {"policy_blocked": True},
        }
        self.emit(on_event, {"type": "tool_result", "name": call.name, "id": call.id, "result": result})
        tool_message = {
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.name,
            "content": self.history.dumps(self.history.result(call.name, result)),
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
        return self.llm_client.call_with_retry(payload, stop_event, on_event, pass_id=f"{pass_id}_vision_retry")

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

    def _model_error(self, state: TurnState, messages: list[ChatMessage], exc: Exception, on_event) -> AgentTurnResult:
        message = self._friendly_vision_request_error(messages, exc)
        self.emit(on_event, {"type": "error", "text": message})
        return AgentTurnResult(
            status="error",
            content="",
            reasoning=state.full_reasoning,
            skill_exchanges=state.skill_exchanges,
            error=message,
            error_code="E_PROVIDER",
        )

    def record_tool_effects(self, state: TurnState, call: ToolCall, result: dict[str, object], *, policy_blocked: bool = False) -> None:
        self.tool_execution_engine.record_tool_effects(state, call, result, policy_blocked=policy_blocked)

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
            search_mode=state.classification.time_sensitive and state.search_tools_enabled,
            search_failures=state.completion.search_failure_count,
            fetched_urls=len(state.completion.fetched_urls),
            blocked_domains=sorted(state.completion.blocked_fetch_domains),
            collaboration_mode=self._normalize_collaboration_mode(getattr(state, "collaboration_mode", "execute")),
            content_chars=len(result.content),
            reasoning_chars=len(result.reasoning),
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
        session_id: str = "",
        stop_event=None,
    ) -> TurnState:
        branch_labels = branch_labels or []
        attachments = attachments or []
        ctx = self.classifier.build_skill_context(user_input, branch_labels, attachments, history_messages, loaded_skill_ids or [])
        classification = self.classifier.classify(ctx, stop_event=stop_event)
        selected = self.skill_runtime.select_skills(ctx)
        ctx.context_summary = str(context_summary or "").strip()
        relevant_skill_ids = [getattr(skill, "id", "") for skill in selected if getattr(skill, "id", "")]
        project_skill = self.skill_runtime.get_skill("project-ops")
        if (
            (classification.requires_project_action or classification.prefer_local_project_tools)
            and project_skill is not None
            and project_skill.enabled
            and project_skill.available
            and "project-ops" not in relevant_skill_ids
        ):
            relevant_skill_ids.append("project-ops")
            selected.append(project_skill)
        ctx.relevant_skill_ids = relevant_skill_ids
        state = self.policy_engine.build_turn_state(
            ctx,
            selected,
            history_messages,
            classification,
            collaboration_mode=self._normalize_collaboration_mode(collaboration_mode),
            context_summary=ctx.context_summary,
        )
        state.workspace_id = str(self.skill_runtime.project.project_root)
        state.session_id = session_id
        return state

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

    def _maybe_summarize_history(
        self, state: TurnState, system_messages: list[ChatMessage], tools: list[dict[str, Any]], stop_event
    ) -> str:
        if not self._summary_needed(system_messages + state.dynamic_history, tools):
            return "not_needed"
        summarize, retained = self.context_mgr.split_for_summary(state.dynamic_history)
        if not summarize:
            return "not_possible"
        previous_summary = str(state.context_summary or "").strip()
        summary = self.context_mgr.deterministic_summary(previous_summary, summarize)
        status = "deterministic"
        state.context_summary = summary
        state.ctx.context_summary = summary
        state.dynamic_history = retained
        return status

    def _system_content(self, state: TurnState) -> tuple[str, str]:
        snapshot = self.policy_engine.build_policy_snapshot(state)
        content = self.prompt_renderer.compose_system_content(state.selected, state.ctx)
        rules = self.prompt_renderer.render_policy_rules(snapshot)
        return content + ("\n\n" + rules if rules else ""), rules

    def run_model_pass(
        self,
        state: TurnState,
        thinking: bool,
        *,
        stop_event=None,
        on_event: Callable[[JsonObject], None] | None = None,
    ) -> AgentTurnResult | tuple[str, str, Any]:
        self.policy_engine.refresh_search_tools_enabled(state)
        state.pass_index += 1
        state.telemetry.pass_index = state.pass_index
        pass_id = f"pass_{state.pass_index}"

        if stop_event is not None and stop_event.is_set():
            return cancelled_turn_result(state)

        tools = self.skill_runtime.tools_for_turn(state.selected, ctx=state.ctx)
        if self._normalize_collaboration_mode(getattr(state, "collaboration_mode", "execute")) == "plan":
            tools = [
                item
                for item in tools
                if isinstance(item, dict)
                and isinstance(item.get("function"), dict)
                and self._tool_allowed_in_plan_mode(str(item["function"].get("name", "")).strip())
            ]
        system_content, policy_rules = self._system_content(state)
        system_messages: list[ChatMessage] = [cast(ChatMessage, {"role": "system", "content": system_content})]
        summary_status = self._maybe_summarize_history(state, system_messages, tools, stop_event)
        if summary_status == "deterministic":
            system_content, policy_rules = self._system_content(state)
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
        self.emit(on_event, {"type": "pass_start", "pass_id": pass_id})

        try:
            stream_result = self.llm_client.call_with_retry(payload, stop_event, on_event, pass_id=pass_id)
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
                    return self._model_error(state, model_messages, retry_exc, on_event)
                if stream_result is None:
                    return self._model_error(state, model_messages, exc, on_event)
            else:
                return self._model_error(state, model_messages, exc, on_event)

        if stream_result is None:
            return cancelled_turn_result(state)

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
            return cancelled_turn_result(state)

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
        session_id: str = "",
        stop_event=None,
        on_event: Callable[[JsonObject], None] | None = None,
        request_approval: ApprovalRequestFn | None = None,
        request_user_input: UserInputRequestFn | None = None,
        get_steering_messages: Callable[[], list[str]] | None = None,
    ) -> AgentTurnResult:
        state = self.prepare_turn(
            history_messages,
            user_input,
            branch_labels=branch_labels,
            attachments=attachments,
            loaded_skill_ids=loaded_skill_ids,
            context_summary=context_summary,
            collaboration_mode=collaboration_mode,
            session_id=session_id,
            stop_event=stop_event,
        )
        self.inject_policy_retrieval_context(state, on_event=on_event)

        def finish(result: AgentTurnResult) -> AgentTurnResult:
            result.journal = self.build_turn_journal(state, result)
            self.log_turn_summary(state, result)
            return result

        while True:
            if state.pass_index >= self.max_iterations or time.time() - state.telemetry.started_at >= self.turn_timeout_s:
                return finish(
                    AgentTurnResult(
                        status="error",
                        content="The task stopped after reaching its configured iteration or time limit.",
                        reasoning=state.full_reasoning,
                        skill_exchanges=state.skill_exchanges,
                        error="turn_budget_exhausted",
                        error_code="E_TOOL",
                    )
                )
            if self._is_stop_requested(stop_event):
                return finish(cancelled_turn_result(state))
            if get_steering_messages:
                for message in get_steering_messages():
                    state.dynamic_history.append(cast(ChatMessage, {"role": "user", "content": message}))
                    self.emit(on_event, {"type": "info", "text": "Applied queued steering message."})
            model_phase = self.run_model_pass(state, thinking, stop_event=stop_event, on_event=on_event)
            if isinstance(model_phase, AgentTurnResult):
                return finish(model_phase)

            pass_id, system_content, stream_result = model_phase

            if stream_result.finish_reason == "tool_calls":
                action, tool_phase_result = self.tool_loop.execute_tool_calls(
                    system_content=system_content,
                    state=state,
                    pass_id=pass_id,
                    stream_result=stream_result,
                    stop_event=stop_event,
                    on_event=on_event,
                    request_approval=request_approval,
                    request_user_input=request_user_input,
                )
                if action == "continue":
                    continue
                if tool_phase_result is None:
                    continue
                return finish(tool_phase_result)

            return finish(
                AgentTurnResult(
                    status="done",
                    content=self.sanitizer.sanitize_final_content(stream_result.content),
                    reasoning=state.full_reasoning,
                    skill_exchanges=state.skill_exchanges,
                )
            )


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
