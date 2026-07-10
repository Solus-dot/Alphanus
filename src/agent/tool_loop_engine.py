from __future__ import annotations

import json
import time
import urllib.parse
from collections.abc import Callable
from typing import cast

from agent.policies import search_rule
from core.message_types import ChatMessage, JsonObject
from core.types import AgentTurnResult, ApprovalRequestFn, TurnState, UserInputRequestFn


class ToolLoopEngine:
    INSPECTION_LOOP_TOOLS = {"read_file", "read_files", "list_files", "project_tree", "find_files", "search_code"}
    MUTATION_INTENT_TERMS = {
        "add",
        "capitalize",
        "change",
        "create",
        "delete",
        "edit",
        "fix",
        "modify",
        "move",
        "remove",
        "rename",
        "replace",
        "save",
        "update",
        "write",
    }

    def __init__(self, orchestrator) -> None:
        self.orchestrator = orchestrator

    def __getattr__(self, name: str):
        return getattr(self.orchestrator, name)

    @staticmethod
    def _tool_signature(call) -> str:
        try:
            args = json.dumps(call.arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except TypeError:
            args = json.dumps({key: str(value) for key, value in call.arguments.items()}, sort_keys=True, separators=(",", ":"))
        return f"{call.name}:{args}"

    def _is_non_mutating_project_inspection(self, state: TurnState, tool_name: str) -> bool:
        if tool_name not in self.INSPECTION_LOOP_TOOLS:
            return False
        reg = self.skill_runtime.tool_registration(tool_name)
        capability = str(getattr(reg, "capability", "") or "").strip().lower()
        return capability in {"project_read", "project_tree"} and not self.skill_runtime.tool_is_mutating(tool_name)

    def _requires_project_mutation(self, state: TurnState) -> bool:
        if not state.requires_project_action or self.project_mutation_count(state) > 0:
            return False
        user_text = str(getattr(state.ctx, "user_input", "") or "").lower()
        return any(term in user_text for term in self.MUTATION_INTENT_TERMS)

    def _loop_block_tool(
        self,
        *,
        state: TurnState,
        call,
        pass_id: str,
        code: str,
        message: str,
        on_event: Callable[[JsonObject], None] | None,
    ) -> None:
        self._policy_block_tool(state=state, call=call, pass_id=pass_id, code=code, message=message, on_event=on_event)

    def _maybe_block_repeated_inspection(
        self,
        *,
        state: TurnState,
        call,
        pass_id: str,
        on_event: Callable[[JsonObject], None] | None,
    ) -> tuple[bool, AgentTurnResult | None]:
        if not self._is_non_mutating_project_inspection(state, call.name):
            return False, None
        signature = self._tool_signature(call)
        if signature not in state.successful_inspection_tool_signatures:
            return False, None
        if signature in state.blocked_inspection_tool_signatures:
            message = (
                f"{call.name} already succeeded with the same arguments and was already blocked once. "
                "The turn is stopped to avoid an inspection loop."
            )
            self._loop_block_tool(state=state, call=call, pass_id=pass_id, code="E_TOOL_LOOP_STUCK", message=message, on_event=on_event)
            return (
                True,
                AgentTurnResult(
                    status="error",
                    content=f"[agent error] {message}",
                    reasoning=state.full_reasoning,
                    skill_exchanges=state.skill_exchanges,
                    error="tool_loop_stuck",
                ),
            )
        message = (
            f"{call.name} already succeeded with the same arguments in this turn. Use the prior result; "
            "choose a broader discovery tool, perform the requested mutation, or explain the blocker."
        )
        state.blocked_inspection_tool_signatures.add(signature)
        self._loop_block_tool(state=state, call=call, pass_id=pass_id, code="E_REPEATED_TOOL_CALL", message=message, on_event=on_event)
        return True, None

    def _maybe_block_stalled_project_action(
        self,
        *,
        state: TurnState,
        call,
        pass_id: str,
        on_event: Callable[[JsonObject], None] | None,
    ) -> tuple[bool, AgentTurnResult | None]:
        if not self._requires_project_mutation(state) or not state.project_target_inspected:
            return False, None
        if not self._is_non_mutating_project_inspection(state, call.name):
            return False, None
        if state.post_target_inspection_calls < 2:
            return False, None
        state.project_action_stall_blocks += 1
        if state.project_action_stall_blocks >= 2:
            message = (
                "The requested project mutation has enough inspection evidence, but the model kept requesting "
                "non-mutating project inspection tools. The turn is stopped to avoid an inspection loop."
            )
            self._loop_block_tool(state=state, call=call, pass_id=pass_id, code="E_TOOL_LOOP_STUCK", message=message, on_event=on_event)
            return (
                True,
                AgentTurnResult(
                    status="error",
                    content=f"[agent error] {message}",
                    reasoning=state.full_reasoning,
                    skill_exchanges=state.skill_exchanges,
                    error="project_action_stuck",
                ),
            )
        message = (
            "The requested project mutation has enough inspection evidence. Do not inspect again; call the "
            "appropriate mutating project tool now, or explain the exact blocker."
        )
        self._loop_block_tool(state=state, call=call, pass_id=pass_id, code="E_PROJECT_ACTION_STALLED", message=message, on_event=on_event)
        return True, None

    def _record_loop_progress_after_result(self, state: TurnState, call, result: dict[str, object]) -> None:
        if not bool(result.get("ok")):
            return
        if self.project_mutation_count(state) > 0:
            state.project_target_inspected = False
            state.post_target_inspection_calls = 0
            state.project_action_stall_blocks = 0
            return
        if not self._is_non_mutating_project_inspection(state, call.name):
            return
        state.successful_inspection_tool_signatures.add(self._tool_signature(call))
        if not self._requires_project_mutation(state):
            return
        if state.project_target_inspected:
            state.post_target_inspection_calls += 1
            return
        if call.name in {"read_file", "read_files", "find_files", "project_tree"}:
            state.project_target_inspected = True
            state.post_target_inspection_calls = 0

    def execute_tool_calls(
        self,
        *,
        system_content: str,
        state: TurnState,
        pass_id: str,
        stream_result,
        stop_event=None,
        on_event: Callable[[JsonObject], None] | None = None,
        request_approval: ApprovalRequestFn | None = None,
        request_user_input: UserInputRequestFn | None = None,
    ) -> tuple[str, AgentTurnResult | None]:
        if self._is_stop_requested(stop_event):
            return (
                "result",
                AgentTurnResult(
                    status="cancelled",
                    content="",
                    reasoning=state.full_reasoning,
                    skill_exchanges=state.skill_exchanges,
                ),
            )
        if not stream_result.tool_calls:
            return (
                "result",
                AgentTurnResult(
                    status="error",
                    content="",
                    reasoning=state.full_reasoning,
                    skill_exchanges=state.skill_exchanges,
                    error="finish_reason tool_calls without tool calls",
                ),
            )

        assistant_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": self.safe_json_dumps(self.tool_call_args_for_history(call.arguments)),
                    },
                }
                for call in stream_result.tool_calls
            ],
        }
        assistant_chat_message = cast(ChatMessage, assistant_msg)
        state.dynamic_history.append(assistant_chat_message)
        state.skill_exchanges.append(assistant_chat_message)

        force_finalize_reason = ""
        if state.action_depth >= self.max_action_depth:
            for call in stream_result.tool_calls:
                self.emit(
                    on_event,
                    {"type": "tool_call", "stream_id": call.stream_id, "name": call.name, "arguments": call.arguments, "id": call.id},
                )
                self._loop_block_tool(
                    state=state,
                    call=call,
                    pass_id=pass_id,
                    code="E_TOOL_LOOP_BUDGET",
                    message=f"Max skill action depth ({self.max_action_depth}) exceeded before executing {call.name}.",
                    on_event=on_event,
                )
            if state.search_mode and state.completion.search_has_success:
                return (
                    "finalized",
                    self.finalize_turn(
                        system_content,
                        state,
                        stop_event,
                        on_event,
                        pass_id,
                        search_rule(
                            "The search loop budget is exhausted.",
                            "Answer using the successful search or fetch results already in the conversation.",
                            "Do not search again.",
                        ),
                    ),
                )
            return (
                "result",
                AgentTurnResult(
                    status="error",
                    content="",
                    reasoning=state.full_reasoning,
                    skill_exchanges=state.skill_exchanges,
                    error=f"Max skill action depth ({self.max_action_depth}) exceeded",
                ),
            )
        state.action_depth += 1

        for call in stream_result.tool_calls:
            call_trace = {
                "pass_id": pass_id,
                "id": call.id,
                "name": call.name,
                "arguments": dict(call.arguments),
                "started_at": time.time(),
            }
            self._trace_add(state, "tool_calls", call_trace)
            if self._is_stop_requested(stop_event):
                return (
                    "result",
                    AgentTurnResult(
                        status="cancelled",
                        content="",
                        reasoning=state.full_reasoning,
                        skill_exchanges=state.skill_exchanges,
                    ),
                )
            self.emit(
                on_event, {"type": "tool_call", "stream_id": call.stream_id, "name": call.name, "arguments": call.arguments, "id": call.id}
            )

            blocked_current_call, blocked_result = self._maybe_block_repeated_inspection(
                state=state,
                call=call,
                pass_id=pass_id,
                on_event=on_event,
            )
            if blocked_result is not None:
                return "result", blocked_result
            if blocked_current_call:
                if self._is_stop_requested(stop_event):
                    return (
                        "result",
                        AgentTurnResult(
                            status="cancelled",
                            content="",
                            reasoning=state.full_reasoning,
                            skill_exchanges=state.skill_exchanges,
                        ),
                    )
                continue

            stalled_current_call, stalled_result = self._maybe_block_stalled_project_action(
                state=state,
                call=call,
                pass_id=pass_id,
                on_event=on_event,
            )
            if stalled_result is not None:
                return "result", stalled_result
            if stalled_current_call:
                if self._is_stop_requested(stop_event):
                    return (
                        "result",
                        AgentTurnResult(
                            status="cancelled",
                            content="",
                            reasoning=state.full_reasoning,
                            skill_exchanges=state.skill_exchanges,
                        ),
                    )
                continue

            force_finalize_reason = self.tool_budget_reason(state, call)
            if force_finalize_reason:
                if not state.search_mode:
                    return (
                        "result",
                        AgentTurnResult(
                            status="error",
                            content="",
                            reasoning=state.full_reasoning,
                            skill_exchanges=state.skill_exchanges,
                            error=force_finalize_reason,
                        ),
                    )
                break

            if self._normalize_collaboration_mode(
                getattr(state, "collaboration_mode", "execute")
            ) == "plan" and not self._tool_allowed_in_plan_mode(call.name):
                self._policy_block_tool(
                    state=state,
                    call=call,
                    pass_id=pass_id,
                    message=(f"{call.name} is not allowed in plan mode; use non-mutating inspection tools or switch to execute mode."),
                    on_event=on_event,
                )
                if self._is_stop_requested(stop_event):
                    self.emit(
                        on_event,
                        {
                            "type": "info",
                            "text": f"Cancellation requested after completed tool '{call.name}'. Stopping turn.",
                        },
                    )
                    return (
                        "result",
                        AgentTurnResult(
                            status="cancelled",
                            content="",
                            reasoning=state.full_reasoning,
                            skill_exchanges=state.skill_exchanges,
                        ),
                    )
                continue

            if state.prefer_local_project_tools and self.skill_runtime.tool_is_blocked_for_local_project(call.name):
                if ":" in call.name or "." in call.name:
                    message = (
                        f"{call.name} is not exposed in this turn. Load the matching skill with skill_view(name), "
                        "then call the exact unqualified project tool name that appears in the tool list."
                    )
                else:
                    message = f"{call.name} is not allowed for local project file tasks; use project tools instead."
                self._policy_block_tool(
                    state=state,
                    call=call,
                    pass_id=pass_id,
                    message=message,
                    on_event=on_event,
                )
                if self._is_stop_requested(stop_event):
                    self.emit(
                        on_event,
                        {
                            "type": "info",
                            "text": f"Cancellation requested after completed tool '{call.name}'. Stopping turn.",
                        },
                    )
                    return (
                        "result",
                        AgentTurnResult(
                            status="cancelled",
                            content="",
                            reasoning=state.full_reasoning,
                            skill_exchanges=state.skill_exchanges,
                        ),
                    )
                continue

            if state.search_mode and call.name == "fetch_url":
                raw_url = str(call.arguments.get("url", "")).strip()
                if raw_url:
                    host = urllib.parse.urlparse(raw_url).netloc.lower()
                    if raw_url in state.completion.fetched_urls:
                        force_finalize_reason = search_rule(
                            "This URL was already fetched in this turn.",
                            "Do not retry the same page.",
                            "Answer from the evidence already gathered.",
                        )
                        break
                    if host and host in state.completion.blocked_fetch_domains:
                        force_finalize_reason = search_rule(
                            "This source domain already blocked a fetch attempt in this turn.",
                            "Do not retry the same blocked domain.",
                            "Answer from the remaining evidence.",
                        )
                        break

            result = self.skill_runtime.execute_tool_call(
                call.name,
                call.arguments,
                selected=state.selected,
                ctx=state.ctx,
                request_approval=request_approval,
                request_user_input=request_user_input,
            )
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
            self.record_tool_effects(state, call, result)
            self._record_loop_progress_after_result(state, call, result)
            self._trace_add(
                state,
                "tool_results",
                {
                    "pass_id": pass_id,
                    "id": call.id,
                    "name": call.name,
                    "result": result,
                    "policy_blocked": False,
                    "finished_at": time.time(),
                },
            )
            if self._is_stop_requested(stop_event):
                self.emit(
                    on_event,
                    {
                        "type": "info",
                        "text": f"Cancellation requested after completed tool '{call.name}'. Stopping turn.",
                    },
                )
                return (
                    "result",
                    AgentTurnResult(
                        status="cancelled",
                        content="",
                        reasoning=state.full_reasoning,
                        skill_exchanges=state.skill_exchanges,
                    ),
                )
            if call.name == "skill_view" and result.get("ok"):
                state.selected = self.skill_runtime.select_skills(state.ctx)
                self.refresh_search_tools_enabled(state)

            if (
                call.name == "request_user_input"
                and result.get("ok")
                and isinstance(result.get("data"), dict)
                and bool(result["data"].get("awaiting_user_input"))
            ):
                prompt_data = result["data"]
                question = str(prompt_data.get("question", "")).strip()
                options = prompt_data.get("options")
                lines = [question] if question else []
                if isinstance(options, list) and options:
                    lines.append("Options: " + " | ".join(str(item) for item in options[:6]))
                prompt_text = "\n".join(lines)
                return (
                    "result",
                    AgentTurnResult(
                        status="done",
                        content=prompt_text,
                        reasoning=state.full_reasoning,
                        skill_exchanges=state.skill_exchanges,
                    ),
                )

            if not state.search_mode:
                continue
            if call.name == "fetch_url" and state.completion.search_failure_count >= 2:
                force_finalize_reason = search_rule(
                    "The search provider has already failed repeatedly.",
                    "Do not use memory or prior knowledge to fill gaps.",
                    "If the fetched page does not explicitly answer the question, say you could not verify it.",
                )
                break
            if call.name not in {"web_search", "fetch_url"} and state.completion.search_failure_count >= 2:
                force_finalize_reason = search_rule(
                    "Search has failed repeatedly.",
                    "Do not switch to memory recall or unrelated tools.",
                    "Answer only with verified evidence, or say verification failed.",
                )
                break
            if (
                call.name == "web_search"
                and not result.get("ok")
                and state.completion.search_failure_count >= 2
                and not state.completion.search_has_success
            ):
                finalized = self.finalize_turn(
                    system_content,
                    state,
                    stop_event,
                    on_event,
                    pass_id,
                    search_rule(
                        "Search failed repeatedly and no successful results were gathered.",
                        "State plainly that you could not verify the answer from reliable web results in this turn.",
                        "Do not speculate or answer from prior knowledge.",
                    ),
                )
                return "finalized", finalized
            if call.name == "fetch_url" and not result.get("ok") and state.completion.search_has_success:
                force_finalize_reason = search_rule(
                    "A page fetch failed.",
                    "Continue with the successful search results and any successful fetches already gathered.",
                    "Do not keep retrying searches indefinitely.",
                )
                break
            if (
                call.name == "web_search"
                and state.completion.tool_counts.get("web_search", 0) >= state.tool_budgets.get("web_search", 0)
                and state.completion.search_has_success
            ):
                force_finalize_reason = search_rule(
                    "Enough search attempts have already been made.",
                    "Summarize from the best available results now.",
                    "Do not issue more search calls.",
                )
                break
            if (
                call.name == "fetch_url"
                and state.completion.tool_counts.get("fetch_url", 0) >= state.tool_budgets.get("fetch_url", 0)
                and state.completion.search_has_fetch_content
            ):
                force_finalize_reason = search_rule(
                    "Enough pages have been fetched.",
                    "Answer from the gathered evidence now.",
                    "Do not fetch additional pages.",
                )
                break

        if force_finalize_reason:
            return (
                "finalized",
                self.finalize_turn(system_content, state, stop_event, on_event, pass_id, force_finalize_reason),
            )
        return "continue", None
