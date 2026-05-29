from __future__ import annotations

import time
import urllib.parse
from collections.abc import Callable
from typing import cast

from agent.policies import search_rule
from core.message_types import ChatMessage, JsonObject
from core.types import AgentTurnResult, ShellConfirmationFn, TurnState, UserInputRequestFn


class ToolLoopEngine:
    def __init__(self, orchestrator) -> None:
        self.orchestrator = orchestrator

    def __getattr__(self, name: str):
        return getattr(self.orchestrator, name)

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
        state.action_depth += 1
        if state.action_depth > self.max_action_depth:
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

            if state.prefer_local_workspace_tools and self.skill_runtime.tool_is_blocked_for_local_workspace(call.name):
                if ":" in call.name or "." in call.name:
                    message = (
                        f"{call.name} is not exposed in this turn. Load the matching skill with skill_view(name), "
                        "then call the exact unqualified workspace tool name that appears in the tool list."
                    )
                else:
                    message = f"{call.name} is not allowed for local workspace file tasks; use workspace tools instead."
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
                confirm_shell=confirm_shell,
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
