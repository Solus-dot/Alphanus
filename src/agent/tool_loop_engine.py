from __future__ import annotations

import json
import time
import urllib.parse
from collections.abc import Callable
from typing import cast

from agent.policies import search_rule
from core.message_types import ChatMessage, JsonObject
from core.types import AgentTurnResult, ApprovalRequestFn, TurnState, UserInputRequestFn, cancelled_turn_result


class ToolLoopEngine:
    INSPECTION_LOOP_TOOLS = {"read_file", "read_files", "list_files", "project_tree", "find_files", "search_code"}

    def __init__(self, orchestrator) -> None:
        self.orchestrator = orchestrator

    @staticmethod
    def _tool_signature(call) -> str:
        try:
            args = json.dumps(call.arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except TypeError:
            args = json.dumps({key: str(value) for key, value in call.arguments.items()}, sort_keys=True, separators=(",", ":"))
        return f"{call.name}:{args}"

    @staticmethod
    def _approval_was_denied(result: dict[str, object]) -> bool:
        error = result.get("error")
        message = str(error.get("message", "")) if isinstance(error, dict) else ""
        return (
            not result.get("ok") and isinstance(error, dict) and error.get("code") == "E_POLICY" and "rejected by user" in message.lower()
        )

    def _is_non_mutating_project_inspection(self, state: TurnState, tool_name: str) -> bool:
        if tool_name not in self.INSPECTION_LOOP_TOOLS:
            return False
        reg = self.orchestrator.skill_runtime.tool_registration(tool_name)
        capability = str(getattr(reg, "capability", "") or "").strip().lower()
        return capability in {"project_read", "project_tree"} and not self.orchestrator.skill_runtime.tool_is_mutating(tool_name)

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
        self.orchestrator._policy_block_tool(state=state, call=call, pass_id=pass_id, code=code, message=message, on_event=on_event)

    @staticmethod
    def _error_result(state: TurnState, error: str) -> tuple[str, AgentTurnResult]:
        return "result", AgentTurnResult(
            status="error",
            content="",
            reasoning=state.full_reasoning,
            skill_exchanges=state.skill_exchanges,
            error=error,
            error_code="E_TOOL",
        )

    def _cancelled_after_tool(self, state: TurnState, call, stop_event, on_event) -> tuple[str, AgentTurnResult] | None:
        if not self.orchestrator._is_stop_requested(stop_event):
            return None
        self.orchestrator.emit(
            on_event,
            {"type": "info", "text": f"Cancellation requested after completed tool '{call.name}'. Stopping turn."},
        )
        return "result", cancelled_turn_result(state)

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
                    error_code="E_TOOL",
                ),
            )
        message = (
            f"{call.name} already succeeded with the same arguments in this turn. Use the prior result; "
            "choose a broader discovery tool, perform the requested mutation, or explain the blocker."
        )
        state.blocked_inspection_tool_signatures.add(signature)
        self._loop_block_tool(state=state, call=call, pass_id=pass_id, code="E_REPEATED_TOOL_CALL", message=message, on_event=on_event)
        return True, None

    @staticmethod
    def _inspection_paths(call) -> set[str]:
        if call.name == "read_file":
            path = str(call.arguments.get("filepath", "")).strip()
            return {path} if path else set()
        if call.name == "read_files":
            return {str(path).strip() for path in call.arguments.get("paths", []) if str(path).strip()}
        return set()

    @staticmethod
    def _merge_range(ranges: list[tuple[int, int]], start: int, end: int) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for current_start, current_end in sorted([*ranges, (start, end)]):
            if merged and current_start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], current_end))
            else:
                merged.append((current_start, current_end))
        return merged

    @staticmethod
    def _requested_read_range(state: TurnState, call) -> tuple[str, int, int] | None:
        if call.name != "read_file":
            return None
        path = str(call.arguments.get("filepath", "")).strip()
        if not path:
            return None
        start = max(1, int(call.arguments.get("start_line") or 1))
        end = int(call.arguments.get("end_line") or state.read_total_lines.get(path) or 2**63 - 1)
        return path, start, max(start, end)

    def _covered_read_reason(self, state: TurnState, call) -> str:
        requested = self._requested_read_range(state, call)
        if requested is None:
            return ""
        path, start, end = requested
        if not any(covered_start <= start and covered_end >= end for covered_start, covered_end in state.read_line_ranges.get(path, [])):
            return ""
        covered = ", ".join(f"{a}–{b}" for a, b in state.read_line_ranges[path])
        total = state.read_total_lines.get(path)
        unread = "none" if total and covered == f"1–{total}" else "request a range outside the covered intervals"
        return f"{path} lines {start}–{end} were already read. Covered: {covered}. Unread: {unread}. Use prior results."

    def _maybe_finalize_verified_write_readback(self, state: TurnState, call) -> str:
        paths = self._inspection_paths(call)
        if not paths or not paths <= state.verified_write_paths:
            return ""
        if not paths <= state.verified_write_readbacks:
            return ""
        return (
            "The requested file write already succeeded and was verified, and the written file has already been "
            "read back once. Finish now using that evidence. Do not inspect or rewrite the file again."
        )

    def _record_loop_progress_after_result(self, state: TurnState, call, result: dict[str, object]) -> None:
        if not bool(result.get("ok")):
            return
        data = result.get("data")
        payload = data if isinstance(data, dict) else {}
        if call.name in {"create_file", "edit_file"} and bool(payload.get("write_verified")):
            path = str(payload.get("filepath", "")).strip()
            if path:
                state.verified_write_paths.add(path)
            return
        if not self._is_non_mutating_project_inspection(state, call.name):
            return
        state.successful_inspection_tool_signatures.add(self._tool_signature(call))
        paths = self._inspection_paths(call)
        state.verified_write_readbacks.update(paths & state.verified_write_paths)
        requested = self._requested_read_range(state, call)
        if requested is not None:
            path, requested_start, requested_end = requested
            start = max(1, int(payload.get("resolved_start_line") or requested_start))
            end = max(start, int(payload.get("resolved_end_line") or requested_end))
            total = int(payload.get("total_line_count") or 0)
            if total:
                state.read_total_lines[path] = total
                end = min(end, total)
            state.read_line_ranges[path] = self._merge_range(state.read_line_ranges.get(path, []), start, end)

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
        if self.orchestrator._is_stop_requested(stop_event):
            return (
                "result",
                cancelled_turn_result(state),
            )
        if not stream_result.tool_calls:
            return self._error_result(state, "finish_reason tool_calls without tool calls")

        assistant_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": self.orchestrator.history.dumps(self.orchestrator.history.arguments(call.arguments)),
                    },
                }
                for call in stream_result.tool_calls
            ],
        }
        assistant_chat_message = cast(ChatMessage, assistant_msg)
        state.dynamic_history.append(assistant_chat_message)
        state.skill_exchanges.append(assistant_chat_message)

        force_continue_reason = ""

        for call in stream_result.tool_calls:
            call_trace = {
                "pass_id": pass_id,
                "id": call.id,
                "name": call.name,
                "arguments": dict(call.arguments),
                "started_at": time.time(),
            }
            self.orchestrator._trace_add(state, "tool_calls", call_trace)
            if self.orchestrator._is_stop_requested(stop_event):
                return (
                    "result",
                    cancelled_turn_result(state),
                )
            self.orchestrator.emit(
                on_event, {"type": "tool_call", "stream_id": call.stream_id, "name": call.name, "arguments": call.arguments, "id": call.id}
            )

            covered_read_reason = self._covered_read_reason(state, call)
            if covered_read_reason:
                self._loop_block_tool(
                    state=state,
                    call=call,
                    pass_id=pass_id,
                    code="E_READ_RANGE_COVERED",
                    message=covered_read_reason,
                    on_event=on_event,
                )
                continue

            verified_write_reason = self._maybe_finalize_verified_write_readback(state, call)
            if verified_write_reason:
                self._loop_block_tool(
                    state=state,
                    call=call,
                    pass_id=pass_id,
                    code="E_VERIFIED_WRITE_READBACK_COMPLETE",
                    message=verified_write_reason,
                    on_event=on_event,
                )
                continue

            blocked_current_call, blocked_result = self._maybe_block_repeated_inspection(
                state=state,
                call=call,
                pass_id=pass_id,
                on_event=on_event,
            )
            if blocked_result is not None:
                return "result", blocked_result
            if blocked_current_call:
                if self.orchestrator._is_stop_requested(stop_event):
                    return (
                        "result",
                        cancelled_turn_result(state),
                    )
                continue

            force_continue_reason = self.orchestrator.policy_engine.tool_budget_reason(state, call) or ""
            if force_continue_reason:
                if call.name not in {"web_search", "fetch_url"} or not state.search_tools_enabled:
                    return self._error_result(state, force_continue_reason)
                self._loop_block_tool(
                    state=state,
                    call=call,
                    pass_id=pass_id,
                    code="E_TOOL_LOOP_BUDGET",
                    message=force_continue_reason,
                    on_event=on_event,
                )
                break

            if self.orchestrator._normalize_collaboration_mode(
                getattr(state, "collaboration_mode", "execute")
            ) == "plan" and not self.orchestrator._tool_allowed_in_plan_mode(call.name):
                self.orchestrator._policy_block_tool(
                    state=state,
                    call=call,
                    pass_id=pass_id,
                    message=(f"{call.name} is not allowed in plan mode; use non-mutating inspection tools or switch to execute mode."),
                    on_event=on_event,
                )
                if cancelled := self._cancelled_after_tool(state, call, stop_event, on_event):
                    return cancelled
                continue

            if state.classification.prefer_local_project_tools and self.orchestrator.skill_runtime.tool_is_blocked_for_local_project(
                call.name
            ):
                if ":" in call.name or "." in call.name:
                    message = (
                        f"{call.name} is not exposed in this turn. Load the matching skill with skill_view(name), "
                        "then call the exact unqualified project tool name that appears in the tool list."
                    )
                else:
                    message = f"{call.name} is not allowed for local project file tasks; use project tools instead."
                self.orchestrator._policy_block_tool(
                    state=state,
                    call=call,
                    pass_id=pass_id,
                    message=message,
                    on_event=on_event,
                )
                if cancelled := self._cancelled_after_tool(state, call, stop_event, on_event):
                    return cancelled
                continue

            if state.classification.time_sensitive and state.search_tools_enabled and call.name == "fetch_url":
                raw_url = str(call.arguments.get("url", "")).strip()
                if raw_url:
                    host = urllib.parse.urlparse(raw_url).netloc.lower()
                    if raw_url in state.completion.fetched_urls:
                        force_continue_reason = search_rule(
                            "This URL was already fetched in this turn.",
                            "Do not retry the same page.",
                            "Answer from the evidence already gathered.",
                        )
                        break
                    if host and host in state.completion.blocked_fetch_domains:
                        force_continue_reason = search_rule(
                            "This source domain already blocked a fetch attempt in this turn.",
                            "Do not retry the same blocked domain.",
                            "Answer from the remaining evidence.",
                        )
                        break

            result = self.orchestrator.skill_runtime.execute_tool_call(
                call.name,
                call.arguments,
                selected=state.selected,
                ctx=state.ctx,
                request_approval=request_approval,
                request_user_input=request_user_input,
                stop_event=stop_event,
            )
            self.orchestrator.emit(on_event, {"type": "tool_result", "name": call.name, "id": call.id, "result": result})
            tool_message = {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": self.orchestrator.history.dumps(self.orchestrator.history.result(call.name, result)),
            }
            tool_chat_message = cast(ChatMessage, tool_message)
            state.dynamic_history.append(tool_chat_message)
            state.skill_exchanges.append(tool_chat_message)
            approval_denied = self._approval_was_denied(result)
            self.orchestrator.record_tool_effects(state, call, result, policy_blocked=approval_denied)
            self._record_loop_progress_after_result(state, call, result)
            self.orchestrator._trace_add(
                state,
                "tool_results",
                {
                    "pass_id": pass_id,
                    "id": call.id,
                    "name": call.name,
                    "result": result,
                    "policy_blocked": approval_denied,
                    "finished_at": time.time(),
                },
            )
            if cancelled := self._cancelled_after_tool(state, call, stop_event, on_event):
                return cancelled
            if approval_denied:
                return (
                    "result",
                    AgentTurnResult(
                        status="done",
                        content="The requested action was not performed because approval was denied.",
                        reasoning=state.full_reasoning,
                        skill_exchanges=state.skill_exchanges,
                    ),
                )
            if call.name == "skill_view" and result.get("ok"):
                state.selected = self.orchestrator.skill_runtime.select_skills(state.ctx)
                self.orchestrator.policy_engine.refresh_search_tools_enabled(state)

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

            if not (state.classification.time_sensitive and state.search_tools_enabled):
                continue
            if call.name == "fetch_url" and state.completion.search_failure_count >= 2:
                force_continue_reason = search_rule(
                    "The search provider has already failed repeatedly.",
                    "Do not use memory or prior knowledge to fill gaps.",
                    "If the fetched page does not explicitly answer the question, say you could not verify it.",
                )
                break
            if call.name not in {"web_search", "fetch_url"} and state.completion.search_failure_count >= 2:
                force_continue_reason = search_rule(
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
                force_continue_reason = search_rule(
                    "Search failed repeatedly and no successful results were gathered.",
                    "State plainly that you could not verify the answer from reliable web results in this turn.",
                    "Do not speculate or answer from prior knowledge.",
                )
                break
            if call.name == "fetch_url" and not result.get("ok") and state.completion.search_has_success:
                force_continue_reason = search_rule(
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
                force_continue_reason = search_rule(
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
                force_continue_reason = search_rule(
                    "Enough pages have been fetched.",
                    "Answer from the gathered evidence now.",
                    "Do not fetch additional pages.",
                )
                break

        if force_continue_reason:
            state.dynamic_history.append(cast(ChatMessage, {"role": "system", "content": force_continue_reason}))
        return "continue", None
