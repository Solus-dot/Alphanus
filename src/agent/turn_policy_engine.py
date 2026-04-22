from __future__ import annotations

import uuid

from core.message_types import ChatMessage
from core.types import CompletionEvidence, TurnPolicySnapshot, TurnState, TurnTelemetry


class TurnPolicyEngine:
    def __init__(self, skill_runtime, default_tool_budgets: dict[str, int]) -> None:
        self.skill_runtime = skill_runtime
        self.default_tool_budgets = dict(default_tool_budgets)

    def build_turn_state(self, ctx, selected, history_messages: list[ChatMessage], classification) -> TurnState:
        state = TurnState(
            ctx=ctx,
            selected=selected,
            dynamic_history=list(history_messages),
            skill_exchanges=[],
            classification=classification,
            completion=CompletionEvidence(),
            telemetry=TurnTelemetry(
                turn_id=f"turn_{uuid.uuid4().hex[:10]}",
                classification_source=classification.source,
            ),
            search_tools_enabled=False,
            evidence=[],
            tool_budgets=dict(self.default_tool_budgets),
        )
        self.refresh_search_tools_enabled(state)
        return state

    def refresh_search_tools_enabled(self, state: TurnState) -> None:
        turn_tool_names = set(self.skill_runtime.allowed_tool_names(state.selected, ctx=state.ctx))
        state.search_tools_enabled = "web_search" in turn_tool_names

    def tool_budget_reason(self, state: TurnState, call) -> str | None:
        limit = state.tool_budgets.get(call.name)
        if not limit:
            return None
        if state.completion.tool_counts.get(call.name, 0) < limit:
            return None
        if call.name == "web_search" and state.time_sensitive_query:
            return "\n".join(
                [
                    "Search completion rule:",
                    "- The search-attempt budget is exhausted.",
                    "- Answer only from the evidence already gathered.",
                    "- Do not issue more search calls.",
                ]
            )
        if call.name == "fetch_url" and state.time_sensitive_query:
            return "\n".join(
                [
                    "Fetch completion rule:",
                    "- The fetch budget is exhausted.",
                    "- Answer only from the evidence already gathered.",
                    "- Do not fetch more pages.",
                ]
            )
        return f"Tool budget exceeded for {call.name} ({limit})"

    def build_policy_snapshot(self, state: TurnState) -> TurnPolicySnapshot:
        turn_tool_names = set(self.skill_runtime.allowed_tool_names(state.selected, ctx=state.ctx))
        return TurnPolicySnapshot(
            search_mode=state.search_mode,
            time_sensitive_query=state.time_sensitive_query,
            forced_search_retry=state.forced_search_retry and state.completion.tool_counts.get("web_search", 0) == 0,
            requires_workspace_action=state.requires_workspace_action,
            forced_action_retry=state.forced_action_retry and not state.completion.tool_counts,
            explicit_external_path=state.explicit_external_path,
            prefer_local_workspace_tools=state.prefer_local_workspace_tools,
            shell_tool_exposed="shell_command" in turn_tool_names,
        )
