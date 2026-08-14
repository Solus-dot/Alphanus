from __future__ import annotations

import time
import uuid

from core.message_types import ChatMessage
from core.types import CompletionEvidence, TurnPolicySnapshot, TurnState, TurnTelemetry


def build_turn_state(
    skill_runtime,
    default_tool_budgets: dict[str, int],
    ctx,
    selected,
    history_messages: list[ChatMessage],
    classification,
    *,
    collaboration_mode: str = "execute",
    context_summary: str = "",
) -> TurnState:
    normalized_mode = "plan" if str(collaboration_mode or "").strip().lower() == "plan" else "execute"
    state = TurnState(
        ctx=ctx,
        selected=selected,
        dynamic_history=list(history_messages),
        skill_exchanges=[],
        classification=classification,
        completion=CompletionEvidence(),
        telemetry=TurnTelemetry(turn_id=f"turn_{uuid.uuid4().hex[:10]}"),
        search_tools_enabled=False,
        evidence=[],
        tool_budgets=dict(default_tool_budgets),
        collaboration_mode=normalized_mode,
        context_summary=str(context_summary or ""),
        trace_data={"started_at": time.time(), "passes": [], "tool_calls": [], "tool_results": []},
    )
    refresh_search_tools_enabled(skill_runtime, state)
    return state


def refresh_search_tools_enabled(skill_runtime, state: TurnState) -> None:
    state.search_tools_enabled = "web_search" in set(skill_runtime.allowed_tool_names(state.selected, ctx=state.ctx))


def tool_budget_reason(state: TurnState, call) -> str | None:
    limit = state.tool_budgets.get(call.name)
    if not limit or state.completion.tool_counts.get(call.name, 0) < limit:
        return None
    if call.name in {"web_search", "fetch_url"}:
        operation, directive = ("search-attempt", "issue more search calls") if call.name == "web_search" else ("fetch", "fetch more pages")
        return "\n".join(
            [
                "Search completion rule:",
                f"- The {operation} budget is exhausted.",
                "- Answer only from the evidence already gathered.",
                f"- Do not {directive}.",
            ]
        )
    return f"Tool budget exceeded for {call.name} ({limit})"


def build_policy_snapshot(skill_runtime, state: TurnState) -> TurnPolicySnapshot:
    turn_tool_names = set(skill_runtime.allowed_tool_names(state.selected, ctx=state.ctx))
    return TurnPolicySnapshot(
        search_mode=state.classification.time_sensitive and state.search_tools_enabled,
        time_sensitive_query=state.classification.time_sensitive,
        requires_project_action=state.classification.requires_project_action,
        explicit_external_path=state.classification.explicit_external_path,
        prefer_local_project_tools=state.classification.prefer_local_project_tools,
        shell_tool_exposed="shell_command" in turn_tool_names,
        collaboration_mode=("plan" if str(getattr(state, "collaboration_mode", "execute") or "").strip().lower() == "plan" else "execute"),
    )
