from __future__ import annotations

from typing import List, cast

from core.message_types import JSONValue
from core.types import JsonObject, ToolExecutionRecord, TurnState


class EvidenceGuard:
    def __init__(self, skill_runtime) -> None:
        self.skill_runtime = skill_runtime

    @staticmethod
    def workspace_materialization_count(state: TurnState) -> int:
        return len(state.completion.materialized_paths)

    def tool_counts_as_workspace_mutation(self, record: ToolExecutionRecord) -> bool:
        if record.policy_blocked or not bool(record.result.get("ok")):
            return False
        if self.skill_runtime.tool_is_mutating(record.name):
            return True
        if record.name != "shell_command":
            return False
        meta = record.result.get("meta")
        return bool(isinstance(meta, dict) and meta.get("workspace_changed"))

    def workspace_mutation_count(self, state: TurnState) -> int:
        return sum(1 for record in state.evidence if self.tool_counts_as_workspace_mutation(record))

    @staticmethod
    def workspace_readback_count(state: TurnState) -> int:
        return len(state.completion.readback_paths)

    @staticmethod
    def needs_fetch_evidence(state: TurnState) -> bool:
        return state.search_mode and state.time_sensitive_query and not state.completion.search_has_fetch_content

    def workspace_action_evidence(self, state: TurnState) -> JsonObject:
        successful_mutating_tools: List[JSONValue] = []
        policy_blocked_tools: List[JSONValue] = []
        recent_tools: List[JSONValue] = []
        for record in state.evidence[-12:]:
            ok = bool(record.result.get("ok"))
            mutating = self.tool_counts_as_workspace_mutation(record)
            if mutating and record.name not in successful_mutating_tools:
                successful_mutating_tools.append(record.name)
            if record.policy_blocked and record.name not in policy_blocked_tools:
                policy_blocked_tools.append(record.name)
            error_obj = record.result.get("error")
            error = error_obj if isinstance(error_obj, dict) else {}
            recent_tools.append(
                cast(
                    JSONValue,
                    {
                        "name": record.name,
                        "ok": ok,
                        "mutating": mutating,
                        "policy_blocked": record.policy_blocked,
                        "error_code": str(error.get("code", "")).strip(),
                        "error_message": str(error.get("message", "")).strip()[:240],
                    },
                )
            )
        payload: JsonObject = {
            "tool_counts": dict(state.completion.tool_counts),
            "has_successful_mutation": bool(successful_mutating_tools),
            "successful_mutating_tools": successful_mutating_tools,
            "policy_blocked_tools": policy_blocked_tools,
            "recent_tools": recent_tools,
        }
        return payload
