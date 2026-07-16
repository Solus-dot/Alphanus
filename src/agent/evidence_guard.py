from __future__ import annotations

from typing import cast

from core.message_types import JSONValue
from core.types import JsonObject, ToolExecutionRecord, TurnState


class EvidenceGuard:
    def __init__(self, skill_runtime) -> None:
        self.skill_runtime = skill_runtime

    @staticmethod
    def project_materialization_count(state: TurnState) -> int:
        return len(state.completion.materialized_paths)

    def tool_counts_as_project_mutation(self, record: ToolExecutionRecord) -> bool:
        if record.policy_blocked or not bool(record.result.get("ok")):
            return False
        if record.name == "shell_command":
            meta = record.result.get("meta")
            return bool(isinstance(meta, dict) and meta.get("project_changed"))
        if record.name == "run_skill":
            meta = record.result.get("meta")
            return bool(isinstance(meta, dict) and meta.get("project_changed"))
        reg = self.skill_runtime.tool_registration(record.name)
        capability = str(getattr(reg, "capability", "") or "").strip().lower()
        return capability.startswith("project_") and self.skill_runtime.tool_is_mutating(record.name)

    def project_mutation_count(self, state: TurnState) -> int:
        return sum(1 for record in state.evidence if self.tool_counts_as_project_mutation(record))

    @staticmethod
    def project_readback_count(state: TurnState) -> int:
        return len(state.completion.readback_paths)

    @staticmethod
    def needs_fetch_evidence(state: TurnState) -> bool:
        return state.classification.time_sensitive and state.search_tools_enabled and not state.completion.search_has_fetch_content

    def project_action_evidence(self, state: TurnState) -> JsonObject:
        successful_tools: list[JSONValue] = []
        successful_mutating_tools: list[JSONValue] = []
        successful_non_mutating_tools: list[JSONValue] = []
        successful_action_labels: list[str] = []
        policy_blocked_tools: list[JSONValue] = []
        recent_tools: list[JSONValue] = []
        for record in state.evidence[-12:]:
            ok = bool(record.result.get("ok"))
            mutating = self.tool_counts_as_project_mutation(record)
            reg = self.skill_runtime.tool_registration(record.name)
            capability = str(getattr(reg, "capability", "") or "").strip()
            actions = [str(item).strip().lower() for item in (getattr(reg, "actions", ()) or ()) if str(item).strip()]
            if ok and not record.policy_blocked and record.name != "skill_view" and record.name not in successful_tools:
                successful_tools.append(record.name)
            if mutating and record.name not in successful_mutating_tools:
                successful_mutating_tools.append(record.name)
            if ok and not mutating and not record.policy_blocked and record.name != "skill_view" and record.name not in successful_non_mutating_tools:
                successful_non_mutating_tools.append(record.name)
                for action in actions:
                    if action not in successful_action_labels:
                        successful_action_labels.append(action)
            if record.policy_blocked and record.name not in policy_blocked_tools:
                policy_blocked_tools.append(record.name)
            error_obj = record.result.get("error")
            error = error_obj if isinstance(error_obj, dict) else {}
            recent_tools.append(
                cast(
                    JSONValue,
                    {
                        "name": record.name,
                        "capability": capability,
                        "actions": actions,
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
            "has_successful_tool": bool(successful_tools),
            "successful_tools": successful_tools,
            "has_successful_mutation": bool(successful_mutating_tools),
            "successful_mutating_tools": successful_mutating_tools,
            "has_successful_non_mutating_tool": bool(successful_non_mutating_tools),
            "successful_non_mutating_tools": successful_non_mutating_tools,
            "successful_action_labels": cast(JSONValue, successful_action_labels),
            "policy_blocked_tools": policy_blocked_tools,
            "recent_tools": recent_tools,
        }
        return payload
