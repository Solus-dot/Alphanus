from __future__ import annotations

import time
from typing import cast

from core.file_audit import build_file_audit_from_evidence
from core.message_types import JsonObject, JSONValue
from core.types import AgentTurnResult, TurnState


class TurnJournalBuilder:
    @staticmethod
    def trace_list(state: TurnState, key: str) -> list[dict[str, object]]:
        existing = state.trace_data.get(key)
        if isinstance(existing, list):
            return [cast(dict[str, object], item) for item in existing if isinstance(item, dict)]
        return []

    @staticmethod
    def set_trace_list(state: TurnState, key: str, rows: list[dict[str, object]]) -> None:
        state.trace_data[key] = cast(JSONValue, rows)

    def trace_add(self, state: TurnState, key: str, row: dict[str, object]) -> None:
        rows = self.trace_list(state, key)
        rows.append(row)
        self.set_trace_list(state, key, rows)

    def build(self, state: TurnState, result: AgentTurnResult, *, collaboration_mode: str) -> JsonObject:
        started_raw = state.trace_data.get("started_at", state.telemetry.started_at)
        started_at = float(started_raw) if isinstance(started_raw, (int, float)) else float(state.telemetry.started_at)
        finished_at = time.time()
        elapsed_ms = max(0, int((finished_at - started_at) * 1000))
        pass_first_tokens: list[int] = []
        for item in self.trace_list(state, "passes"):
            raw = item.get("first_token_latency_ms")
            if isinstance(raw, bool):
                continue
            if isinstance(raw, (int, float)):
                pass_first_tokens.append(max(0, int(raw)))
        first_token_latency_ms: int | None = pass_first_tokens[0] if pass_first_tokens else None
        tool_loop_depth = int(sum(max(0, int(v)) for v in state.completion.tool_counts.values()))
        return {
            "status": result.status,
            "error": result.error or "",
            "collaboration_mode": collaboration_mode,
            "selected_skills": [getattr(skill, "id", "") for skill in state.selected],
            "loaded_skill_ids": list(getattr(state.ctx, "loaded_skill_ids", []) or []),
            "tool_counts": dict(state.completion.tool_counts),
            "tool_evidence": [
                {"name": item.name, "args": item.args, "result": item.result, "policy_blocked": item.policy_blocked}
                for item in state.evidence
            ],
            "classification": {
                "source": state.classification.source,
                "followup_kind": state.classification.followup_kind,
                "time_sensitive": state.classification.time_sensitive,
                "requires_project_action": state.classification.requires_project_action,
                "prefer_local_project_tools": state.classification.prefer_local_project_tools,
            },
            "search_mode": state.classification.time_sensitive and state.search_tools_enabled,
            "search_failures": state.completion.search_failure_count,
            "has_fetch_evidence": state.completion.search_has_fetch_content,
            "model_usage": dict(state.telemetry.model_usage),
            "context_summary": str(getattr(state, "context_summary", "") or ""),
            "context_report": cast(JSONValue, dict(getattr(state, "context_report", {}) or {})),
            "timing": {
                "started_at": started_at,
                "finished_at": finished_at,
                "elapsed_ms": elapsed_ms,
                "pass_count": state.pass_index,
                "first_token_latency_ms": first_token_latency_ms,
            },
            "tool_loop_depth": tool_loop_depth,
            "file_audit": cast(
                JSONValue,
                build_file_audit_from_evidence(
                    state.evidence,
                    project_root=getattr(state.ctx, "project_root", ""),
                ),
            ),
            "turn_trace": {
                "passes": cast(JSONValue, self.trace_list(state, "passes")),
                "tool_calls": cast(JSONValue, self.trace_list(state, "tool_calls")),
                "tool_results": cast(JSONValue, self.trace_list(state, "tool_results")),
            },
        }
