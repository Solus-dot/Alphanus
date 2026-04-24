from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.types import AgentTurnResult


@dataclass(slots=True)
class HarnessMetrics:
    turns_total: int = 0
    turns_done: int = 0
    turns_cancelled: int = 0
    turns_error: int = 0
    tool_calls_total: int = 0
    tool_failures_total: int = 0
    tool_loop_depth_total: int = 0
    first_token_latency_samples: int = 0
    first_token_latency_total_ms: int = 0

    @staticmethod
    def _to_int(value: Any, default: int = 0) -> int:
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return int(value)
        return default

    def record(self, result: AgentTurnResult) -> None:
        self.turns_total += 1
        status = str(getattr(result, "status", "")).strip().lower()
        if status == "done":
            self.turns_done += 1
        elif status == "cancelled":
            self.turns_cancelled += 1
        else:
            self.turns_error += 1

        journal = result.journal if isinstance(result.journal, dict) else {}
        tool_counts_raw = journal.get("tool_counts", {})
        if isinstance(tool_counts_raw, dict):
            for value in tool_counts_raw.values():
                self.tool_calls_total += max(0, self._to_int(value))

        loop_depth = self._to_int(journal.get("tool_loop_depth"))
        if loop_depth <= 0 and isinstance(tool_counts_raw, dict):
            loop_depth = sum(max(0, self._to_int(value)) for value in tool_counts_raw.values())
        self.tool_loop_depth_total += max(0, loop_depth)

        turn_trace = journal.get("turn_trace", {})
        if isinstance(turn_trace, dict):
            tool_results = turn_trace.get("tool_results", [])
            if isinstance(tool_results, list):
                for row in tool_results:
                    if not isinstance(row, dict):
                        continue
                    payload = row.get("result")
                    if isinstance(payload, dict) and payload.get("ok") is False:
                        self.tool_failures_total += 1

        timing = journal.get("timing", {})
        if isinstance(timing, dict):
            first_token_latency = self._to_int(timing.get("first_token_latency_ms"), default=-1)
            if first_token_latency >= 0:
                self.first_token_latency_samples += 1
                self.first_token_latency_total_ms += first_token_latency

    def snapshot(self) -> dict[str, float | int]:
        total_turns = max(1, self.turns_total)
        total_tools = max(1, self.tool_calls_total)
        return {
            "turns_total": self.turns_total,
            "turns_done": self.turns_done,
            "turns_cancelled": self.turns_cancelled,
            "turns_error": self.turns_error,
            "task_completion_rate": round(self.turns_done / total_turns, 4),
            "human_interruption_rate": round(self.turns_cancelled / total_turns, 4),
            "tool_calls_total": self.tool_calls_total,
            "tool_failures_total": self.tool_failures_total,
            "tool_failure_rate": round(self.tool_failures_total / total_tools, 4),
            "avg_tool_loop_depth": round(self.tool_loop_depth_total / total_turns, 4),
            "first_token_latency_ms_avg": round(
                self.first_token_latency_total_ms / max(1, self.first_token_latency_samples),
                2,
            ),
            "first_token_latency_samples": self.first_token_latency_samples,
        }
