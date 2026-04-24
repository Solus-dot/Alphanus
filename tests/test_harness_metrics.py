from __future__ import annotations

from agent.harness_metrics import HarnessMetrics
from agent.types import AgentTurnResult


def _result(status: str, journal: dict | None = None) -> AgentTurnResult:
    return AgentTurnResult(
        status=status,
        content="",
        reasoning="",
        skill_exchanges=[],
        journal=journal or {},
    )


def test_harness_metrics_records_completion_failure_and_latency() -> None:
    metrics = HarnessMetrics()
    metrics.record(
        _result(
            "done",
            {
                "tool_counts": {"create_file": 2, "read_file": 1},
                "tool_loop_depth": 3,
                "timing": {"first_token_latency_ms": 120},
                "turn_trace": {
                    "tool_results": [
                        {"result": {"ok": True}},
                        {"result": {"ok": False}},
                    ]
                },
            },
        )
    )
    metrics.record(_result("cancelled"))

    snapshot = metrics.snapshot()
    assert snapshot["turns_total"] == 2
    assert snapshot["turns_done"] == 1
    assert snapshot["turns_cancelled"] == 1
    assert snapshot["tool_calls_total"] == 3
    assert snapshot["tool_failures_total"] == 1
    assert snapshot["task_completion_rate"] == 0.5
    assert snapshot["human_interruption_rate"] == 0.5
    assert snapshot["tool_failure_rate"] == 0.3333
    assert snapshot["avg_tool_loop_depth"] == 1.5
    assert snapshot["first_token_latency_ms_avg"] == 120.0
