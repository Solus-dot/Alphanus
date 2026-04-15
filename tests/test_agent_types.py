from __future__ import annotations

from core.types import ModelStatus, ToolCallAccumulator


def test_tool_call_accumulator_reconstructs_streamed_tool_call() -> None:
    acc = ToolCallAccumulator("turn_1")

    acc.ingest(
        [
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "create_file", "arguments": '{"filepath": "a.txt"'},
            }
        ]
    )
    acc.ingest(
        [
            {
                "index": 0,
                "function": {"arguments": ', "content": "hello"}'},
            }
        ]
    )

    calls = acc.finalize()

    assert len(calls) == 1
    assert calls[0].name == "create_file"
    assert calls[0].id == "call_1"
    assert calls[0].arguments == {"filepath": "a.txt", "content": "hello"}


def test_tool_call_accumulator_preserves_raw_arguments_when_json_is_invalid() -> None:
    acc = ToolCallAccumulator("turn_2")
    acc.ingest(
        [
            {
                "index": 0,
                "function": {"name": "edit_file", "arguments": '{"filepath": "broken"'},
            }
        ]
    )

    calls = acc.finalize()

    assert len(calls) == 1
    assert calls[0].arguments == {"_raw": '{"filepath": "broken"'}


def test_model_status_only_reports_fresh_online_or_offline_states() -> None:
    online = ModelStatus(state="online", last_checked_at=100.0)
    offline = ModelStatus(state="offline", last_checked_at=100.0)
    unknown = ModelStatus(state="unknown", last_checked_at=100.0)

    assert online.is_fresh(now=102.0, online_ttl_s=5.0)
    assert offline.is_fresh(now=101.5, offline_ttl_s=2.0)
    assert not unknown.is_fresh(now=101.0)
