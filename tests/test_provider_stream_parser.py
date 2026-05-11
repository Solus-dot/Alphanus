from __future__ import annotations

from agent.provider_stream_parser import ProviderStreamParser
from core.types import ToolCallAccumulator


def test_parse_chat_chunk_extracts_content_reasoning_tool_delta_and_usage() -> None:
    parser = ProviderStreamParser()

    parsed = parser.parse_chat_chunk(
        {
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "thinking",
                        "content": "hello",
                        "tool_calls": [{"index": 0, "function": {"name": "read_file", "arguments": "{}"}}],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        ToolCallAccumulator(pass_id="pass_1"),
    )

    assert parsed["content"] == "hello"
    assert parsed["reasoning"] == "thinking"
    assert parsed["finish_reason"] == "tool_calls"
    assert parsed["usage"] == {"prompt_tokens": 3, "completion_tokens": 2}
    assert parsed["tool_deltas"]


def test_parse_responses_chunk_extracts_function_call_delta() -> None:
    parser = ProviderStreamParser()

    parsed = parser.parse_responses_chunk(
        {
            "type": "response.output_item.added",
            "output_index": 2,
            "item": {"type": "function_call", "call_id": "call_1", "name": "search_code", "arguments": "{\"query\":\"x\"}"},
        },
        ToolCallAccumulator(pass_id="pass_1"),
    )

    assert parsed["tool_deltas"] == [
        {
            "index": 2,
            "id": "call_1",
            "type": "function",
            "function": {"name": "search_code", "arguments": "{\"query\":\"x\"}"},
        }
    ]


def test_parse_responses_completed_extracts_usage_and_stop_reason() -> None:
    parser = ProviderStreamParser()

    parsed = parser.parse_responses_chunk(
        {"type": "response.completed", "response": {"usage": {"input_tokens": 5, "output_tokens": 7}}},
        ToolCallAccumulator(pass_id="pass_1"),
    )

    assert parsed["finish_reason"] == "stop"
    assert parsed["usage"] == {"input_tokens": 5, "output_tokens": 7}
