from __future__ import annotations

from core.types import ToolCallAccumulator


class ProviderStreamParser:
    @staticmethod
    def normalize_usage(payload: object) -> dict[str, int]:
        if not isinstance(payload, dict):
            return {}
        usage: dict[str, int] = {}
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                usage[str(key)] = int(value)
        return usage

    @staticmethod
    def _coerce_index(*values: object) -> int:
        for value in values:
            if isinstance(value, bool) or value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            try:
                return max(0, int(text))
            except (TypeError, ValueError):
                continue
        return 0

    def parse_chat_chunk(self, chunk: dict[str, object], _tool_acc: ToolCallAccumulator) -> dict[str, object]:
        out: dict[str, object] = {
            "content": "",
            "reasoning": "",
            "tool_deltas": [],
            "finish_reason": "",
            "usage": {},
        }
        choices = chunk.get("choices", [])
        chunk_usage = self.normalize_usage(chunk.get("usage"))
        if chunk_usage:
            out["usage"] = chunk_usage
        if not isinstance(choices, list) or not choices:
            message = chunk.get("message")
            if isinstance(message, dict):
                out["content"] = str(message.get("content") or "")
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    out["tool_deltas"] = tool_calls
            if bool(chunk.get("done")):
                out["finish_reason"] = str(chunk.get("done_reason") or "stop")
            return out
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta", {}) if isinstance(choice.get("delta"), dict) else {}
        message = choice.get("message", {}) if isinstance(choice.get("message"), dict) else {}
        out["reasoning"] = str(delta.get("reasoning_content") or message.get("reasoning_content") or "")
        out["content"] = str(delta.get("content") or message.get("content") or "")
        tool_deltas = delta.get("tool_calls")
        if not isinstance(tool_deltas, list):
            tool_deltas = message.get("tool_calls")
        if isinstance(tool_deltas, list):
            out["tool_deltas"] = tool_deltas
        if choice.get("finish_reason"):
            out["finish_reason"] = str(choice["finish_reason"])
        return out

    def parse_responses_chunk(self, chunk: dict[str, object], tool_acc: ToolCallAccumulator) -> dict[str, object]:
        out: dict[str, object] = {
            "content": "",
            "reasoning": "",
            "tool_deltas": [],
            "finish_reason": "",
            "usage": {},
        }
        event_type = str(chunk.get("type") or "").strip()
        if not event_type:
            return out

        if event_type in {"response.output_text.delta", "response.output_text.annotation.added"}:
            out["content"] = str(chunk.get("delta") or "")
            return out

        if event_type in {
            "response.reasoning.delta",
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        }:
            out["reasoning"] = str(chunk.get("delta") or "")
            return out

        if event_type == "response.output_item.added":
            raw_item = chunk.get("item")
            item = raw_item if isinstance(raw_item, dict) else {}
            item_type = str(item.get("type") or "")
            if item_type in {"function_call", "tool_call"}:
                output_index = chunk.get("output_index")
                item_index = item.get("index")
                index = self._coerce_index(output_index, item_index)
                function_name = str(item.get("name") or item.get("function_name") or "").strip()
                call_id = str(item.get("call_id") or item.get("id") or "").strip()
                args = str(item.get("arguments") or "")
                out["tool_deltas"] = [
                    {
                        "index": index,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": function_name, "arguments": args},
                    }
                ]
            return out

        if event_type == "response.function_call_arguments.delta":
            output_index = chunk.get("output_index")
            index = self._coerce_index(output_index)
            call_id = str(chunk.get("call_id") or chunk.get("item_id") or "").strip()
            delta = str(chunk.get("delta") or "")
            out["tool_deltas"] = [
                {
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {"arguments": delta},
                }
            ]
            return out

        if event_type in {"response.completed", "response.incomplete", "response.failed"}:
            raw_response = chunk.get("response")
            response_obj = raw_response if isinstance(raw_response, dict) else {}
            usage = self.normalize_usage(response_obj.get("usage"))
            if usage:
                out["usage"] = usage
            if event_type == "response.completed":
                out["finish_reason"] = "stop"
            elif event_type == "response.incomplete":
                out["finish_reason"] = "length"
            else:
                out["finish_reason"] = "error"
            return out

        if event_type == "response.output_item.done":
            raw_item = chunk.get("item")
            item = raw_item if isinstance(raw_item, dict) else {}
            if str(item.get("type") or "") in {"function_call", "tool_call"}:
                output_index = chunk.get("output_index")
                item_index = item.get("index")
                index = self._coerce_index(output_index, item_index)
                call_id = str(item.get("call_id") or item.get("id") or "").strip()
                args = str(item.get("arguments") or "")
                if args:
                    out["tool_deltas"] = [
                        {
                            "index": index,
                            "id": call_id,
                            "type": "function",
                            "function": {"arguments": args},
                        }
                    ]
                if tool_acc.finalize():
                    out["finish_reason"] = "tool_calls"
            return out

        return out
