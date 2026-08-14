from __future__ import annotations

import json
from typing import cast

from core.config_model import AgentConfig
from core.message_types import JsonObject, JSONValue

_WRITE_FIELDS = frozenset(
    "filepath basename created edited changed write_verified sha256 bytes_written chars_written bytes_before bytes_after line_count "
    "line_count_before line_count_after changed_lines edit_mode replacements_applied section_scoped resolved_start_line resolved_end_line "
    "total_line_count_before total_line_count_after diff diff_truncated diff_omitted_chars".split()
)


class ToolHistoryCompactor:
    """Bound model-history payloads without changing live tool results."""

    def __init__(self, config: AgentConfig) -> None:
        self.max_chars = config.max_tool_result_chars

    @staticmethod
    def dumps(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def truncate(text: str, limit: int) -> str:
        if limit <= 0 or len(text) <= limit:
            return text
        return f"{text[:limit]}\n...[truncated {len(text) - limit} chars]"

    @staticmethod
    def truncate_middle(text: str, limit: int) -> tuple[str, bool, int]:
        if limit <= 0 or len(text) <= limit:
            return text, False, 0
        budget = max(2, limit - 32)
        head = max(1, budget // 2)
        tail = max(1, budget - head)
        omitted = len(text) - head - tail
        marker = f"\n...[{omitted} chars truncated]...\n"
        if len(marker) + head + tail > limit:
            budget = max(2, limit - len(marker))
            head = max(1, budget // 2)
            tail = max(1, budget - head)
            omitted = len(text) - head - tail
            marker = f"\n...[{omitted} chars truncated]...\n"
        return text[:head] + marker + text[-tail:], True, omitted

    def compact_json(self, value: object, depth: int = 0, *, string_limit: int | None = None) -> JSONValue:
        limit = self.max_chars if string_limit is None else max(0, int(string_limit))
        if isinstance(value, str):
            return self.truncate_middle(value, limit)[0]
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if depth >= 8:
            return "[nested value omitted]"
        if isinstance(value, list):
            items = [self.compact_json(item, depth + 1, string_limit=limit) for item in value[:80]]
            if len(value) > 80:
                items.append(f"... [{len(value) - 80} more items truncated]")
            return items
        if isinstance(value, dict):
            output: JsonObject = {
                str(key): self.compact_json(item, depth + 1, string_limit=limit) for key, item in list(value.items())[:120]
            }
            if len(value) > 120:
                output["__truncated_keys__"] = len(value) - 120
            return output
        return str(value)

    def _bound_result(self, result: JsonObject) -> JsonObject:
        encoded = self.dumps(result)
        if self.max_chars <= 0 or len(encoded) <= self.max_chars:
            return result
        raw_error = result.get("error")
        error: JsonObject | None = (
            {
                "code": str(raw_error.get("code") or "")[:80],
                "message": self.truncate_middle(str(raw_error.get("message") or ""), 200)[0],
            }
            if isinstance(raw_error, dict)
            else None
        )
        bounded: JsonObject = {
            "ok": bool(result.get("ok")),
            "data": {"history_truncated": True},
            "error": error,
            "meta": {"original_chars": len(encoded)},
        }
        overhead = len(self.dumps(bounded))
        excerpt = self.truncate_middle(encoded, max(0, self.max_chars - overhead - 24))[0]
        data = cast(dict[str, JSONValue], bounded["data"])
        data["result_excerpt"] = excerpt
        while excerpt and len(self.dumps(bounded)) > self.max_chars:
            excerpt = excerpt[: max(0, len(excerpt) - (len(self.dumps(bounded)) - self.max_chars))]
            data["result_excerpt"] = excerpt
        return bounded

    def compact_result(self, result: JsonObject) -> JsonObject:
        compacted = self.compact_json(result)
        output: JsonObject = cast(JsonObject, compacted) if isinstance(compacted, dict) else {"value": compacted}
        return self._bound_result(output)

    def result(self, tool_name: str, result: JsonObject) -> JsonObject:
        if tool_name in {"create_file", "edit_file"}:
            output = cast(JsonObject, self.compact_json(result))
            data = result.get("data")
            if isinstance(data, dict):
                receipt = {key: self.compact_json(value) for key, value in data.items() if key in _WRITE_FIELDS}
                if receipt.get("write_verified"):
                    receipt["write_receipt"] = (
                        "AUTHORITATIVE: the complete tool argument was written and verified; "
                        "content samples are intentionally absent from model history."
                    )
                output["data"] = receipt
            return self._bound_result(output)
        return self.compact_result(result)

    def arguments(self, args: JsonObject) -> JsonObject:
        output: JsonObject = {}
        for key, value in args.items():
            if not isinstance(value, str):
                output[key] = self.compact_json(value)
            elif len(value) <= 1200:
                output[key] = value
            elif key in {"content", "old_string", "new_string"}:
                output[key] = (
                    f"[AUTHORITATIVE HISTORY RECEIPT: complete {key} argument contained {len(value)} characters "
                    "when the tool ran; hidden here only to save context]"
                )
            else:
                output[key] = value[:1200] + f"...[truncated {len(value) - 1200} chars]"
        if self.max_chars > 0 and len(self.dumps(output)) > self.max_chars:
            return {"_history_truncated": True, "original_chars": len(self.dumps(output))}
        return output
