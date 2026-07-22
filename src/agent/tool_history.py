from __future__ import annotations

import json
from typing import cast

from core.config_model import AgentConfig
from core.message_types import JsonObject, JSONValue

_WRITE_FIELDS = frozenset(
    "filepath basename created edited changed write_verified sha256 bytes_written chars_written bytes_before bytes_after line_count "
    "line_count_before line_count_after changed_lines edit_mode replacements_applied section_scoped resolved_start_line resolved_end_line "
    "total_line_count_before total_line_count_after content_preview content_preview_truncated preview_chars preview_omitted_chars diff "
    "diff_truncated diff_omitted_chars".split()
)


class ToolHistoryCompactor:
    """Bounds tool payloads retained in model history without changing live results."""

    def __init__(self, config: AgentConfig) -> None:
        self.max_chars = config.max_tool_result_chars
        self.enabled = config.compact_tool_results_in_history
        self.included_tools = {name.strip() for name in config.compact_tool_result_tools if name.strip()}

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
        if limit <= 32:
            return text[:limit], True, len(text) - limit
        text_budget = max(2, limit - 32)
        head_len = max(1, text_budget // 2)
        tail_len = max(1, text_budget - head_len)
        omitted = len(text) - head_len - tail_len
        marker = f"\n...[{omitted} chars truncated]...\n"
        if len(marker) + head_len + tail_len > limit:
            text_budget = max(2, limit - len(marker))
            head_len = max(1, text_budget // 2)
            tail_len = max(1, text_budget - head_len)
            omitted = len(text) - head_len - tail_len
            marker = f"\n...[{omitted} chars truncated]...\n"
        return text[:head_len] + marker + text[-tail_len:], True, omitted

    def compact_json(self, value: object, depth: int = 0, *, string_limit: int | None = None) -> JSONValue:
        limit = self.max_chars if string_limit is None else max(0, int(string_limit))
        if isinstance(value, str):
            return self.truncate_middle(value, limit)[0]
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if depth >= 8:
            if isinstance(value, list):
                return {"__omitted_nested__": True, "type": "list", "item_count": len(value)}
            if isinstance(value, dict):
                return {"__omitted_nested__": True, "type": "dict", "key_count": len(value), "keys": [str(key) for key in list(value)[:20]]}
            return str(value)
        if isinstance(value, list):
            list_output = [self.compact_json(item, depth + 1, string_limit=string_limit) for item in value[:80]]
            if len(value) > 80:
                list_output.append(f"... [{len(value) - 80} more items truncated]")
            return list_output
        if isinstance(value, dict):
            dict_output: JsonObject = {
                str(key): self.compact_json(item, depth + 1, string_limit=string_limit) for key, item in list(value.items())[:120]
            }
            if len(value) > 120:
                dict_output["__truncated_keys__"] = len(value) - 120
            return dict_output
        return str(value)

    def clone_json(self, value: object) -> JSONValue:
        if isinstance(value, list):
            return [self.clone_json(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self.clone_json(item) for key, item in value.items()}
        return value if value is None or isinstance(value, (str, bool, int, float)) else str(value)

    def _metadata(self, value: object) -> JSONValue:
        if not isinstance(value, dict):
            return self.compact_json(value, string_limit=1000)
        output: JsonObject = {}
        for key, item in list(value.items())[:40]:
            if isinstance(item, (str, int, float, bool)) or item is None:
                output[str(key)] = self.compact_json(item, string_limit=1000)
            elif isinstance(item, list):
                children = [
                    self.compact_json(child, string_limit=1000)
                    for child in item[:20]
                    if isinstance(child, (str, int, float, bool)) or child is None
                ]
                if len(item) > 20:
                    children.append(f"... [{len(item) - 20} more items truncated]")
                output[str(key)] = children
            elif isinstance(item, dict):
                output[str(key)] = {
                    str(child_key): self.compact_json(child, string_limit=1000)
                    for child_key, child in list(item.items())[:20]
                    if isinstance(child, (str, int, float, bool)) or child is None
                }
        if len(value) > 40:
            output["__truncated_keys__"] = len(value) - 40
        return output

    def _memory_item(self, item: object) -> JSONValue:
        if not isinstance(item, dict):
            return self.compact_json(item, string_limit=4000)
        keep = {
            "id",
            "text",
            "type",
            "memory_type",
            "score",
            "importance",
            "timestamp",
            "created_at",
            "last_accessed",
            "access_count",
            "metadata",
        }
        output: JsonObject = {}
        for key in keep & item.keys():
            value = item[key]
            if key == "text" and isinstance(value, str):
                text, truncated, omitted = self.truncate_middle(value, 4000)
                output.update({key: text, "text_truncated": truncated})
                if truncated:
                    output["text_omitted_chars"] = omitted
            elif key == "metadata":
                output[key] = self._metadata(value)
            else:
                output[key] = self.compact_json(value, string_limit=1000)
        return output

    def _envelope(self, result: JsonObject, *, compact_data: bool = True) -> JsonObject:
        return {
            key: cast(
                JSONValue,
                self.compact_json(value)
                if key == "data" and compact_data
                else self.clone_json(value)
                if key == "data"
                else self.compact_json(value, string_limit=12000),
            )
            for key, value in result.items()
        }

    def _memory_result(self, result: JsonObject) -> JsonObject:
        output = self._envelope(result, compact_data=False)
        data = output.get("data")
        if not isinstance(data, dict):
            return output
        for key, value in list(data.items()):
            if key not in {"hits", "memories"}:
                data[key] = self.compact_json(value)
        for key in ("hits", "memories"):
            items = data.get(key)
            if isinstance(items, list):
                compacted = [self._memory_item(item) for item in items[:20]]
                if len(items) > 20:
                    compacted.append(f"... [{len(items) - 20} more {'memory hits' if key == 'hits' else 'memories'} truncated]")
                data[key] = compacted
        return output

    def _text_field(self, data: JsonObject, key: str, limit: int) -> None:
        value = data.get(key)
        if not isinstance(value, str):
            return
        text, truncated, omitted = self.truncate_middle(value, limit)
        data[key] = text
        data[f"{key}_truncated"] = bool(data.get(f"{key}_truncated", False) or truncated)
        if truncated:
            data[f"{key}_omitted_chars"] = omitted

    def _data_result(
        self,
        result: JsonObject,
        text_limits: dict[str, int],
        list_fields: dict[str, tuple[int, tuple[str, ...]]] | None = None,
    ) -> JsonObject:
        output = self._envelope(result, compact_data=False)
        data = output.get("data")
        if not isinstance(data, dict):
            return output
        lists = list_fields or {}
        for key, value in list(data.items()):
            if key not in text_limits and key not in lists:
                data[key] = self.compact_json(value)
        for key, limit in text_limits.items():
            self._text_field(data, key, limit)
        for key, (limit, text_fields) in lists.items():
            items = data.get(key)
            if not isinstance(items, list):
                continue
            compacted: list[JSONValue] = []
            for item in items[:limit]:
                row = (
                    {str(name): value if name in text_fields else self.compact_json(value) for name, value in item.items()}
                    if isinstance(item, dict)
                    else self.compact_json(item)
                )
                if isinstance(row, dict):
                    for field in text_fields:
                        self._text_field(row, field, text_limits.get(field, max(text_limits.values())))
                compacted.append(row)
            if len(items) > limit:
                compacted.append(f"... [{len(items) - limit} more {key} truncated]")
            data[key] = compacted
        return output

    def compact_result(self, result: JsonObject) -> JsonObject:
        if self.max_chars <= 0:
            return result
        compacted = self.compact_json(result)
        return cast(JsonObject, compacted) if isinstance(compacted, dict) else {"value": compacted}

    def result(self, tool_name: str, result: JsonObject) -> JsonObject:
        if not self.enabled or self.included_tools and tool_name not in self.included_tools or self.max_chars <= 0:
            return result
        if tool_name in {"recall_memory", "list_memories"}:
            return self._memory_result(result)
        if tool_name in {"read_file", "read_files"}:
            return self._data_result(result, {"content": 64000}, {"files": (40, ("content",))})
        if tool_name in {"create_file", "edit_file"}:
            output = self._envelope(result, compact_data=False)
            data = output.get("data")
            if isinstance(data, dict):
                trimmed = {key: value for key, value in data.items() if key in _WRITE_FIELDS}
                self._text_field(trimmed, "content_preview", 1200)
                self._text_field(trimmed, "diff", 12000)
                output["data"] = trimmed
            return output
        if tool_name == "shell_command":
            return self._data_result(result, {key: 12000 for key in ("stdout", "stderr", "aggregated_output", "output")})
        if tool_name in {"find_files", "search_code", "web_search", "fetch_url", "search_local_files", "retrieve_knowledge"}:
            fields = ("content", "text", "snippet", "summary", "line")
            return self._data_result(result, {key: 4000 for key in fields}, {"results": (40, fields)})
        return self.compact_result(result)

    def arguments(self, args: JsonObject) -> JsonObject:
        output: JsonObject = {}
        for key, value in args.items():
            if not isinstance(value, str):
                output[key] = self.compact_json(value)
            elif len(value) <= 1200:
                output[key] = value
            elif key in {"content", "old_string", "new_string"}:
                output[key] = value[:1200] + f"\n...[history excerpt; {len(value) - 1200} chars omitted]"
            else:
                output[key] = value[:1200] + f"...[truncated {len(value) - 1200} chars]"
        return output
