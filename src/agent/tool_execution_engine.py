from __future__ import annotations

import urllib.parse
from typing import cast

from core.message_types import JSONValue
from core.types import ToolCall, ToolExecutionRecord, TurnState


class ToolExecutionEngine:
    @staticmethod
    def tool_result_paths(name: str, payload: dict[str, object]) -> list[str]:
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            return []
        if name in {"create_file", "edit_file", "create_directory", "read_file"}:
            path = str(data.get("filepath", "")).strip()
            return [path] if path else []
        if name == "read_files":
            created = data.get("created") or data.get("files")
            if not isinstance(created, list):
                return []
            out: list[str] = []
            for item in created:
                if not isinstance(item, dict):
                    continue
                path = str(item.get("filepath", "")).strip()
                if path:
                    out.append(path)
            return out
        return []

    def record_tool_effects(self, state: TurnState, call: ToolCall, result: dict[str, object], *, policy_blocked: bool = False) -> None:
        result_json = cast(dict[str, JSONValue], result)
        state.completion.tool_counts[call.name] = state.completion.tool_counts.get(call.name, 0) + 1
        state.evidence.append(
            ToolExecutionRecord(
                name=call.name,
                args=dict(call.arguments),
                result=result_json,
                policy_blocked=policy_blocked,
            )
        )
        if result.get("ok"):
            paths = self.tool_result_paths(call.name, result)
            if call.name in {"create_file", "edit_file"}:
                for path in paths:
                    if path and path not in state.completion.materialized_paths:
                        state.completion.materialized_paths.append(path)
            if call.name in {"read_file", "read_files"}:
                for path in paths:
                    if path and path not in state.completion.readback_paths:
                        state.completion.readback_paths.append(path)

        if not state.search_mode or call.name not in {"web_search", "fetch_url"}:
            return
        if result.get("ok"):
            state.completion.search_has_success = True
            if call.name == "fetch_url":
                state.completion.search_has_fetch_content = True
                data_obj = result.get("data")
                fetched_payload = data_obj if isinstance(data_obj, dict) else {}
                for key in ("url", "final_url"):
                    seen_url = str(fetched_payload.get(key, "")).strip()
                    if seen_url:
                        state.completion.fetched_urls.add(seen_url)
            return
        if call.name == "web_search":
            state.completion.search_failure_count += 1
        if call.name == "fetch_url":
            error_raw = result.get("error")
            error_obj = error_raw if isinstance(error_raw, dict) else {}
            message = str(error_obj.get("message", "")).lower()
            raw_url = str(call.arguments.get("url", "")).strip()
            host = urllib.parse.urlparse(raw_url).netloc.lower()
            if host and any(code in message for code in ("http 401", "http 403", "http 429")):
                state.completion.blocked_fetch_domains.add(host)
