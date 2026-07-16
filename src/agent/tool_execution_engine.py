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

        if not (state.classification.time_sensitive and state.search_tools_enabled) or call.name not in {"web_search", "fetch_url"}:
            return
        if result.get("ok"):
            data_obj = result.get("data")
            payload = data_obj if isinstance(data_obj, dict) else {}
            attempts = payload.get("attempts")
            if call.name == "web_search" and isinstance(attempts, list):
                for attempt in attempts:
                    if isinstance(attempt, dict):
                        state.completion.search_attempts.append(cast(dict[str, JSONValue], attempt))
                        failure_class = str(attempt.get("failure_class", "")).strip()
                        if failure_class:
                            state.completion.search_failure_classes.append(failure_class)
                if payload.get("failure_class"):
                    state.completion.search_failure_classes.append(str(payload.get("failure_class")))
                if payload.get("evidence_quality") == "none" or not payload.get("results"):
                    state.completion.search_failure_count += 1
                    return
            state.completion.search_has_success = True
            if call.name == "fetch_url":
                fetched_payload = payload
                if bool(fetched_payload.get("usable_text", True)):
                    state.completion.search_has_fetch_content = True
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
            if "private or local network url" in message:
                state.completion.search_failure_classes.append("fetch_blocked")
                if host:
                    state.completion.blocked_fetch_domains.add(host)
            elif host and any(code in message for code in ("http 401", "http 403", "http 429")):
                state.completion.blocked_fetch_domains.add(host)
