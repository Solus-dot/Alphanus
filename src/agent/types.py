from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Literal, cast

from core.message_types import ChatMessage, JSONValue, ToolCallDelta, ToolCallUpdate
from core.skill_parser import SkillManifest
from core.skills import SkillContext


@dataclass(slots=True)
class ToolCall:
    stream_id: str
    index: int
    id: str
    name: str
    arguments: dict[str, JSONValue]


@dataclass(slots=True)
class StreamPassResult:
    finish_reason: str
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    first_token_latency_ms: int | None = None


ModelStatusState = Literal["unknown", "online", "offline"]


@dataclass(slots=True)
class ModelStatus:
    state: ModelStatusState = "unknown"
    model_name: str | None = None
    context_window: int | None = None
    last_checked_at: float = 0.0
    last_success_at: float = 0.0
    last_error: str = ""
    endpoint: str = ""

    def is_fresh(
        self,
        *,
        now: float | None = None,
        online_ttl_s: float = 5.0,
        offline_ttl_s: float = 2.0,
    ) -> bool:
        if self.last_checked_at <= 0:
            return False
        if self.state == "online":
            ttl = max(0.0, float(online_ttl_s))
        elif self.state == "offline":
            ttl = max(0.0, float(offline_ttl_s))
        else:
            return False
        current = time.monotonic() if now is None else float(now)
        return (current - self.last_checked_at) <= ttl


@dataclass(slots=True)
class AgentTurnResult:
    status: str
    content: str
    reasoning: str
    skill_exchanges: list[ChatMessage]
    error: str | None = None
    journal: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(slots=True)
class ToolExecutionRecord:
    name: str
    args: dict[str, JSONValue]
    result: dict[str, JSONValue]
    policy_blocked: bool = False


@dataclass(slots=True)
class TurnClassification:
    time_sensitive: bool = False
    requires_workspace_action: bool = False
    prefer_local_workspace_tools: bool = False
    explicit_external_path: str = ""
    followup_kind: str = "new_request"
    used_model: bool = False
    source: str = "fallback"


@dataclass(slots=True)
class TurnPolicySnapshot:
    search_mode: bool = False
    time_sensitive_query: bool = False
    forced_search_retry: bool = False
    requires_workspace_action: bool = False
    forced_action_retry: bool = False
    explicit_external_path: str = ""
    prefer_local_workspace_tools: bool = False
    shell_tool_exposed: bool = False
    collaboration_mode: str = "execute"


@dataclass(slots=True)
class CompletionEvidence:
    tool_counts: dict[str, int] = field(default_factory=dict)
    materialized_paths: list[str] = field(default_factory=list)
    readback_paths: list[str] = field(default_factory=list)
    fetched_urls: set[str] = field(default_factory=set)
    blocked_fetch_domains: set[str] = field(default_factory=set)
    search_failure_count: int = 0
    search_has_success: bool = False
    search_has_fetch_content: bool = False


@dataclass(slots=True)
class TurnTelemetry:
    turn_id: str
    started_at: float = field(default_factory=time.time)
    pass_index: int = 0
    model_usage: dict[str, int] = field(default_factory=dict)
    classification_source: str = ""
    finalization_attempts: int = 0
    finalization_repairs: int = 0
    finalization_fallback_applied: bool = False


@dataclass(slots=True)
class TurnState:
    ctx: SkillContext
    selected: list[SkillManifest]
    dynamic_history: list[ChatMessage]
    skill_exchanges: list[ChatMessage]
    classification: TurnClassification
    completion: CompletionEvidence
    telemetry: TurnTelemetry
    search_tools_enabled: bool = False
    evidence: list[ToolExecutionRecord] = field(default_factory=list)
    full_reasoning: str = ""
    pass_index: int = 0
    action_depth: int = 0
    collaboration_mode: str = "execute"
    forced_search_retry: bool = False
    forced_action_retry: bool = False
    tool_budgets: dict[str, int] = field(default_factory=dict)
    trace_data: dict[str, JSONValue] = field(default_factory=dict)

    @property
    def search_mode(self) -> bool:
        return self.classification.time_sensitive and self.search_tools_enabled

    @property
    def time_sensitive_query(self) -> bool:
        return self.classification.time_sensitive

    @time_sensitive_query.setter
    def time_sensitive_query(self, value: bool) -> None:
        self.classification.time_sensitive = bool(value)

    @property
    def requires_workspace_action(self) -> bool:
        return self.classification.requires_workspace_action

    @property
    def prefer_local_workspace_tools(self) -> bool:
        return self.classification.prefer_local_workspace_tools

    @property
    def explicit_external_path(self) -> str:
        return self.classification.explicit_external_path

    @property
    def tool_counts(self) -> dict[str, int]:
        return self.completion.tool_counts

    @property
    def search_has_success(self) -> bool:
        return self.completion.search_has_success

    @search_has_success.setter
    def search_has_success(self, value: bool) -> None:
        self.completion.search_has_success = bool(value)

    @property
    def search_has_fetch_content(self) -> bool:
        return self.completion.search_has_fetch_content

    @search_has_fetch_content.setter
    def search_has_fetch_content(self, value: bool) -> None:
        self.completion.search_has_fetch_content = bool(value)


@dataclass(slots=True)
class _ToolCallState:
    stream_id: str
    call_id: str
    call_type: str
    name: str
    arguments: str


class ToolCallAccumulator:
    def __init__(self, pass_id: str) -> None:
        self._pass_id = pass_id
        self._items: dict[int, _ToolCallState] = {}

    def ingest(self, deltas: list[ToolCallDelta]) -> list[ToolCallUpdate]:
        updates: list[ToolCallUpdate] = []
        for delta in deltas:
            index = int(delta.get("index", 0))
            item = self._items.setdefault(
                index,
                _ToolCallState(
                    stream_id=f"{self._pass_id}_call_{index}",
                    call_id="",
                    call_type="function",
                    name="",
                    arguments="",
                ),
            )
            delta_id = delta.get("id")
            if delta_id:
                item.call_id = str(delta_id)
            delta_type = delta.get("type")
            if delta_type:
                item.call_type = str(delta_type)
            fn_delta = delta.get("function") or {}
            if isinstance(fn_delta, dict):
                fn_name = fn_delta.get("name")
                if fn_name:
                    item.name += str(fn_name)
                fn_arguments = fn_delta.get("arguments")
                if fn_arguments:
                    item.arguments += str(fn_arguments)
            updates.append(
                {
                    "index": index,
                    "stream_id": item.stream_id,
                    "id": item.call_id,
                    "name": item.name.strip(),
                    "raw_arguments": item.arguments,
                }
            )
        return updates

    def finalize(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for index in sorted(self._items):
            item = self._items[index]
            raw_args = item.arguments
            try:
                loaded = json.loads(raw_args) if raw_args.strip() else {}
                args = loaded if isinstance(loaded, dict) else {"_raw": raw_args}
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            calls.append(
                ToolCall(
                    stream_id=item.stream_id,
                    index=index,
                    id=item.call_id or f"call_{index}",
                    name=item.name.strip(),
                    arguments=cast(dict[str, JSONValue], args),
                )
            )
        return calls
