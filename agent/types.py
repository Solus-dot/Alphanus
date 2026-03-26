from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.skills import SkillContext


@dataclass(slots=True)
class ToolCall:
    stream_id: str
    index: int
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass(slots=True)
class StreamPassResult:
    finish_reason: str
    content: str = ""
    reasoning: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class AgentTurnResult:
    status: str
    content: str
    reasoning: str
    skill_exchanges: List[Dict[str, Any]]
    error: Optional[str] = None
    journal: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolExecutionRecord:
    name: str
    args: Dict[str, Any]
    result: Dict[str, Any]
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
    selected_shell_workflow_skills: List[str] = field(default_factory=list)
    requested_opaque_artifact_extensions: List[str] = field(default_factory=list)
    has_selected_materializers: bool = False


@dataclass(slots=True)
class CompletionEvidence:
    tool_counts: Dict[str, int] = field(default_factory=dict)
    materialized_paths: List[str] = field(default_factory=list)
    readback_paths: List[str] = field(default_factory=list)
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
    model_usage: Dict[str, int] = field(default_factory=dict)
    classification_source: str = ""


@dataclass(slots=True)
class BackgroundSkillAgentTask:
    task_id: str
    agent_name: str
    prompt: str
    skill_id: str = ""
    status: str = "running"
    output: str = ""
    error: str = ""
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0


@dataclass(slots=True)
class TurnState:
    ctx: SkillContext
    selected: List[Any]
    dynamic_history: List[Dict[str, Any]]
    skill_exchanges: List[Dict[str, Any]]
    classification: TurnClassification
    completion: CompletionEvidence
    telemetry: TurnTelemetry
    evidence: List[ToolExecutionRecord] = field(default_factory=list)
    full_reasoning: str = ""
    pass_index: int = 0
    action_depth: int = 0
    forced_search_retry: bool = False
    forced_action_retry: bool = False
    tool_budgets: Dict[str, int] = field(default_factory=dict)
    _search_mode_override: Optional[bool] = None

    @property
    def search_mode(self) -> bool:
        if self._search_mode_override is not None:
            return self._search_mode_override
        return self.classification.time_sensitive and any(getattr(skill, "id", "") == "search-ops" for skill in self.selected)

    @search_mode.setter
    def search_mode(self, value: bool) -> None:
        self._search_mode_override = bool(value)

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
    def tool_counts(self) -> Dict[str, int]:
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


class ToolCallAccumulator:
    def __init__(self, pass_id: str) -> None:
        self._pass_id = pass_id
        self._items: Dict[int, Dict[str, Any]] = {}

    def ingest(self, deltas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        updates: List[Dict[str, Any]] = []
        for delta in deltas:
            index = int(delta.get("index", 0))
            item = self._items.setdefault(
                index,
                {
                    "stream_id": f"{self._pass_id}_call_{index}",
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if delta.get("id"):
                item["id"] = delta["id"]
            if delta.get("type"):
                item["type"] = delta["type"]
            fn_delta = delta.get("function", {})
            if fn_delta.get("name"):
                item["function"]["name"] += fn_delta["name"]
            if fn_delta.get("arguments"):
                item["function"]["arguments"] += fn_delta["arguments"]
            updates.append(
                {
                    "index": index,
                    "stream_id": item.get("stream_id", f"{self._pass_id}_call_{index}"),
                    "id": item.get("id", ""),
                    "name": item["function"].get("name", "").strip(),
                    "raw_arguments": item["function"].get("arguments", ""),
                }
            )
        return updates

    def finalize(self) -> List[ToolCall]:
        calls: List[ToolCall] = []
        for index in sorted(self._items):
            item = self._items[index]
            raw_args = item["function"].get("arguments", "")
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            calls.append(
                ToolCall(
                    stream_id=item.get("stream_id", f"{self._pass_id}_call_{index}"),
                    index=index,
                    id=item.get("id") or f"call_{index}",
                    name=item["function"].get("name", "").strip(),
                    arguments=args,
                )
            )
        return calls
