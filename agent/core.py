from __future__ import annotations

import json
import os
import re
import copy
import threading
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.context import ContextWindowManager
from agent.prompts import build_system_prompt
from core.configuration import validate_endpoint_policy
from core.skills import SkillContext, SkillRuntime
from core.streaming import build_ssl_context, should_retry, stream_chat_completions


@dataclass
class ToolCall:
    stream_id: str
    index: int
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class StreamPassResult:
    finish_reason: str
    content: str = ""
    reasoning: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class AgentTurnResult:
    status: str  # done | cancelled | error
    content: str
    reasoning: str
    skill_exchanges: List[Dict[str, Any]]
    error: Optional[str] = None
    journal: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolEvidence:
    name: str
    args: Dict[str, Any]
    result: Dict[str, Any]


@dataclass(slots=True)
class SkillRoutingSnapshot:
    generation: int
    skills: List[Any]
    catalog: str


@dataclass(slots=True)
class TurnState:
    ctx: SkillContext
    selected: List[Any]
    loaded_skill_ids: List[str]
    dynamic_history: List[Dict[str, Any]]
    skill_exchanges: List[Dict[str, Any]]
    evidence: List[ToolEvidence]
    full_reasoning: str = ""
    pass_index: int = 0
    action_depth: int = 0
    forced_search_retry: bool = False
    forced_action_retry: bool = False
    search_mode: bool = False
    time_sensitive_query: bool = False
    search_failure_count: int = 0
    search_has_success: bool = False
    search_has_fetch_content: bool = False
    blocked_fetch_domains: set[str] = field(default_factory=set)
    fetched_urls: set[str] = field(default_factory=set)
    tool_counts: Dict[str, int] = field(default_factory=dict)
    tool_budgets: Dict[str, int] = field(default_factory=dict)
    requires_workspace_action: bool = False
    batch_workspace_action: bool = False
    prefer_local_workspace_tools: bool = False
    explicit_external_path: str = ""
    workspace_scaffold_action: bool = False
    workspace_materialization_target: int = 0
    forced_workspace_retry: bool = False
    model_usage: Dict[str, int] = field(default_factory=dict)


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


class Agent:
    def __init__(
        self,
        config: Dict[str, Any],
        skill_runtime: SkillRuntime,
        debug: bool = False,
    ) -> None:
        self.skill_runtime = skill_runtime
        self.debug = debug
        self.auth_header = (
            os.environ.get("ALPHANUS_AUTH_HEADER", "").strip()
            or os.environ.get("AUTH_HEADER", "").strip()
            or None
        )
        self.connect_timeout_s = 10
        self.per_turn_retries = 1
        self.retry_backoff_s = 0.5
        self.system_prompt = build_system_prompt(self.skill_runtime.workspace.workspace_root)

        context_cfg = config.get("context", {})
        self.context_mgr = ContextWindowManager(
            context_limit=int(context_cfg.get("context_limit", 8192)),
            keep_last_n=int(context_cfg.get("keep_last_n", 10)),
            safety_margin=int(context_cfg.get("safety_margin", 500)),
        )

        self._ready_checked = False
        self._skill_snapshot: Optional[SkillRoutingSnapshot] = None
        self._bg_skill_agent_tasks: Dict[str, BackgroundSkillAgentTask] = {}
        self._bg_skill_agent_lock = threading.Lock()
        self.reload_config(config)

    def reload_config(self, config: Dict[str, Any]) -> None:
        self.config = config
        context_cfg = config.get("context", {})
        self.context_mgr = ContextWindowManager(
            context_limit=int(context_cfg.get("context_limit", 8192)),
            keep_last_n=int(context_cfg.get("keep_last_n", 10)),
            safety_margin=int(context_cfg.get("safety_margin", 500)),
        )

        agent_cfg = config.get("agent", {})
        self.model_endpoint = agent_cfg.get("model_endpoint", "http://127.0.0.1:8080/v1/chat/completions")
        self.models_endpoint = agent_cfg.get("models_endpoint", "http://127.0.0.1:8080/v1/models")
        self.tls_verify = bool(agent_cfg.get("tls_verify", True))
        self.ca_bundle_path = agent_cfg.get("ca_bundle_path")
        self.allow_cross_host = bool(agent_cfg.get("allow_cross_host_endpoints", False))
        self.request_timeout_s = float(agent_cfg.get("request_timeout_s", 180))
        self.readiness_timeout_s = float(agent_cfg.get("readiness_timeout_s", 30))
        self.readiness_poll_s = float(agent_cfg.get("readiness_poll_s", 0.5))

        raw_max_tokens = agent_cfg.get("max_tokens")
        if raw_max_tokens in (None, "", 0):
            self.default_max_tokens = None
        else:
            value = int(raw_max_tokens)
            self.default_max_tokens = value if value > 0 else None
        self.context_budget_max_tokens = int(
            agent_cfg.get("context_budget_max_tokens", self.default_max_tokens or 1024)
        )

        self.max_action_depth = int(agent_cfg.get("max_action_depth", 10))
        self.max_tool_result_chars = int(agent_cfg.get("max_tool_result_chars", 12000))
        self.max_reasoning_chars = max(0, int(agent_cfg.get("max_reasoning_chars", 20000)))
        self.compact_tool_results_in_history = bool(agent_cfg.get("compact_tool_results_in_history", False))
        compact_tools = agent_cfg.get("compact_tool_result_tools", [])
        if isinstance(compact_tools, list):
            self.compact_tool_result_tools = {str(name).strip() for name in compact_tools if str(name).strip()}
        else:
            self.compact_tool_result_tools = set()
        self.debug_log_path = str(agent_cfg.get("debug_log_path", "")).strip()
        self.ssl_context = build_ssl_context(self.tls_verify, self.ca_bundle_path)

        budgets = agent_cfg.get("tool_budgets", {})
        self.default_tool_budgets = {
            "web_search": 2,
            "fetch_url": 2,
            "recall_memory": 2,
        }
        if isinstance(budgets, dict):
            for key, value in budgets.items():
                try:
                    self.default_tool_budgets[str(key)] = max(1, int(value))
                except Exception:
                    continue

        self._ready_checked = False

    def _headers(self) -> Dict[str, str]:
        headers = {}
        if self.auth_header:
            if ":" in self.auth_header:
                key, value = self.auth_header.split(":", 1)
                headers[key.strip()] = value.strip()
        return headers

    def _emit(self, on_event: Optional[Callable[[Dict[str, Any]], None]], event: Dict[str, Any]) -> None:
        if not on_event:
            return
        try:
            on_event(event)
        except Exception:
            return

    @staticmethod
    def _debug_compact(value: Any, depth: int = 0) -> Any:
        if depth >= 8:
            return "[truncated]"
        if isinstance(value, str):
            if value.startswith("data:image/"):
                prefix, _, encoded = value.partition(",")
                return f"{prefix},...[{len(encoded)} base64 chars]"
            if len(value) <= 4000:
                return value
            return value[:4000] + f"...[truncated {len(value) - 4000} chars]"
        if isinstance(value, list):
            items = [Agent._debug_compact(item, depth + 1) for item in value[:50]]
            if len(value) > 50:
                items.append(f"...[{len(value) - 50} more items]")
            return items
        if isinstance(value, dict):
            items = list(value.items())
            out: Dict[str, Any] = {}
            for key, item in items[:80]:
                out[str(key)] = Agent._debug_compact(item, depth + 1)
            if len(items) > 80:
                out["__truncated_keys__"] = len(items) - 80
            return out
        return value

    def _debug_log(self, event_type: str, **payload: Any) -> None:
        if not self.debug or not self.debug_log_path:
            return
        try:
            path = Path(self.debug_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event_type,
                "payload": self._debug_compact(payload),
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            return

    @staticmethod
    def _stop_requested(stop_event) -> bool:
        return bool(stop_event is not None and stop_event.is_set())

    @staticmethod
    def _sleep_with_stop(duration_s: float, stop_event) -> bool:
        if duration_s <= 0:
            return not Agent._stop_requested(stop_event)
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            if Agent._stop_requested(stop_event):
                return False
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        return not Agent._stop_requested(stop_event)

    def ensure_ready(self, stop_event=None, on_event: Optional[Callable[[Dict[str, Any]], None]] = None, timeout_s: Optional[float] = None) -> Optional[bool]:
        timeout = self.readiness_timeout_s if timeout_s is None else max(0.0, float(timeout_s))
        deadline = time.monotonic() + timeout
        headers = self._headers()
        req = urllib.request.Request(self.models_endpoint, headers=headers, method="GET")
        attempt_timeout = min(self.connect_timeout_s, 1.0)
        self._debug_log(
            "readiness_start",
            endpoint=self.models_endpoint,
            timeout_s=timeout,
            header_keys=sorted(headers.keys()),
        )
        self._emit(on_event, {"type": "info", "text": f"waiting for endpoint handshake: {self.models_endpoint}"})

        while time.monotonic() < deadline:
            if self._stop_requested(stop_event):
                self._ready_checked = False
                self._debug_log("readiness_cancelled", endpoint=self.models_endpoint)
                return None
            try:
                with urllib.request.urlopen(req, timeout=attempt_timeout, context=self.ssl_context) as resp:
                    if 200 <= resp.status < 300:
                        self._ready_checked = True
                        self._debug_log("readiness_ok", endpoint=self.models_endpoint, status=resp.status)
                        return True
            except Exception as exc:
                self._debug_log("readiness_retry", endpoint=self.models_endpoint, error=str(exc))
                if not self._sleep_with_stop(self.readiness_poll_s, stop_event):
                    self._ready_checked = False
                    self._debug_log("readiness_cancelled", endpoint=self.models_endpoint)
                    return None
                continue
        self._ready_checked = False
        self._debug_log("readiness_failed", endpoint=self.models_endpoint)
        return False

    @staticmethod
    def _extract_model_name(payload: Any) -> Optional[str]:
        def _candidate(value: Any) -> Optional[str]:
            text = str(value or "").strip()
            return text or None

        def _from_item(item: Any) -> Optional[str]:
            if isinstance(item, dict):
                for key in ("id", "name", "model"):
                    candidate = _candidate(item.get(key))
                    if candidate:
                        return candidate
            return _candidate(item)

        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list):
                for item in data:
                    candidate = _from_item(item)
                    if candidate:
                        return candidate
            elif data is not None:
                candidate = _from_item(data)
                if candidate:
                    return candidate
            return _from_item(payload)

        if isinstance(payload, list):
            for item in payload:
                candidate = _from_item(item)
                if candidate:
                    return candidate
        return None

    @staticmethod
    def _extract_model_context_window(payload: Any) -> Optional[int]:
        context_keys = (
            "context_length",
            "context_window",
            "max_context_length",
            "max_model_len",
            "num_ctx",
            "n_ctx",
            "n_ctx_slot",
            "n_ctx_train",
        )
        visited: set[int] = set()

        def _candidate_int(value: Any) -> Optional[int]:
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return value if value > 0 else None
            if isinstance(value, float):
                parsed = int(value)
                return parsed if parsed > 0 else None
            if isinstance(value, str):
                try:
                    parsed = int(value.strip())
                    return parsed if parsed > 0 else None
                except Exception:
                    return None
            return None

        def _from_item(item: Any) -> Optional[int]:
            if isinstance(item, dict):
                marker = id(item)
                if marker in visited:
                    return None
                visited.add(marker)
                for key in context_keys:
                    candidate = _candidate_int(item.get(key))
                    if candidate is not None:
                        return candidate
                for value in item.values():
                    candidate = _from_item(value)
                    if candidate is not None:
                        return candidate
                return None
            if isinstance(item, list):
                marker = id(item)
                if marker in visited:
                    return None
                visited.add(marker)
                for value in item:
                    candidate = _from_item(value)
                    if candidate is not None:
                        return candidate
                return None
            return None

        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list):
                for item in data:
                    candidate = _from_item(item)
                    if candidate is not None:
                        return candidate
            elif data is not None:
                candidate = _from_item(data)
                if candidate is not None:
                    return candidate
            return _from_item(payload)

        if isinstance(payload, list):
            for item in payload:
                candidate = _from_item(item)
                if candidate is not None:
                    return candidate
        return None

    @staticmethod
    def _props_endpoint_from_models_endpoint(models_endpoint: str) -> str:
        parsed = urllib.parse.urlparse(models_endpoint)
        path = parsed.path or ""
        if path.endswith("/v1/models"):
            path = path[: -len("/v1/models")] + "/props"
        elif path.endswith("/models"):
            path = path[: -len("/models")] + "/props"
        else:
            path = "/props"
        return urllib.parse.urlunparse(parsed._replace(path=path, params="", query="", fragment=""))

    def _fetch_json(self, url: str, timeout_s: Optional[float] = None) -> Any:
        headers = self._headers()
        request = urllib.request.Request(url, headers=headers, method="GET")
        timeout = self.connect_timeout_s if timeout_s is None else max(0.1, float(timeout_s))
        with urllib.request.urlopen(request, timeout=timeout, context=self.ssl_context) as response:
            return json.loads(response.read().decode("utf-8"))

    def fetch_model_metadata(self, timeout_s: Optional[float] = None) -> tuple[Optional[str], Optional[int]]:
        try:
            payload = self._fetch_json(self.models_endpoint, timeout_s=timeout_s)
        except Exception as exc:
            self._debug_log("model_fetch_failed", endpoint=self.models_endpoint, error=str(exc))
            return None, None
        model_name = self._extract_model_name(payload)
        context_window = self._extract_model_context_window(payload)
        if context_window is not None:
            return model_name, context_window
        props_endpoint = self._props_endpoint_from_models_endpoint(self.models_endpoint)
        try:
            props_payload = self._fetch_json(props_endpoint, timeout_s=timeout_s)
        except Exception as exc:
            self._debug_log("model_props_fetch_failed", endpoint=props_endpoint, error=str(exc))
            return model_name, None
        return model_name, self._extract_model_context_window(props_payload)

    def fetch_model_name(self, timeout_s: Optional[float] = None) -> Optional[str]:
        model_name, _context_window = self.fetch_model_metadata(timeout_s=timeout_s)
        return model_name

    def _validate_endpoints(self) -> Optional[str]:
        try:
            validate_endpoint_policy(
                {
                    "agent": {
                        "model_endpoint": self.model_endpoint,
                        "models_endpoint": self.models_endpoint,
                        "allow_cross_host_endpoints": self.allow_cross_host,
                    }
                }
            )
        except ValueError as exc:
            return str(exc)
        return None

    def doctor_report(self) -> Dict[str, Any]:
        endpoint_error = self._validate_endpoints()
        workspace_root = Path(self.skill_runtime.workspace.workspace_root)
        memory_stats = self.skill_runtime.memory.stats()
        search_cfg = self.config.get("search", {}) if isinstance(self.config, dict) else {}
        provider = str(search_cfg.get("provider", "tavily")).strip().lower() or "tavily"
        provider_env = {
            "tavily": "TAVILY_API_KEY",
            "brave": "BRAVE_SEARCH_API_KEY",
        }
        required_env = provider_env.get(provider, "")
        search_ready = bool(os.environ.get(required_env, "").strip()) if required_env else False
        ready = self.ensure_ready(timeout_s=min(self.readiness_timeout_s, 3.0))
        return {
            "agent": {
                "model_endpoint": self.model_endpoint,
                "models_endpoint": self.models_endpoint,
                "ready": bool(ready),
                "endpoint_policy_error": endpoint_error or "",
                "auth_header_source": "env" if self.auth_header else "none",
            },
            "workspace": {
                "path": str(workspace_root),
                "exists": workspace_root.exists(),
                "writable": os.access(workspace_root, os.W_OK),
            },
            "memory": {
                "backend": memory_stats.get("embedding_backend"),
                "configured_backend": memory_stats.get("configured_embedding_backend"),
                "allow_model_download": memory_stats.get("allow_model_download"),
                "encoder_status": memory_stats.get("encoder_status"),
                "encoder_source": memory_stats.get("encoder_source"),
                "encoder_detail": memory_stats.get("encoder_detail"),
                "mode": memory_stats.get("mode_label"),
                "model_name": memory_stats.get("model_name"),
                "recommended_model_name": memory_stats.get("recommended_model_name"),
                "dimension": memory_stats.get("dimension"),
                "count": memory_stats.get("count"),
            },
            "search": {
                "provider": provider,
                "ready": search_ready,
                "reason": "" if search_ready or not required_env else f"missing env: {required_env}",
            },
            "skills": self.skill_runtime.skill_health_report(),
        }

    def build_support_bundle(self, tree_payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "doctor": self.doctor_report(),
            "tree": tree_payload,
        }

    def _build_skill_context(
        self,
        user_input: str,
        branch_labels: List[str],
        attachments: List[str],
        history_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> SkillContext:
        explicit_skill_id = ""
        explicit_skill_args = ""
        lowered = user_input.strip()
        match = re.match(r"(?is)^\s*use\s+skill\s+([a-z0-9][a-z0-9-]{0,63})(?::|\s+|$)(.*)$", lowered)
        if match:
            explicit_skill_id = str(match.group(1) or "").strip()
            explicit_skill_args = str(match.group(2) or "").strip()
        hits = self.skill_runtime.memory.search(user_input, top_k=3, min_score=0.45)
        recent_hint, sticky_skill_ids = self._recent_routing_context(history_messages or [])
        return SkillContext(
            user_input=user_input,
            branch_labels=branch_labels,
            attachments=attachments,
            workspace_root=str(self.skill_runtime.workspace.workspace_root),
            memory_hits=hits,
            recent_routing_hint=recent_hint,
            sticky_skill_ids=sticky_skill_ids,
            explicit_skill_id=explicit_skill_id,
            explicit_skill_args=explicit_skill_args,
        )

    @staticmethod
    def _message_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")).strip())
            return "\n".join(part for part in parts if part).strip()
        return str(value or "").strip()

    @staticmethod
    def _is_confirmation_like(text: str) -> bool:
        lowered = " ".join(text.strip().lower().split())
        if not lowered:
            return False
        direct = {
            "yes",
            "yeah",
            "yep",
            "ok",
            "okay",
            "sure",
            "do it",
            "go ahead",
            "continue",
            "proceed",
            "run it",
            "do that",
            "delete them",
            "delete it",
            "all of them",
        }
        if lowered in direct:
            return True
        if len(lowered.split()) <= 3 and lowered in {"please do", "please continue", "yes please"}:
            return True
        affirmative_prefixes = ("yes ", "yeah ", "yep ", "ok ", "okay ", "sure ")
        if len(lowered.split()) <= 6 and any(lowered.startswith(prefix) for prefix in affirmative_prefixes):
            return True
        return False

    def _recent_routing_context(self, history_messages: List[Dict[str, Any]]) -> tuple[str, List[str]]:
        if not history_messages:
            return "", []
        last_user_index = -1
        for idx in range(len(history_messages) - 1, -1, -1):
            if str(history_messages[idx].get("role", "")).lower() == "user":
                last_user_index = idx
                break
        if last_user_index < 0:
            return "", []
        recent = history_messages[last_user_index:]
        recent_user = self._message_text(recent[0].get("content", ""))
        tool_names: List[str] = []
        assistant_text = ""
        seen_tools = set()
        sticky_skill_ids: List[str] = []
        seen_skills = set()
        registry = getattr(self.skill_runtime, "_tool_registry", {})
        for msg in recent[1:]:
            role = str(msg.get("role", "")).lower()
            if role == "assistant":
                text = self._message_text(msg.get("content", ""))
                if text:
                    assistant_text = text
                for call in msg.get("tool_calls", []) or []:
                    name = str(((call or {}).get("function") or {}).get("name", "")).strip()
                    if not name or name in seen_tools:
                        continue
                    tool_names.append(name)
                    seen_tools.add(name)
                    reg = registry.get(name)
                    if reg and reg.skill_id not in seen_skills:
                        sticky_skill_ids.append(reg.skill_id)
                        seen_skills.add(reg.skill_id)
            elif role == "tool":
                name = str(msg.get("name", "")).strip()
                if not name or name in seen_tools:
                    continue
                tool_names.append(name)
                seen_tools.add(name)
                reg = registry.get(name)
                if reg and reg.skill_id not in seen_skills:
                    sticky_skill_ids.append(reg.skill_id)
                    seen_skills.add(reg.skill_id)
                try:
                    payload = json.loads(self._message_text(msg.get("content", "")) or "{}")
                except Exception:
                    payload = {}
                data = payload.get("data") if isinstance(payload, dict) else {}
                loaded_skill_id = str(data.get("skill_id", "")).strip() if isinstance(data, dict) else ""
                if loaded_skill_id and loaded_skill_id not in seen_skills:
                    sticky_skill_ids.append(loaded_skill_id)
                    seen_skills.add(loaded_skill_id)
        parts: List[str] = []
        if recent_user:
            parts.append(f"previous user request: {recent_user}")
        if assistant_text:
            compact_assistant = " ".join(assistant_text.split())
            if len(compact_assistant) > 240:
                compact_assistant = compact_assistant[:237].rstrip() + "..."
            parts.append(f"assistant just said: {compact_assistant}")
        if tool_names:
            parts.append(f"tools just used: {', '.join(tool_names[:4])}")
        return "\n".join(parts), sticky_skill_ids[:3]

    def _get_skill_snapshot(self, force_refresh: bool = False) -> SkillRoutingSnapshot:
        generation = int(getattr(self.skill_runtime, "generation", 0))
        if not force_refresh and self._skill_snapshot and self._skill_snapshot.generation == generation:
            return self._skill_snapshot
        skills = list(self.skill_runtime.enabled_skills())
        catalog = self.skill_runtime.skill_cards_text(skills)
        self._skill_snapshot = SkillRoutingSnapshot(generation=generation, skills=skills, catalog=catalog)
        return self._skill_snapshot

    def reload_skills(self) -> int:
        self.skill_runtime.load_skills()
        self._skill_snapshot = None
        return int(getattr(self.skill_runtime, "generation", 0))

    def _compose_system_content(self, selected: List[Any], ctx: SkillContext, loaded_skill_ids: Optional[List[str]] = None) -> str:
        loaded_skill_ids = loaded_skill_ids or []
        parts = [self.system_prompt]
        unloaded = [skill for skill in selected if getattr(skill, "id", "") not in set(loaded_skill_ids)]
        if unloaded:
            cards = self.skill_runtime.skill_cards_text(unloaded)
            if cards:
                parts.append(
                    "Available skills:\n"
                    "- These are compact routing cards. Use `load_skill` before relying on full skill instructions or bundled resources.\n"
                    f"{cards}"
                )
        loaded = self.skill_runtime.loaded_skills(selected, loaded_skill_ids)
        if loaded:
            skill_block = self.skill_runtime.compose_skill_block(
                loaded,
                ctx,
                context_limit=self.context_mgr.context_limit,
            )
            if skill_block:
                parts.append(f"Active skill guidance:\n\n{skill_block}")
        return "\n\n".join(part for part in parts if part.strip())

    @staticmethod
    def _parse_skill_route(content: str, valid_ids: List[str], limit: int) -> Optional[List[str]]:
        text = content.strip()
        if not text:
            return None
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        parsed_ids: List[str] = []
        parsed_explicitly = False
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                raw = data.get("skills")
            else:
                raw = data
            if isinstance(raw, list):
                parsed_explicitly = True
                parsed_ids = [str(item).strip() for item in raw]
            elif raw == []:
                parsed_explicitly = True
        except Exception:
            parsed_ids = []

        lowered = text.lower()
        if parsed_explicitly and not parsed_ids:
            return []

        if lowered in {"none", "no skills", "[]", "{\"skills\": []}", "{\"skills\":[]}"}:
            return []

        if not parsed_ids:
            parsed_ids = [skill_id for skill_id in valid_ids if skill_id.lower() in lowered]

        out: List[str] = []
        seen = set()
        valid = set(valid_ids)
        for skill_id in parsed_ids:
            if skill_id in valid and skill_id not in seen:
                out.append(skill_id)
                seen.add(skill_id)
            if len(out) >= limit:
                break
        return out or None

    def _route_skills_with_model(
        self,
        ctx: SkillContext,
        snapshot: SkillRoutingSnapshot,
        stop_event,
    ) -> Optional[List[Any]]:
        skills_cfg = self.config.get("skills", {}) if isinstance(self.config, dict) else {}
        configured_limit = int(skills_cfg.get("max_active_skills", getattr(self.skill_runtime, "max_active_skills", 2)))
        limit = configured_limit if configured_limit > 0 else 2
        candidate_items = self.skill_runtime.propose_skill_candidates(ctx, top_n=max(limit * 3, 6))
        enabled = snapshot.skills
        if not enabled:
            return []

        heuristic_fallback = self.skill_runtime.rerank_skill_candidates(ctx, candidate_items, limit)
        use_recent_hint = self._should_use_recent_routing_hint(ctx)
        routing_system = (
            "You are selecting local skills for the next assistant turn.\n"
            "Choose the smallest set of skills needed for the user's request.\n"
            "Use the skill descriptions, tags, produced artifact hints, and available tool names to infer intent.\n"
            f"Return strict JSON only in the form {{\"skills\": [\"skill-id\"]}} with at most {limit} ids.\n"
            "Return an empty list when no skill is needed.\n"
            "Do not explain your choice."
        )
        if use_recent_hint:
            routing_system += (
                "\nFor short confirmations like yes/continue/do it, use only the immediate prior exchange summary below."
                "\nDo not infer tools from older conversation context."
            )
        user_content = f"User request:\n{ctx.user_input}"
        if use_recent_hint and ctx.recent_routing_hint:
            user_content += f"\n\nImmediate prior exchange:\n{ctx.recent_routing_hint}"
        if use_recent_hint and ctx.sticky_skill_ids:
            user_content += f"\n\nLikely carry-over skills for this confirmation:\n{', '.join(ctx.sticky_skill_ids[:limit])}"
        shortlisted_skills = [item.skill for item in candidate_items[: min(len(candidate_items), max(limit * 2, self.skill_runtime.shortlist_size))]]
        if shortlisted_skills:
            user_content += (
                "\n\nSkill shortlist:\n"
                + self.skill_runtime.skill_cards_text(shortlisted_skills)
            )
        elif snapshot.catalog:
            user_content += f"\n\nAvailable skills:\n{snapshot.catalog}"
        routing_messages = [
            {"role": "system", "content": routing_system},
            {"role": "user", "content": user_content},
        ]
        payload = self._build_payload(routing_messages, thinking=False, tools=None)
        try:
            result = self._call_with_retry(payload, stop_event, None, pass_id="skill_route")
        except Exception:
            if heuristic_fallback:
                return heuristic_fallback
            return self.skill_runtime.skills_by_ids(ctx.sticky_skill_ids[:limit]) if use_recent_hint and ctx.sticky_skill_ids else None
        if result.finish_reason == "cancelled":
            return None
        skill_ids = self._parse_skill_route(result.content, [skill.id for skill in enabled], limit)
        if skill_ids is None:
            if heuristic_fallback:
                return heuristic_fallback
            return self.skill_runtime.skills_by_ids(ctx.sticky_skill_ids[:limit]) if use_recent_hint and ctx.sticky_skill_ids else None
        return self.skill_runtime.expand_selected_skills(ctx, self.skill_runtime.skills_by_ids(skill_ids))

    def _prefer_deterministic_skill_selection(self, ctx: SkillContext) -> bool:
        if ctx.explicit_skill_id:
            return True
        ranked = [(score, skill) for score, skill in self.skill_runtime.score_skills(ctx) if score > 0]
        if len(ranked) <= 1:
            return True
        top_score = ranked[0][0]
        second_score = ranked[1][0]
        if top_score >= 60 and top_score - second_score >= 18:
            return True
        if top_score >= 90 and second_score <= 24:
            return True
        return False

    def _select_skills(
        self,
        ctx: SkillContext,
        stop_event,
    ) -> List[Any]:
        skills_cfg = self.config.get("skills", {}) if isinstance(self.config, dict) else {}
        mode = str(skills_cfg.get("selection_mode", getattr(self.skill_runtime, "selection_mode", ""))).strip().lower()
        if mode in {"model", "hybrid_lazy"}:
            if mode == "hybrid_lazy" and self._prefer_deterministic_skill_selection(ctx):
                return self.skill_runtime.select_skills(ctx)
            snapshot = self._get_skill_snapshot()
            routed = self._route_skills_with_model(ctx, snapshot, stop_event)
            if routed is not None:
                if routed:
                    return routed
                fallback = self._contextual_skill_fallback(ctx)
                if fallback:
                    return fallback
                return routed
        return self.skill_runtime.select_skills(ctx)

    def _contextual_skill_fallback(self, ctx: SkillContext) -> List[Any]:
        if not self._should_use_recent_routing_hint(ctx):
            return []
        merged_text = "\n".join(part for part in (ctx.user_input, ctx.recent_routing_hint) if part).strip()
        if not merged_text:
            return []
        merged_ctx = SkillContext(
            user_input=merged_text,
            branch_labels=list(ctx.branch_labels),
            attachments=list(ctx.attachments),
            workspace_root=ctx.workspace_root,
            memory_hits=list(ctx.memory_hits),
            recent_routing_hint=ctx.recent_routing_hint,
            sticky_skill_ids=list(ctx.sticky_skill_ids),
        )
        return self.skill_runtime.select_skills(merged_ctx)

    def _build_payload(
        self,
        model_messages: List[Dict[str, Any]],
        thinking: bool,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "messages": model_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        }
        if self.default_max_tokens is not None:
            payload["max_tokens"] = self.default_max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    @staticmethod
    def _safe_json_dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        if limit <= 0 or len(text) <= limit:
            return text
        clipped = text[:limit]
        remaining = len(text) - limit
        return f"{clipped}\n...[truncated {remaining} chars]"

    def _compact_jsonish(self, value: Any, depth: int = 0) -> Any:
        if depth >= 4:
            return "[truncated]"
        if isinstance(value, str):
            return self._truncate_text(value, self.max_tool_result_chars)
        if isinstance(value, list):
            max_items = 80
            out = [self._compact_jsonish(item, depth + 1) for item in value[:max_items]]
            if len(value) > max_items:
                out.append(f"... [{len(value) - max_items} more items truncated]")
            return out
        if isinstance(value, dict):
            max_keys = 120
            out: Dict[str, Any] = {}
            items = list(value.items())
            for key, item in items[:max_keys]:
                out[str(key)] = self._compact_jsonish(item, depth + 1)
            if len(items) > max_keys:
                out["__truncated_keys__"] = len(items) - max_keys
            return out
        return value

    def _compact_tool_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if self.max_tool_result_chars <= 0:
            return result
        compacted = self._compact_jsonish(result)
        if isinstance(compacted, dict):
            return compacted
        return {"value": compacted}

    def _tool_result_for_history(self, tool_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        if not self.compact_tool_results_in_history:
            return result
        if self.compact_tool_result_tools and tool_name not in self.compact_tool_result_tools:
            return result
        return self._compact_tool_result(result)

    def _tool_call_args_for_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(args, dict):
            return {}
        out: Dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str):
                if len(value) > 1200:
                    out[key] = value[:1200] + f"...[truncated {len(value) - 1200} chars]"
                else:
                    out[key] = value
                continue
            out[key] = self._compact_jsonish(value)
        return out

    def _append_reasoning(self, full_reasoning: str, delta_reasoning: str) -> str:
        if not delta_reasoning:
            return full_reasoning
        delta_reasoning = self._sanitize_reasoning_markup(delta_reasoning)
        if not delta_reasoning:
            return full_reasoning
        if self.max_reasoning_chars <= 0:
            return ""
        if len(full_reasoning) >= self.max_reasoning_chars:
            return full_reasoning
        remaining = self.max_reasoning_chars - len(full_reasoning)
        if len(delta_reasoning) <= remaining:
            return full_reasoning + delta_reasoning
        return full_reasoning + delta_reasoning[:remaining]

    @staticmethod
    def _sanitize_reasoning_markup(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"</?tool_call>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"</?function(?:=[^>]+)?>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"</?parameter(?:=[^>]+)?>", "", text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _sanitize_final_content(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
        kept: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if re.fullmatch(r"</?think>", stripped, flags=re.IGNORECASE):
                continue
            if re.fullmatch(r"</?tool_call>", stripped, flags=re.IGNORECASE):
                continue
            if re.fullmatch(r"<function=[^>]+>", stripped, flags=re.IGNORECASE):
                continue
            if re.fullmatch(r"</function>", stripped, flags=re.IGNORECASE):
                continue
            if re.fullmatch(r"<parameter=[^>]+>", stripped, flags=re.IGNORECASE):
                continue
            if re.fullmatch(r"</parameter>", stripped, flags=re.IGNORECASE):
                continue
            kept.append(line)
        deduped: List[str] = []
        previous = None
        for line in kept:
            stripped = line.strip()
            if stripped and stripped == previous:
                continue
            deduped.append(line)
            previous = stripped if stripped else previous
        return "\n".join(deduped).strip()

    @staticmethod
    def _contains_tool_markup(text: str) -> bool:
        if not text:
            return False
        return bool(
            re.search(
                r"<tool_call\b|</tool_call>|<function=[^>]+>|</function>|<parameter=[^>]+>|</parameter>",
                text,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _fallback_answer_from_evidence(evidence: List[ToolEvidence]) -> str:
        search_results: List[Dict[str, str]] = []
        fetched_sources: List[Dict[str, str]] = []
        fetch_errors: List[str] = []
        generic_successes: List[Dict[str, Any]] = []

        for item in evidence:
            name = item.name
            payload = item.result
            if not payload.get("ok"):
                if name == "fetch_url":
                    msg = str(payload.get("error", {}).get("message", "")).strip()
                    if msg:
                        fetch_errors.append(msg)
                continue
            data = payload.get("data")
            if name == "web_search" and isinstance(data, dict):
                for item in data.get("results", [])[:5]:
                    if not isinstance(item, dict):
                        continue
                    title = str(item.get("title", "")).strip()
                    url = str(item.get("url", "")).strip()
                    domain = str(item.get("domain", "")).strip()
                    if title and url:
                        search_results.append({"title": title, "url": url, "domain": domain})
            elif name == "fetch_url" and isinstance(data, dict):
                title = str(data.get("title", "")).strip()
                final_url = str(data.get("final_url", "")).strip() or str(data.get("url", "")).strip()
                if final_url:
                    fetched_sources.append({"title": title or final_url, "url": final_url})
            else:
                generic_successes.append({"name": name, "data": data})

        if fetched_sources:
            lines = [
                "I found relevant sources, but the model failed to produce a clean final summary.",
                "Verified fetched sources:",
            ]
            for idx, item in enumerate(fetched_sources[:3], start=1):
                lines.append(f"{idx}. {item['title']} — {item['url']}")
            if fetch_errors:
                lines.append(f"Some pages were blocked or failed to load ({fetch_errors[0]}).")
            return "\n".join(lines)

        if search_results:
            lines = [
                "I found relevant search results, but the model failed to produce a clean final summary.",
                "Relevant sources to check:",
            ]
            seen = set()
            count = 0
            for item in search_results:
                key = item["url"]
                if key in seen:
                    continue
                seen.add(key)
                domain = f" ({item['domain']})" if item.get("domain") else ""
                lines.append(f"{count + 1}. {item['title']}{domain} — {item['url']}")
                count += 1
                if count >= 3:
                    break
            if fetch_errors:
                lines.append(f"Some fetched pages were blocked or failed to load ({fetch_errors[0]}).")
            return "\n".join(lines)

        if fetch_errors:
            return (
                "I couldn't produce a verified answer because the fetched sources failed to load cleanly "
                f"({fetch_errors[0]}), and the model did not return a usable final response."
            )

        if generic_successes:
            latest = generic_successes[-1]
            name = str(latest.get("name", "tool")).strip() or "tool"
            data = latest.get("data")
            if isinstance(data, dict) and data:
                preview_items = []
                for key, value in list(data.items())[:6]:
                    rendered = str(value)
                    if len(rendered) > 160:
                        rendered = rendered[:160] + "..."
                    preview_items.append(f"- {key}: {rendered}")
                if preview_items:
                    return (
                        f"The `{name}` tool completed successfully, but the model failed to produce a clean final summary.\n\n"
                        "Tool result:\n"
                        + "\n".join(preview_items)
                    )
            rendered = str(data)
            if len(rendered) > 400:
                rendered = rendered[:400] + "..."
            return (
                f"The `{name}` tool completed successfully, but the model failed to produce a clean final summary.\n\n"
                f"Tool result: {rendered}"
            )

        return ""

    @staticmethod
    def _evidence_from_history(dynamic_history: List[Dict[str, Any]]) -> List[ToolEvidence]:
        out: List[ToolEvidence] = []
        for message in dynamic_history:
            if message.get("role") != "tool":
                continue
            try:
                payload = json.loads(str(message.get("content", "{}")))
            except Exception:
                continue
            out.append(
                ToolEvidence(
                    name=str(message.get("name", "")).strip(),
                    args={},
                    result=payload if isinstance(payload, dict) else {},
                )
            )
        return out

    @staticmethod
    def _unverified_search_answer() -> str:
        return (
            "I couldn't verify the answer from reliable web results in this turn, and I don't want to speculate "
            "without confirmed sources."
        )

    @staticmethod
    def _generic_finalization_failure_answer() -> str:
        return (
            "I couldn't turn the available tool and model output into a clean user-facing answer in this turn."
        )

    @staticmethod
    def _workspace_action_failure_answer() -> str:
        return (
            "I couldn't complete the requested workspace action in this turn because the model failed to use the "
            "available workspace tools."
        )

    @staticmethod
    def _search_fallback_allowed(state: TurnState) -> bool:
        return (
            state.tool_counts.get("web_search", 0) > 0
            or state.tool_counts.get("fetch_url", 0) > 0
            or state.search_failure_count > 0
            or state.search_has_success
            or state.search_has_fetch_content
        )

    @staticmethod
    def _needs_fetch_evidence(state: TurnState) -> bool:
        return state.search_mode and state.time_sensitive_query and not state.search_has_fetch_content

    @staticmethod
    def _looks_like_manual_workspace_advice(text: str) -> bool:
        lowered = text.lower()
        return (
            "can't" in lowered
            or "cannot" in lowered
            or "manual" in lowered
            or "terminal" in lowered
            or "run the following command" in lowered
            or "rm " in lowered
            or "rm -rf" in lowered
        )

    @staticmethod
    def _looks_like_workspace_action_claim(text: str) -> bool:
        lowered = text.lower()
        completed = (
            "i deleted" in lowered
            or "deleted the following" in lowered
            or "workspace is now empty" in lowered
            or "i created" in lowered
            or "i updated" in lowered
            or "i renamed" in lowered
            or "successfully deleted" in lowered
        )
        return completed

    @staticmethod
    def _is_time_sensitive_query(text: str) -> bool:
        lowered = text.lower()
        phrases = (
            "latest",
            "recent",
            "current",
            "today",
            "right now",
            "up to date",
            "current situation",
            "news",
            "this week",
            "this month",
            "as of",
        )
        return any(phrase in lowered for phrase in phrases)

    def _run_finalization_pass(
        self,
        system_content: str,
        state: TurnState,
        stop_event,
        on_event: Optional[Callable[[Dict[str, Any]], None]],
        pass_id: str,
        extra_rules: str = "",
    ) -> AgentTurnResult:
        def finalize_once(extra: str, suffix: str):
            finalize_system = (
                system_content
                + "\n\nFinalization rule:\n"
                + "- Provide only the final user-facing answer using prior tool results.\n"
                + "- Do not call tools.\n"
                + "- Do not include hidden reasoning.\n"
                + "- Never emit XML, HTML-like tool markup, or <tool_call> blocks."
            )
            if extra:
                finalize_system += "\n" + extra.strip()
            finalize_messages = [{"role": "system", "content": finalize_system}] + state.dynamic_history
            finalize_messages = self.context_mgr.prune(finalize_messages, self.context_budget_max_tokens)
            finalize_payload = self._build_payload(finalize_messages, thinking=False, tools=None)
            try:
                return self._call_with_retry(
                    finalize_payload,
                    stop_event,
                    None,
                    pass_id=f"{pass_id}_{suffix}",
                )
            except Exception as exc:
                message = str(exc)
                self._emit(on_event, {"type": "error", "text": message})
                return AgentTurnResult(
                    status="error",
                    content="",
                    reasoning=state.full_reasoning,
                    skill_exchanges=state.skill_exchanges,
                    error=message,
                )

        def coerce_result(stream_result, current_reasoning: str) -> AgentTurnResult:
            if stream_result.finish_reason == "cancelled":
                return AgentTurnResult(
                    status="cancelled",
                    content="",
                    reasoning=self._append_reasoning(current_reasoning, stream_result.reasoning),
                    skill_exchanges=state.skill_exchanges,
                )
            if stream_result.finish_reason == "tool_calls":
                return AgentTurnResult(
                    status="error",
                    content="",
                    reasoning=self._append_reasoning(current_reasoning, stream_result.reasoning),
                    skill_exchanges=state.skill_exchanges,
                    error="Finalization pass unexpectedly returned tool calls",
                )
            cleaned = self._sanitize_final_content(stream_result.content)
            return AgentTurnResult(
                status="done",
                content=cleaned,
                reasoning=self._append_reasoning(current_reasoning, stream_result.reasoning),
                skill_exchanges=state.skill_exchanges,
            )

        first = finalize_once(extra_rules, "final")
        if isinstance(first, AgentTurnResult):
            return first
        first_result = coerce_result(first, state.full_reasoning)
        if first_result.status != "done":
            return first_result
        leaked_markup = "<tool_call>" in first.content.lower()
        if not leaked_markup:
            return first_result
        fallback = self._fallback_answer_from_evidence(state.evidence or self._evidence_from_history(state.dynamic_history))
        if fallback:
            return AgentTurnResult(
                status="done",
                content=fallback,
                reasoning=first_result.reasoning,
                skill_exchanges=state.skill_exchanges,
            )
        if not self._search_fallback_allowed(state):
            return AgentTurnResult(
                status="done",
                content=self._generic_finalization_failure_answer(),
                reasoning=first_result.reasoning,
                skill_exchanges=state.skill_exchanges,
            )
        return AgentTurnResult(
            status="done",
            content=self._unverified_search_answer(),
            reasoning=first_result.reasoning,
            skill_exchanges=state.skill_exchanges,
        )

    @staticmethod
    def _is_search_skill_selected(selected: List[Any]) -> bool:
        return any(getattr(skill, "id", "") == "search-ops" for skill in selected)

    def _stream_one_pass(
        self,
        payload: Dict[str, Any],
        stop_event,
        on_event: Optional[Callable[[Dict[str, Any]], None]],
        pass_id: str,
    ) -> StreamPassResult:
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        finish_reason = "stop"
        tool_acc = ToolCallAccumulator(pass_id=pass_id)
        tool_phase_started = False
        suppress_content_stream = False
        usage: Dict[str, int] = {}
        self._debug_log("chat_pass_start", pass_id=pass_id, endpoint=self.model_endpoint, payload=payload)

        for chunk in stream_chat_completions(
            endpoint=self.model_endpoint,
            payload=payload,
            timeout_s=self.request_timeout_s,
            headers=self._headers(),
            ssl_context=self.ssl_context,
            stop_event=stop_event,
            on_debug_event=lambda event: self._debug_log("http_stream", pass_id=pass_id, **event),
        ):
            if stop_event is not None and stop_event.is_set():
                self._debug_log("chat_pass_cancelled", pass_id=pass_id)
                return StreamPassResult(finish_reason="cancelled")
            choices = chunk.get("choices", [])
            chunk_usage = chunk.get("usage") or {}
            if isinstance(chunk_usage, dict) and chunk_usage:
                usage = {
                    str(key): int(value)
                    for key, value in chunk_usage.items()
                    if isinstance(value, (int, float))
                }
                if usage:
                    self._emit(on_event, {"type": "usage", "usage": dict(usage)})
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {}) or {}

            reasoning = delta.get("reasoning_content") or ""
            if reasoning:
                reasoning_parts.append(reasoning)
                self._emit(on_event, {"type": "reasoning_token", "text": reasoning})

            content = delta.get("content") or ""
            if content:
                content_parts.append(content)
                if self._contains_tool_markup(content):
                    suppress_content_stream = True
                if not suppress_content_stream:
                    self._emit(on_event, {"type": "content_token", "text": content})

            tool_deltas = delta.get("tool_calls") or []
            if tool_deltas:
                if not tool_phase_started:
                    tool_phase_started = True
                    self._emit(on_event, {"type": "tool_phase_started"})
                updates = tool_acc.ingest(tool_deltas)
                for update in updates:
                    self._emit(
                        on_event,
                        {
                            "type": "tool_call_delta",
                            "stream_id": update["stream_id"],
                            "id": update.get("id") or "",
                            "name": update.get("name") or "",
                            "raw_arguments": update.get("raw_arguments") or "",
                        },
                    )

            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])

        tool_calls = tool_acc.finalize()
        if tool_calls and finish_reason not in {"tool_calls", "cancelled"}:
            finish_reason = "tool_calls"

        self._debug_log(
            "chat_pass_end",
            pass_id=pass_id,
            finish_reason=finish_reason,
            content_chars=len("".join(content_parts)),
            reasoning_chars=len("".join(reasoning_parts)),
            tool_call_count=len(tool_calls),
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            tool_calls=[
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                for call in tool_calls
            ],
        )
        return StreamPassResult(
            finish_reason=finish_reason,
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            tool_calls=tool_calls,
            usage=usage,
        )

    def _call_with_retry(self, payload: Dict[str, Any], stop_event, on_event, pass_id: str):
        attempt = 0
        while True:
            try:
                return self._stream_one_pass(payload, stop_event, on_event, pass_id=pass_id)
            except Exception as exc:
                retryable = should_retry(exc)
                if retryable and attempt < self.per_turn_retries:
                    attempt += 1
                    self._emit(on_event, {"type": "info", "text": f"Retrying request ({attempt}/{self.per_turn_retries})..."})
                    if not self._sleep_with_stop(self.retry_backoff_s, stop_event):
                        return StreamPassResult(finish_reason="cancelled")
                    ready = self.ensure_ready(
                        stop_event=stop_event,
                        on_event=on_event,
                        timeout_s=min(self.readiness_timeout_s, 5.0),
                    )
                    if ready is None:
                        return StreamPassResult(finish_reason="cancelled")
                    if not ready:
                        raise
                    continue
                raise

    @staticmethod
    def _search_rule(*lines: str) -> str:
        return "Search completion rule:\n" + "\n".join(f"- {line}" for line in lines)

    def _build_turn_state(
        self,
        ctx: SkillContext,
        selected: List[Any],
        history_messages: List[Dict[str, Any]],
        user_input: str,
    ) -> TurnState:
        explicit_external_path = self._explicit_path_outside_workspace(ctx.user_input)
        loaded_skill_ids = self.skill_runtime.default_loaded_skill_ids(selected, ctx)
        selection_mode = str(getattr(self.skill_runtime, "selection_mode", "")).strip().lower()
        auto_load_ids: List[str] = []
        explicit = self.skill_runtime.resolve_skill_reference(ctx.explicit_skill_id) if ctx.explicit_skill_id else None
        if explicit is not None:
            auto_load_ids.append(explicit.id)
        elif selection_mode == "hybrid_lazy":
            non_core_selected = [
                getattr(skill, "id", "")
                for skill in selected
                if getattr(skill, "id", "")
                and getattr(skill, "id", "") not in {"workspace-ops", "search-ops", "shell-ops", "memory-rag", "utilities"}
                and not bool(getattr(skill, "disable_model_invocation", False))
            ]
            if len(non_core_selected) == 1:
                auto_load_ids.append(non_core_selected[0])
        loaded_skill_ids = sorted(dict.fromkeys([skill_id for skill_id in loaded_skill_ids + auto_load_ids if skill_id]))
        return TurnState(
            ctx=ctx,
            selected=selected,
            loaded_skill_ids=loaded_skill_ids,
            dynamic_history=list(history_messages),
            skill_exchanges=[],
            evidence=[],
            search_mode=self._is_search_skill_selected(selected),
            time_sensitive_query=self._is_time_sensitive_query(user_input),
            requires_workspace_action=self._requires_workspace_action(ctx, selected),
            batch_workspace_action=self._is_batch_workspace_action(ctx, selected),
            prefer_local_workspace_tools=self._prefers_local_workspace_tools(ctx, selected),
            explicit_external_path=explicit_external_path,
            workspace_scaffold_action=self._is_workspace_scaffold_action(ctx, selected),
            workspace_materialization_target=self._workspace_materialization_target(ctx, selected),
            tool_counts={},
            tool_budgets=dict(self.default_tool_budgets),
        )

    @staticmethod
    def _is_workspace_skill_selected(selected: List[Any]) -> bool:
        return any(getattr(skill, "id", "") == "workspace-ops" for skill in selected)

    @classmethod
    def _is_contextual_followup_like(cls, text: str) -> bool:
        lowered = " ".join(text.strip().lower().split())
        if not lowered or cls._is_confirmation_like(lowered):
            return False
        words = lowered.split()
        if len(words) > 6:
            return False
        prefixes = (
            "where is",
            "where's",
            "what about",
            "and ",
            "also ",
            "add ",
            "include ",
            "make ",
            "put ",
            "now ",
        )
        if any(lowered.startswith(prefix) for prefix in prefixes):
            return True
        return any(token in lowered for token in (" js", " css", " html", "script", "style", "that", "it"))

    @classmethod
    def _should_use_recent_routing_hint(cls, ctx: SkillContext) -> bool:
        if not (ctx.recent_routing_hint or ctx.sticky_skill_ids):
            return False
        return cls._is_confirmation_like(ctx.user_input) or cls._is_contextual_followup_like(ctx.user_input)

    def _requires_workspace_action(self, ctx: SkillContext, selected: List[Any]) -> bool:
        if not self._is_workspace_skill_selected(selected):
            return False
        if not self._is_confirmation_like(ctx.user_input):
            return False
        hint = (ctx.recent_routing_hint or "").lower()
        if not hint:
            return False
        action_markers = (
            "delete ",
            "remove ",
            "edit ",
            "write ",
            "create ",
            "rename ",
            "workspace",
            "file",
            "folder",
            "directory",
        )
        return any(marker in hint for marker in action_markers)

    def _prefers_local_workspace_tools(self, ctx: SkillContext, selected: List[Any]) -> bool:
        if not self._is_workspace_skill_selected(selected):
            return False
        if self._explicit_path_outside_workspace(ctx.user_input):
            return False
        text = " ".join(part for part in (ctx.user_input, ctx.recent_routing_hint) if part).lower()
        if not text:
            return False
        explicit_shell = any(term in text for term in ("shell", "terminal", "bash", "zsh", "cmd ", "powershell", "run this command"))
        explicit_web = any(term in text for term in ("http://", "https://", "website", "web ", "search ", "google ", "fetch "))
        if explicit_shell or explicit_web:
            return False
        local_markers = (
            "workspace",
            "folder",
            "directory",
            "file",
            "files",
            ".py",
            "python",
            "program",
            "package",
            "module",
            "html",
            "css",
            "js",
            "javascript",
            "script",
            "style",
            "landing page",
            "scaffold",
            "component",
        )
        return any(marker in text for marker in local_markers)

    def _explicit_path_outside_workspace(self, text: str) -> str:
        workspace_root = Path(self.skill_runtime.workspace.workspace_root)
        seen: set[str] = set()
        path_pattern = re.compile(
            r'(?P<quoted>(?P<quote>["\'`])(?P<quoted_path>(?:~/|/)[^"\'`]+?)(?P=quote))'
            r"|(?P<plain>(?<![:/\w])(?P<plain_path>(?:~/|/)[^\s\"'`]+))"
        )
        for match in path_pattern.finditer(text or ""):
            raw = match.group("quoted_path") or match.group("plain_path") or ""
            cleaned = raw if match.group("quoted_path") else raw.rstrip(".,:;!?)]}")
            expanded = Path(os.path.expanduser(cleaned))
            if not expanded.is_absolute():
                continue
            resolved = expanded.resolve(strict=False)
            resolved_str = str(resolved)
            if resolved_str in seen:
                continue
            seen.add(resolved_str)
            try:
                resolved.relative_to(workspace_root)
            except ValueError:
                return resolved_str
        return ""

    def _is_batch_workspace_action(self, ctx: SkillContext, selected: List[Any]) -> bool:
        if not self._is_workspace_skill_selected(selected):
            return False
        text = " ".join(part for part in (ctx.user_input, ctx.recent_routing_hint) if part).lower()
        if not text:
            return False
        destructive = any(term in text for term in ("delete", "remove", "wipe", "clear"))
        broad_scope = (
            "all files" in text
            or "all the files" in text
            or "everything" in text
            or "all contents" in text
            or "entire workspace" in text
            or "whole workspace" in text
            or ("workspace" in text and any(term in text for term in ("all", "everything", "files", "contents")))
        )
        return destructive and broad_scope

    def _is_workspace_scaffold_action(self, ctx: SkillContext, selected: List[Any]) -> bool:
        if not self._is_workspace_skill_selected(selected):
            return False
        text = " ".join(part for part in (ctx.user_input, ctx.recent_routing_hint) if part).lower()
        if not text:
            return False
        build_terms = ("make ", "create ", "build ", "generate ", "save it in", "save them in")
        artifact_terms = (
            "landing page",
            "html",
            "css",
            "javascript",
            "js",
            "python",
            ".py",
            "program",
            "package",
            "module",
            "website",
            "web page",
            "page",
            "scaffold",
        )
        multi_file_terms = (
            "folder",
            "directory",
            "files",
            " file",
            ".py",
            "main.py",
            "html css",
            "html, css",
            "css and javascript",
            "html css and javascript",
        )
        return any(term in text for term in build_terms) and any(term in text for term in artifact_terms) and any(
            term in text for term in multi_file_terms
        )

    def _workspace_materialization_target(self, ctx: SkillContext, selected: List[Any]) -> int:
        if not self._is_workspace_skill_selected(selected):
            return 0
        if self._is_confirmation_like(ctx.user_input):
            text = (ctx.recent_routing_hint or ctx.user_input).lower()
        else:
            text = ctx.user_input.lower()
        if not text:
            return 0
        if not any(term in text for term in ("make ", "create ", "build ", "generate ", "write ", "save it in", "save them in")):
            return 0

        wants_dir = any(term in text for term in ("folder", "directory"))
        wants_file = any(
            term in text
            for term in (" file", "files", ".py", ".js", ".html", ".css", "python", "html", "css", "javascript", "js", "script", "code", "program")
        )

        requested_files = 0
        if "html" in text:
            requested_files += 1
        if "css" in text:
            requested_files += 1
        if "javascript" in text or " js " in f" {text} " or "script.js" in text:
            requested_files += 1
        if requested_files == 0 and any(term in text for term in ("landing page", "website", "web page")):
            requested_files = 1
        if requested_files == 0 and wants_file:
            requested_files = 1
        if "files" in text and requested_files < 2:
            requested_files = 2
        if wants_dir and requested_files == 0:
            return 0
        return requested_files

    @staticmethod
    def _workspace_materialization_count(state: TurnState) -> int:
        count = 0
        for item in state.evidence:
            if not isinstance(item.result, dict) or not item.result.get("ok"):
                continue
            data = item.result.get("data")
            if item.name == "create_file":
                count += 1
            elif item.name == "create_files" and isinstance(data, dict):
                created_count = data.get("count")
                if isinstance(created_count, int):
                    count += created_count
                else:
                    created = data.get("created")
                    if isinstance(created, list):
                        count += len(created)
            elif item.name == "edit_file":
                count += 1
        return count

    @staticmethod
    def _tool_result_paths(name: str, payload: Dict[str, Any]) -> List[str]:
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            return []
        if name in {"create_file", "edit_file", "create_directory"}:
            path = str(data.get("filepath", "")).strip()
            return [path] if path else []
        if name == "create_files":
            created = data.get("created")
            if not isinstance(created, list):
                return []
            out: List[str] = []
            for item in created:
                if not isinstance(item, dict):
                    continue
                path = str(item.get("filepath", "")).strip()
                if path:
                    out.append(path)
            return out
        return []

    def _direct_tool_completion_allowed(self, state: TurnState, call: ToolCall, result: Dict[str, Any]) -> bool:
        if call.name not in {"create_directory", "create_file", "create_files", "edit_file"}:
            return True
        requested_exts = {
            str(ext).strip().lower()
            for ext in self.skill_runtime.requested_opaque_artifact_extensions(state.ctx)
            if str(ext).strip()
        }
        if not requested_exts:
            return True
        for path in self._tool_result_paths(call.name, result):
            if Path(path).suffix.lower() in requested_exts:
                return True
        return False

    def _finalize_turn(
        self,
        system_content: str,
        state: TurnState,
        stop_event,
        on_event,
        pass_id: str,
        extra_rules: str = "",
    ) -> AgentTurnResult:
        return self._run_finalization_pass(
            system_content,
            state,
            stop_event,
            on_event,
            pass_id,
            extra_rules=extra_rules,
        )

    def _tool_budget_reason(self, state: TurnState, call: ToolCall) -> str:
        limit = state.tool_budgets.get(call.name)
        count = state.tool_counts.get(call.name, 0)
        if limit is None or count < limit:
            return ""
        if state.search_mode and call.name == "web_search":
            return self._search_rule(
                "The search-attempt budget is exhausted.",
                "Answer only from the evidence already gathered.",
                "Do not issue more search calls.",
            )
        if state.search_mode and call.name == "fetch_url":
            return self._search_rule(
                "The page-fetch budget is exhausted.",
                "Answer only from the evidence already gathered.",
                "Do not fetch more pages.",
        )
        return f"Tool budget exceeded for {call.name} ({limit})"

    def _run_nested_skill_agent(self, agent_name: str, prompt: str, skill_id: str = "") -> AgentTurnResult:
        try:
            agent_contract = self.skill_runtime.load_agent_contract(agent_name, skill_id=skill_id)
        except Exception as exc:
            return AgentTurnResult(status="error", content="", reasoning="", skill_exchanges=[], error=str(exc))
        agent_record = self.skill_runtime.get_agent(agent_name)
        if agent_record is None:
            return AgentTurnResult(
                status="error",
                content="",
                reasoning="",
                skill_exchanges=[],
                error=f"Unknown companion agent: {agent_name}",
            )
        nested_config = copy.deepcopy(self.config) if isinstance(self.config, dict) else {}
        nested_agents_cfg = nested_config.get("agents", {}) if isinstance(nested_config.get("agents"), dict) else {}
        nested_agents_cfg["enable_skill_agents"] = False
        nested_config["agents"] = nested_agents_cfg
        nested = Agent(nested_config, self.skill_runtime, debug=self.debug)
        nested.system_prompt = (
            f"{self.system_prompt}\n\n"
            f"Companion agent profile: {agent_record.name}\n"
            f"{agent_record.description}\n\n"
            f"{agent_contract.prompt}"
        ).strip()
        return nested.run_turn(
            history_messages=[],
            user_input=prompt,
            thinking=False,
            branch_labels=[],
            attachments=[],
            stop_event=threading.Event(),
            on_event=None,
            confirm_shell=None,
        )

    def _background_skill_agent_worker(self, task_id: str, agent_name: str, prompt: str, skill_id: str) -> None:
        result = self._run_nested_skill_agent(agent_name, prompt, skill_id=skill_id)
        with self._bg_skill_agent_lock:
            task = self._bg_skill_agent_tasks.get(task_id)
            if not task:
                return
            task.completed_at = time.time()
            if result.status == "done":
                task.status = "done"
                task.output = result.content
            elif result.status == "cancelled":
                task.status = "cancelled"
                task.error = "cancelled"
            else:
                task.status = "error"
                task.error = result.error or "background agent failed"

    def _handle_spawn_skill_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        agents_cfg = self.config.get("agents", {}) if isinstance(self.config.get("agents"), dict) else {}
        if not bool(agents_cfg.get("enable_skill_agents", True)):
            raise PermissionError("Skill agents are disabled in configuration")
        action = str(args.get("action", "start")).strip().lower() or "start"
        if action in {"status", "wait"}:
            task_id = str(args.get("task_id", "")).strip()
            if not task_id:
                raise ValueError("Missing required argument: task_id")
            deadline = time.time() + max(1, int(args.get("timeout_s", 30))) if action == "wait" else time.time()
            while True:
                with self._bg_skill_agent_lock:
                    task = self._bg_skill_agent_tasks.get(task_id)
                    snapshot = None if task is None else {
                        "task_id": task.task_id,
                        "agent_name": task.agent_name,
                        "skill_id": task.skill_id,
                        "status": task.status,
                        "output": task.output,
                        "error": task.error,
                    }
                if snapshot is None:
                    raise FileNotFoundError(f"Unknown skill agent task: {task_id}")
                if action == "status" or snapshot["status"] != "running" or time.time() >= deadline:
                    return snapshot
                time.sleep(0.05)
        agent_name = str(args.get("agent_name", "")).strip()
        skill_id = str(args.get("skill_id", "")).strip()
        prompt = str(args.get("prompt", "")).strip()
        if not agent_name:
            raise ValueError("Missing required argument: agent_name")
        if not prompt:
            raise ValueError("Missing required argument: prompt")
        background = bool(args.get("background", True))
        if not background:
            result = self._run_nested_skill_agent(agent_name, prompt, skill_id=skill_id)
            return {
                "task_id": "",
                "agent_name": agent_name,
                "skill_id": skill_id,
                "status": result.status,
                "output": result.content,
                "error": result.error or "",
            }
        task_id = f"skill-agent-{uuid.uuid4().hex[:10]}"
        with self._bg_skill_agent_lock:
            self._bg_skill_agent_tasks[task_id] = BackgroundSkillAgentTask(
                task_id=task_id,
                agent_name=agent_name,
                skill_id=skill_id,
                prompt=prompt,
            )
        thread = threading.Thread(
            target=self._background_skill_agent_worker,
            args=(task_id, agent_name, prompt, skill_id),
            daemon=True,
        )
        thread.start()
        return {
            "task_id": task_id,
            "agent_name": agent_name,
            "skill_id": skill_id,
            "status": "running",
            "output": "",
            "error": "",
        }

    @staticmethod
    def _handle_request_user_input(args: Dict[str, Any]) -> Dict[str, Any]:
        question = str(args.get("question", "")).strip()
        if not question:
            raise ValueError("Missing required argument: question")
        options = args.get("options")
        normalized_options = [str(item).strip() for item in options if str(item).strip()] if isinstance(options, list) else []
        return {
            "question": question,
            "options": normalized_options,
            "header": str(args.get("header", "")).strip(),
            "awaiting_user_input": True,
        }

    @staticmethod
    def _direct_tool_answer(name: str, payload: Dict[str, Any]) -> str:
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            return ""
        if name == "request_user_input":
            question = str(data.get("question", "")).strip()
            options = data.get("options")
            lines = [question] if question else []
            if isinstance(options, list) and options:
                lines.append("Options: " + " | ".join(str(item) for item in options[:6]))
            lines.append("Reply with your choice or answer so I can continue.")
            return "\n".join(lines)
        if name == "create_directory":
            base = str(data.get("basename", "")).strip()
            path = str(data.get("filepath", "")).strip()
            if base and path:
                return f"I created the directory `{base}` in your workspace at `{path}`."
        if name == "create_file":
            base = str(data.get("basename", "")).strip()
            path = str(data.get("filepath", "")).strip()
            if base and path:
                return f"I created `{base}` in your workspace at `{path}`."
        if name == "create_files":
            created = data.get("created")
            if isinstance(created, list) and created:
                names = [str(item.get("basename", "")).strip() for item in created if isinstance(item, dict)]
                names = [name for name in names if name]
                if names:
                    if len(names) == 1:
                        return f"I created `{names[0]}` in your workspace."
                    return "I created these files in your workspace: " + ", ".join(f"`{name}`" for name in names) + "."
        if name == "edit_file":
            base = str(data.get("basename", "")).strip()
            path = str(data.get("filepath", "")).strip()
            if base and path:
                return f"I updated `{base}` in your workspace at `{path}`."
        if name == "delete_path":
            path = str(data.get("filepath", "")).strip()
            kind = str(data.get("kind", "path")).strip()
            if path:
                return f"I deleted the {kind} `{path}` from your workspace."
        if name == "list_memories":
            memories = data.get("memories")
            if isinstance(memories, list):
                if not memories:
                    return "You don't have any stored memories yet."
                lines = ["Recent memories:"]
                for item in memories[:5]:
                    if not isinstance(item, dict):
                        continue
                    memory_id = str(item.get("id", "")).strip()
                    memory_type = str(item.get("type", "")).strip()
                    text = str(item.get("text", "")).strip()
                    prefix = f"- #{memory_id}" if memory_id else "- "
                    if memory_type:
                        prefix += f" [{memory_type}]"
                    if text:
                        lines.append(f"{prefix} {text}".rstrip())
                if len(lines) > 1:
                    return "\n".join(lines)
        if name == "get_weather":
            city = str(data.get("city", "")).strip()
            temp = str(data.get("temp_c", "") or data.get("temperature_c", "")).strip()
            desc = str(data.get("desc", "") or data.get("condition", "")).strip()
            if city and temp:
                return f"The current weather in {city} is {temp} C" + (f" with {desc}." if desc else ".")
        return ""

    @staticmethod
    def _build_turn_journal(state: TurnState, result: AgentTurnResult) -> Dict[str, Any]:
        return {
            "status": result.status,
            "error": result.error or "",
            "selected_skills": [getattr(skill, "id", "") for skill in state.selected],
            "loaded_skills": list(state.loaded_skill_ids),
            "tool_counts": dict(state.tool_counts),
            "tool_evidence": [
                {
                    "name": item.name,
                    "args": item.args,
                    "result": item.result,
                }
                for item in state.evidence
            ],
            "search_mode": state.search_mode,
            "time_sensitive_query": state.time_sensitive_query,
            "search_failures": state.search_failure_count,
            "has_fetch_evidence": state.search_has_fetch_content,
            "model_usage": dict(state.model_usage),
        }

    def _record_tool_effects(self, state: TurnState, call: ToolCall, result: Dict[str, Any]) -> None:
        state.tool_counts[call.name] = state.tool_counts.get(call.name, 0) + 1
        state.evidence.append(ToolEvidence(name=call.name, args=dict(call.arguments), result=result))
        if call.name == "load_skill" and result.get("ok"):
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            skill_id = str(data.get("skill_id", "")).strip() if isinstance(data, dict) else ""
            if skill_id and skill_id not in state.loaded_skill_ids:
                state.loaded_skill_ids.append(skill_id)
                state.loaded_skill_ids.sort()
        if not state.search_mode:
            return
        if call.name not in {"web_search", "fetch_url"}:
            return
        if result.get("ok"):
            state.search_has_success = True
            if call.name == "fetch_url":
                state.search_has_fetch_content = True
                fetched_payload = result.get("data") if isinstance(result.get("data"), dict) else {}
                for key in ("url", "final_url"):
                    seen_url = str(fetched_payload.get(key, "")).strip()
                    if seen_url:
                        state.fetched_urls.add(seen_url)
            return
        if call.name == "web_search":
            state.search_failure_count += 1
        if call.name == "fetch_url":
            error_obj = result.get("error") or {}
            message = str(error_obj.get("message", "")).lower()
            raw_url = str(call.arguments.get("url", "")).strip()
            host = urllib.parse.urlparse(raw_url).netloc.lower()
            if host and any(code in message for code in ("http 401", "http 403", "http 429")):
                state.blocked_fetch_domains.add(host)

    def _log_turn_summary(self, state: TurnState, result: AgentTurnResult) -> None:
        self._debug_log(
            "turn_summary",
            status=result.status,
            error=result.error or "",
            selected_skills=[getattr(skill, "id", "") for skill in state.selected],
            tool_counts=state.tool_counts,
            evidence_count=len(state.evidence),
            search_mode=state.search_mode,
            search_failures=state.search_failure_count,
            fetched_urls=len(state.fetched_urls),
            blocked_domains=sorted(state.blocked_fetch_domains),
            content_chars=len(result.content),
            reasoning_chars=len(result.reasoning),
        )

    def run_turn(
        self,
        history_messages: List[Dict[str, Any]],
        user_input: str,
        thinking: bool,
        branch_labels: Optional[List[str]] = None,
        attachments: Optional[List[str]] = None,
        stop_event=None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        confirm_shell: Optional[Callable[[str], bool]] = None,
    ) -> AgentTurnResult:
        endpoint_err = self._validate_endpoints()
        if endpoint_err:
            return AgentTurnResult(status="error", content="", reasoning="", skill_exchanges=[], error=endpoint_err)

        if self._stop_requested(stop_event):
            return AgentTurnResult(status="cancelled", content="", reasoning="", skill_exchanges=[])

        if not self._ready_checked:
            ready = self.ensure_ready(stop_event=stop_event, on_event=on_event)
            if ready is None:
                return AgentTurnResult(status="cancelled", content="", reasoning="", skill_exchanges=[])
            if not ready:
                return AgentTurnResult(
                    status="error",
                    content="",
                    reasoning="",
                    skill_exchanges=[],
                    error=f"Model endpoint not ready: {self.models_endpoint}",
                )

        branch_labels = branch_labels or []
        attachments = attachments or []
        ctx = self._build_skill_context(user_input, branch_labels, attachments, history_messages)
        selected = self._select_skills(ctx, stop_event)
        state = self._build_turn_state(ctx, selected, history_messages, user_input)

        def finish(result: AgentTurnResult) -> AgentTurnResult:
            result.journal = self._build_turn_journal(state, result)
            self._log_turn_summary(state, result)
            return result

        while True:
            state.pass_index += 1
            pass_id = f"pass_{state.pass_index}"
            if stop_event is not None and stop_event.is_set():
                return finish(
                    AgentTurnResult(
                        status="cancelled",
                        content="",
                        reasoning=state.full_reasoning,
                        skill_exchanges=state.skill_exchanges,
                    )
                )

            system_content = self._compose_system_content(state.selected, state.ctx, state.loaded_skill_ids)
            if (
                state.search_mode
                and state.time_sensitive_query
                and state.forced_search_retry
                and state.tool_counts.get("web_search", 0) == 0
            ):
                system_content += (
                    "\n\nMandatory retrieval rule:\n"
                    "- This user request is time-sensitive.\n"
                    "- You must call web_search before answering.\n"
                    "- Do not answer from memory cutoff or prior knowledge alone.\n"
                    "- If web_search fails, say you could not verify the answer."
                )
            if (
                state.requires_workspace_action
                and state.forced_action_retry
                and not state.tool_counts
            ):
                system_content += (
                    "\n\nMandatory action rule:\n"
                    "- This is a confirmation of an immediate prior workspace action request.\n"
                    "- Use the available workspace tools to perform the requested action if policy allows.\n"
                    "- Do not replace an available workspace tool action with manual terminal instructions.\n"
                    "- Only decline if the required workspace tool is unavailable or policy blocks the action."
                )
            if state.workspace_scaffold_action:
                system_content += (
                    "\n\nWorkspace scaffold rule:\n"
                    "- This request requires creating a local scaffold, not just a directory.\n"
                    "- Do not stop after create_directory alone.\n"
                    "- Materialize the requested files with workspace file tools.\n"
                    "- For multi-file scaffolds or programs, prefer separate create_file calls, one file at a time.\n"
                    "- Avoid batching the whole scaffold into one create_files call unless the user explicitly asks for batching."
                )
            elif state.workspace_materialization_target > 0:
                system_content += (
                    "\n\nWorkspace materialization rule:\n"
                    "- This request requires creating local file content, not just a directory.\n"
                    "- Do not stop after create_directory alone.\n"
                    "- Materialize the requested file content with workspace file-creation tools before finishing."
                )
            if state.workspace_scaffold_action and state.forced_workspace_retry:
                system_content += (
                    "\n\nMandatory scaffold completion rule:\n"
                    "- The previous pass stopped before creating the requested files.\n"
                    "- Continue the scaffold now with workspace file-creation tools.\n"
                    "- Do not claim completion until the requested files exist."
                )
            elif state.workspace_materialization_target > 0 and state.forced_workspace_retry:
                system_content += (
                    "\n\nMandatory workspace completion rule:\n"
                    "- The previous pass stopped before creating the requested file content.\n"
                    "- Continue with workspace file-creation tools now.\n"
                    "- Do not claim completion until the requested file content exists."
                )
            if state.explicit_external_path:
                system_content += (
                    "\n\nExplicit path rule:\n"
                    f"- The user explicitly named a filesystem path outside the current workspace: {state.explicit_external_path}\n"
                    "- Do not silently substitute the current workspace root for that path.\n"
                    "- Acknowledge the mismatch if you need to reference the current workspace.\n"
                    "- If a tool can safely operate on the explicit path, pass that path directly.\n"
                    "- If the task requires running a command in that other directory, use a single shell command that targets the explicit path instead of assuming the current workspace."
                )
            shell_workflow_skills = self.skill_runtime.selected_shell_workflow_skills(state.selected, state.ctx)
            allow_shell_workflow = bool(shell_workflow_skills)
            if state.prefer_local_workspace_tools:
                system_content += (
                    "\n\nLocal workspace tool rule:\n"
                    "- This request is about local workspace files or folders.\n"
                    "- Use native workspace tools for local file creation, reading, editing, and folder creation.\n"
                    "- Prefer create_directory for folder creation.\n"
                    "- For multi-file scaffolds or programs, prefer create_file one file at a time so progress stays visible in the UI.\n"
                    "- Use create_files only when batching is acceptable and per-file progress is not important.\n"
                    "- Do not use web_search, fetch_url, open_url, or play_youtube for local workspace file tasks."
                )
                if allow_shell_workflow:
                    system_content += (
                        "\n- shell_command is allowed only for the documented selected-skill workflow required by this artifact task.\n"
                        f"- Skills requiring shell workflow here: {', '.join(shell_workflow_skills[:4])}\n"
                        "- Use documented shell/python commands from the selected skill to install missing dependencies and create the requested artifact.\n"
                        "- Each shell_command must be exactly one plain command.\n"
                        "- Do not use shell control operators, chaining, or redirection fallbacks such as `||`, `&&`, `;`, `|`, or `2>&1`.\n"
                        "- Do not use run_checks for dependency probing or installation on this task.\n"
                        "- Do not use python -c import probes before trying the documented install workflow.\n"
                        "- If a dependency is required or uncertain, run the skill's documented install command first after approval.\n"
                        "- Do not create helper .py files until required dependencies are installed.\n"
                        "- Do not use shell_command for generic local file inspection or folder creation."
                    )
                else:
                    system_content += "\n- Do not use shell_command to create folders or inspect local files."
            requested_exts = self.skill_runtime.requested_opaque_artifact_extensions(state.ctx)
            selected_materializers = self.skill_runtime.selected_artifact_materializers(state.selected, state.ctx)
            if requested_exts and "create" in self.skill_runtime.task_intents(state.ctx) and not selected_materializers:
                system_content += (
                    "\n\nOpaque artifact capability rule:\n"
                    f"- The request asks for a real opaque artifact: {', '.join(requested_exts[:3])}\n"
                    "- None of the selected skills expose an executable creation path for that artifact in this runtime.\n"
                    "- Do not invent script names.\n"
                    "- Do not use shell_command or run_checks to probe or install dependencies for this local workspace file task.\n"
                    "- Do not attempt create_file/create_files as a surrogate for the opaque artifact.\n"
                    "- Say directly that no executable creation path is available."
                )
            system_messages = [{"role": "system", "content": system_content}]

            model_messages = self.context_mgr.prune(system_messages + state.dynamic_history, self.context_budget_max_tokens)
            tools = self.skill_runtime.tools_for_turn(state.selected, ctx=state.ctx, loaded_skill_ids=state.loaded_skill_ids)
            payload = self._build_payload(model_messages, thinking=thinking, tools=tools or None)

            try:
                stream_result = self._call_with_retry(payload, stop_event, on_event, pass_id=pass_id)
            except Exception as exc:
                message = str(exc)
                self._emit(on_event, {"type": "error", "text": message})
                return finish(
                    AgentTurnResult(
                        status="error",
                        content="",
                        reasoning=state.full_reasoning,
                        skill_exchanges=state.skill_exchanges,
                        error=message,
                    )
                )

            if stream_result.finish_reason == "cancelled":
                return finish(
                    AgentTurnResult(
                        status="cancelled",
                        content=stream_result.content,
                        reasoning=self._append_reasoning(state.full_reasoning, stream_result.reasoning),
                        skill_exchanges=state.skill_exchanges,
                    )
                )

            state.full_reasoning = self._append_reasoning(state.full_reasoning, stream_result.reasoning)
            stream_usage = getattr(stream_result, "usage", {}) or {}
            if isinstance(stream_usage, dict) and stream_usage:
                state.model_usage = dict(stream_usage)
            self._emit(
                on_event,
                {
                    "type": "pass_end",
                    "pass_id": pass_id,
                    "finish_reason": stream_result.finish_reason,
                    "has_content": bool(stream_result.content.strip()),
                    "has_tool_calls": bool(stream_result.tool_calls),
                },
            )

            if stream_result.finish_reason == "tool_calls":
                if not stream_result.tool_calls:
                    return finish(
                        AgentTurnResult(
                            status="error",
                            content="",
                            reasoning=state.full_reasoning,
                            skill_exchanges=state.skill_exchanges,
                            error="finish_reason tool_calls without tool calls",
                        )
                    )

                assistant_msg = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": self._safe_json_dumps(self._tool_call_args_for_history(call.arguments)),
                            },
                        }
                        for call in stream_result.tool_calls
                    ],
                }
                state.dynamic_history.append(assistant_msg)
                state.skill_exchanges.append(assistant_msg)

                force_finalize_reason = ""
                direct_answers: List[str] = []
                single_tool_result: Optional[Dict[str, Any]] = None
                all_tool_success = True
                for call in stream_result.tool_calls:
                    state.action_depth += 1
                    if state.action_depth > self.max_action_depth:
                        if state.search_mode and state.search_has_success:
                            return finish(
                                self._finalize_turn(
                                    system_content,
                                    state,
                                    stop_event,
                                    on_event,
                                    pass_id,
                                    self._search_rule(
                                        "The search loop budget is exhausted.",
                                        "Answer using the successful search or fetch results already in the conversation.",
                                        "Do not search again.",
                                    ),
                                )
                            )
                        return finish(
                            AgentTurnResult(
                                status="error",
                                content="",
                                reasoning=state.full_reasoning,
                                skill_exchanges=state.skill_exchanges,
                                error=f"Max skill action depth ({self.max_action_depth}) exceeded",
                            )
                        )

                    self._emit(
                        on_event,
                        {
                            "type": "tool_call",
                            "stream_id": call.stream_id,
                            "name": call.name,
                            "arguments": call.arguments,
                            "id": call.id,
                        },
                    )

                    force_finalize_reason = self._tool_budget_reason(state, call)
                    if force_finalize_reason:
                        if not state.search_mode:
                            return finish(
                                AgentTurnResult(
                                    status="error",
                                    content="",
                                    reasoning=state.full_reasoning,
                                    skill_exchanges=state.skill_exchanges,
                                    error=force_finalize_reason,
                                )
                            )
                        break

                    if state.prefer_local_workspace_tools and call.name in {
                        "shell_command",
                        "web_search",
                        "fetch_url",
                        "open_url",
                        "play_youtube",
                    }:
                        if call.name == "shell_command" and allow_shell_workflow:
                            pass
                        else:
                            result = {
                                "ok": False,
                                "data": None,
                                "error": {
                                    "code": "E_POLICY",
                                    "message": f"{call.name} is not allowed for local workspace file tasks; use workspace tools instead.",
                                },
                                "meta": {},
                            }
                            self._emit(
                                on_event,
                                {
                                    "type": "tool_result",
                                    "name": call.name,
                                    "id": call.id,
                                    "result": result,
                                },
                            )
                            tool_message = {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "name": call.name,
                                "content": self._safe_json_dumps(self._tool_result_for_history(call.name, result)),
                            }
                            state.dynamic_history.append(tool_message)
                            state.skill_exchanges.append(tool_message)
                            self._record_tool_effects(state, call, result)
                            all_tool_success = False
                            continue

                    if state.search_mode and call.name == "fetch_url":
                        raw_url = str(call.arguments.get("url", "")).strip()
                        if raw_url:
                            host = urllib.parse.urlparse(raw_url).netloc.lower()
                            if raw_url in state.fetched_urls:
                                force_finalize_reason = self._search_rule(
                                    "This URL was already fetched in this turn.",
                                    "Do not retry the same page.",
                                    "Answer from the evidence already gathered.",
                                )
                                break
                            if host and host in state.blocked_fetch_domains:
                                force_finalize_reason = self._search_rule(
                                    "This source domain already blocked a fetch attempt in this turn.",
                                    "Do not retry the same blocked domain.",
                                    "Answer from the remaining evidence.",
                                )
                                break

                    result = self.skill_runtime.execute_tool_call(
                        call.name,
                        call.arguments,
                        selected=state.selected,
                        ctx=state.ctx,
                        confirm_shell=confirm_shell,
                        spawn_skill_agent=self._handle_spawn_skill_agent,
                        request_user_input=self._handle_request_user_input,
                    )

                    self._emit(
                        on_event,
                        {
                            "type": "tool_result",
                            "name": call.name,
                            "id": call.id,
                            "result": result,
                        },
                    )

                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": self._safe_json_dumps(self._tool_result_for_history(call.name, result)),
                    }
                    state.dynamic_history.append(tool_message)
                    state.skill_exchanges.append(tool_message)
                    self._record_tool_effects(state, call, result)
                    if not result.get("ok"):
                        all_tool_success = False
                    direct = self._direct_tool_answer(call.name, result)
                    if direct:
                        direct_answers.append(direct)
                    if len(stream_result.tool_calls) == 1:
                        single_tool_result = result
                    if (
                        call.name == "request_user_input"
                        and result.get("ok")
                        and isinstance(result.get("data"), dict)
                        and bool(result["data"].get("awaiting_user_input"))
                    ):
                        self.skill_runtime.post_response(state.selected, state.ctx, direct or "")
                        return finish(
                            AgentTurnResult(
                                status="done",
                                content=direct or "",
                                reasoning=state.full_reasoning,
                                skill_exchanges=state.skill_exchanges,
                            )
                        )

                    if not state.search_mode:
                        continue
                    if call.name == "fetch_url" and state.search_failure_count >= 2:
                        force_finalize_reason = self._search_rule(
                            "The search provider has already failed repeatedly.",
                            "Do not use memory or prior knowledge to fill gaps.",
                            "If the fetched page does not explicitly answer the question, say you could not verify it.",
                        )
                        break
                    if call.name not in {"web_search", "fetch_url"} and state.search_failure_count >= 2:
                        force_finalize_reason = self._search_rule(
                            "Search has failed repeatedly.",
                            "Do not switch to memory recall or unrelated tools.",
                            "Answer only with verified evidence, or say verification failed.",
                        )
                        break
                    if call.name == "web_search" and not result.get("ok") and state.search_failure_count >= 2 and not state.search_has_success:
                        return finish(
                            AgentTurnResult(
                                status="done",
                                content=self._unverified_search_answer(),
                                reasoning=state.full_reasoning,
                                skill_exchanges=state.skill_exchanges,
                            )
                        )
                    if call.name == "fetch_url" and not result.get("ok") and state.search_has_success:
                        force_finalize_reason = self._search_rule(
                            "A page fetch failed.",
                            "Continue with the successful search results and any successful fetches already gathered.",
                            "Do not keep retrying searches indefinitely.",
                        )
                        break
                    if (
                        call.name == "web_search"
                        and state.tool_counts.get("web_search", 0) >= state.tool_budgets.get("web_search", 0)
                        and state.search_has_success
                    ):
                        force_finalize_reason = self._search_rule(
                            "Enough search attempts have already been made.",
                            "Summarize from the best available results now.",
                            "Do not issue more search calls.",
                        )
                        break
                    if (
                        call.name == "fetch_url"
                        and state.tool_counts.get("fetch_url", 0) >= state.tool_budgets.get("fetch_url", 0)
                        and state.search_has_fetch_content
                    ):
                        force_finalize_reason = self._search_rule(
                            "Enough pages have been fetched.",
                            "Answer from the gathered evidence now.",
                            "Do not fetch additional pages.",
                        )
                        break

                if force_finalize_reason:
                    return finish(self._finalize_turn(system_content, state, stop_event, on_event, pass_id, force_finalize_reason))
                search_tool_called = any(call.name in {"web_search", "fetch_url"} for call in stream_result.tool_calls)
                if (
                    len(stream_result.tool_calls) == 1
                    and all_tool_success
                    and len(direct_answers) == 1
                    and single_tool_result is not None
                    and self._direct_tool_completion_allowed(state, stream_result.tool_calls[0], single_tool_result)
                    and (not state.search_mode or not search_tool_called)
                    and not state.batch_workspace_action
                    and not state.workspace_scaffold_action
                    and (
                        state.workspace_materialization_target <= 0
                        or self._workspace_materialization_count(state) >= state.workspace_materialization_target
                    )
                ):
                    self.skill_runtime.post_response(state.selected, state.ctx, direct_answers[0])
                    return finish(
                        AgentTurnResult(
                            status="done",
                            content=direct_answers[0],
                            reasoning=state.full_reasoning,
                            skill_exchanges=state.skill_exchanges,
                        )
                    )
                continue

            raw_final = stream_result.content
            final = self._sanitize_final_content(raw_final)
            if self._contains_tool_markup(raw_final):
                finalized = self._finalize_turn(
                    system_content,
                    state,
                    stop_event,
                    on_event,
                    pass_id,
                    (
                        "Output correction rule:\n"
                        "- The previous reply emitted raw tool markup instead of a user-facing answer.\n"
                        "- Rewrite it as a normal assistant response.\n"
                        "- Do not emit tool markup, XML-like tags, or pseudo-function calls."
                    ),
                )
                if finalized.status != "done":
                    return finish(finalized)
                state.full_reasoning = finalized.reasoning
                final = finalized.content
            if (
                state.workspace_materialization_target > 0
                and self._workspace_materialization_count(state) < state.workspace_materialization_target
            ):
                if not state.forced_workspace_retry:
                    state.forced_workspace_retry = True
                    continue
                return finish(
                    AgentTurnResult(
                        status="done",
                        content="I completed part of the workspace setup, but I failed to finish creating all requested file content with the available workspace tools.",
                        reasoning=state.full_reasoning,
                        skill_exchanges=state.skill_exchanges,
                    )
                )
            if state.search_mode and state.time_sensitive_query and state.tool_counts.get("web_search", 0) == 0:
                if not state.forced_search_retry:
                    state.forced_search_retry = True
                    continue
                return finish(
                    AgentTurnResult(
                        status="done",
                        content=self._unverified_search_answer(),
                        reasoning=state.full_reasoning,
                        skill_exchanges=state.skill_exchanges,
                    )
                )
            if state.requires_workspace_action and not state.tool_counts:
                if self._looks_like_manual_workspace_advice(final) or self._looks_like_workspace_action_claim(final):
                    if not state.forced_action_retry:
                        state.forced_action_retry = True
                        continue
                    return finish(
                        AgentTurnResult(
                            status="done",
                            content=self._workspace_action_failure_answer(),
                            reasoning=state.full_reasoning,
                            skill_exchanges=state.skill_exchanges,
                        )
                    )
                if not state.forced_action_retry and not final.strip():
                    state.forced_action_retry = True
                    continue
            if self._needs_fetch_evidence(state):
                return finish(
                    AgentTurnResult(
                        status="done",
                        content=self._unverified_search_answer(),
                        reasoning=state.full_reasoning,
                        skill_exchanges=state.skill_exchanges,
                    )
                )
            if not final.strip():
                finalized = self._finalize_turn(system_content, state, stop_event, on_event, pass_id)
                if finalized.status != "done":
                    return finish(finalized)
                state.full_reasoning = finalized.reasoning
                final = finalized.content
                if self._needs_fetch_evidence(state):
                    return finish(
                        AgentTurnResult(
                            status="done",
                            content=self._unverified_search_answer(),
                            reasoning=state.full_reasoning,
                            skill_exchanges=state.skill_exchanges,
                        )
                    )

            self.skill_runtime.post_response(state.selected, state.ctx, final)
            return finish(
                AgentTurnResult(
                    status="done",
                    content=final,
                    reasoning=state.full_reasoning,
                    skill_exchanges=state.skill_exchanges,
                )
            )
