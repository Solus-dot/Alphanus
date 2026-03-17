from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.context import ContextWindowManager
from agent.prompts import build_system_prompt
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
    dynamic_history: List[Dict[str, Any]]
    skill_exchanges: List[Dict[str, Any]]
    evidence: List[ToolEvidence]
    full_reasoning: str = ""
    pass_index: int = 0
    action_depth: int = 0
    forced_search_retry: bool = False
    search_mode: bool = False
    time_sensitive_query: bool = False
    search_failure_count: int = 0
    search_has_success: bool = False
    search_has_fetch_content: bool = False
    blocked_fetch_domains: set[str] = field(default_factory=set)
    fetched_urls: set[str] = field(default_factory=set)
    tool_counts: Dict[str, int] = field(default_factory=dict)
    tool_budgets: Dict[str, int] = field(default_factory=dict)


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
        self.config = config
        self.skill_runtime = skill_runtime
        self.debug = debug

        agent_cfg = config.get("agent", {})
        self.model_endpoint = agent_cfg.get("model_endpoint", "http://127.0.0.1:8080/v1/chat/completions")
        self.models_endpoint = agent_cfg.get("models_endpoint", "http://127.0.0.1:8080/v1/models")
        self.auth_header = (
            os.environ.get("ALPHANUS_AUTH_HEADER", "").strip()
            or os.environ.get("AUTH_HEADER", "").strip()
            or None
        )
        self.tls_verify = bool(agent_cfg.get("tls_verify", True))
        self.ca_bundle_path = agent_cfg.get("ca_bundle_path")
        self.allow_cross_host = bool(agent_cfg.get("allow_cross_host_endpoints", False))

        self.connect_timeout_s = 10
        self.request_timeout_s = float(agent_cfg.get("request_timeout_s", 180))
        self.readiness_timeout_s = float(agent_cfg.get("readiness_timeout_s", 30))
        self.readiness_poll_s = float(agent_cfg.get("readiness_poll_s", 0.5))
        self.per_turn_retries = 1
        self.retry_backoff_s = 0.5

        raw_max_tokens = agent_cfg.get("max_tokens")
        if raw_max_tokens in (None, "", 0):
            self.default_max_tokens: Optional[int] = None
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
        self.system_prompt = build_system_prompt(self.skill_runtime.workspace.workspace_root)

        context_cfg = config.get("context", {})
        self.context_mgr = ContextWindowManager(
            context_limit=int(context_cfg.get("context_limit", 8192)),
            keep_last_n=int(context_cfg.get("keep_last_n", 10)),
            safety_margin=int(context_cfg.get("safety_margin", 500)),
        )

        self.ssl_context = build_ssl_context(self.tls_verify, self.ca_bundle_path)
        self._ready_checked = False
        self._skill_snapshot: Optional[SkillRoutingSnapshot] = None
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

    def _validate_endpoints(self) -> Optional[str]:
        model_host = urllib.parse.urlparse(self.model_endpoint).netloc
        models_host = urllib.parse.urlparse(self.models_endpoint).netloc
        if model_host != models_host and not self.allow_cross_host:
            return "agent.model_endpoint and agent.models_endpoint must share host"
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
    ) -> SkillContext:
        hits = self.skill_runtime.memory.search(user_input, top_k=3, min_score=0.45)
        return SkillContext(
            user_input=user_input,
            branch_labels=branch_labels,
            attachments=attachments,
            workspace_root=str(self.skill_runtime.workspace.workspace_root),
            memory_hits=hits,
        )

    def _get_skill_snapshot(self, force_refresh: bool = False) -> SkillRoutingSnapshot:
        generation = int(getattr(self.skill_runtime, "generation", 0))
        if not force_refresh and self._skill_snapshot and self._skill_snapshot.generation == generation:
            return self._skill_snapshot
        skills = list(self.skill_runtime.enabled_skills())
        catalog = self.skill_runtime.skill_catalog_text()
        self._skill_snapshot = SkillRoutingSnapshot(generation=generation, skills=skills, catalog=catalog)
        return self._skill_snapshot

    def reload_skills(self) -> int:
        self.skill_runtime.load_skills()
        self._skill_snapshot = None
        return int(getattr(self.skill_runtime, "generation", 0))

    def _compose_system_content(self, selected: List[Any], ctx: SkillContext) -> str:
        skill_block = self.skill_runtime.compose_skill_block(
            selected,
            ctx,
            context_limit=self.context_mgr.context_limit,
        )
        if not skill_block:
            return self.system_prompt
        return f"{self.system_prompt}\n\nActive skill guidance:\n\n{skill_block}"

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
        enabled = snapshot.skills
        if not enabled:
            return []

        skills_cfg = self.config.get("skills", {}) if isinstance(self.config, dict) else {}
        configured_limit = int(skills_cfg.get("max_active_skills", getattr(self.skill_runtime, "max_active_skills", 2)))
        limit = configured_limit if configured_limit > 0 else 2
        catalog = snapshot.catalog
        routing_system = (
            "You are selecting local skills for the next assistant turn.\n"
            "Choose the smallest set of skills needed for the user's request.\n"
            "Use the skill descriptions, tags, and available tool names to infer intent.\n"
            f"Return strict JSON only in the form {{\"skills\": [\"skill-id\"]}} with at most {limit} ids.\n"
            "Return an empty list when no skill is needed.\n"
            "Do not explain your choice."
        )
        routing_messages = [
            {"role": "system", "content": routing_system},
            {
                "role": "user",
                "content": f"User request:\n{ctx.user_input}\n\nAvailable skills:\n{catalog}",
            },
        ]
        payload = self._build_payload(routing_messages, thinking=False, tools=None)
        try:
            result = self._call_with_retry(payload, stop_event, None, pass_id="skill_route")
        except Exception:
            return None
        if result.finish_reason == "cancelled":
            return None
        skill_ids = self._parse_skill_route(result.content, [skill.id for skill in enabled], limit)
        if skill_ids is None:
            return None
        return self.skill_runtime.skills_by_ids(skill_ids)

    def _select_skills(
        self,
        ctx: SkillContext,
        stop_event,
    ) -> List[Any]:
        skills_cfg = self.config.get("skills", {}) if isinstance(self.config, dict) else {}
        mode = str(skills_cfg.get("selection_mode", getattr(self.skill_runtime, "selection_mode", ""))).strip().lower()
        if mode == "model":
            snapshot = self._get_skill_snapshot()
            routed = self._route_skills_with_model(ctx, snapshot, stop_event)
            if routed is not None:
                return routed
        return self.skill_runtime.select_skills(ctx)

    def _build_payload(
        self,
        model_messages: List[Dict[str, Any]],
        thinking: bool,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "messages": model_messages,
            "stream": True,
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
        if self.max_reasoning_chars <= 0:
            return ""
        if len(full_reasoning) >= self.max_reasoning_chars:
            return full_reasoning
        remaining = self.max_reasoning_chars - len(full_reasoning)
        if len(delta_reasoning) <= remaining:
            return full_reasoning + delta_reasoning
        return full_reasoning + delta_reasoning[:remaining]

    @staticmethod
    def _sanitize_final_content(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.IGNORECASE | re.DOTALL)
        kept: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
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
        if first_result.content and not leaked_markup:
            return first_result
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
        return TurnState(
            ctx=ctx,
            selected=selected,
            dynamic_history=list(history_messages),
            skill_exchanges=[],
            evidence=[],
            search_mode=self._is_search_skill_selected(selected),
            time_sensitive_query=self._is_time_sensitive_query(user_input),
            tool_counts={},
            tool_budgets=dict(self.default_tool_budgets),
        )

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

    @staticmethod
    def _direct_tool_answer(name: str, payload: Dict[str, Any]) -> str:
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            return ""
        if name == "create_file":
            base = str(data.get("basename", "")).strip()
            path = str(data.get("filepath", "")).strip()
            if base and path:
                return f"I created `{base}` in your workspace at `{path}`."
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
        if name == "delete_file":
            path = str(data.get("filepath", "")).strip()
            if path:
                return f"I deleted `{path}` from your workspace."
        if name == "get_weather":
            city = str(data.get("city", "")).strip()
            temp = str(data.get("temp_c", "")).strip()
            desc = str(data.get("desc", "")).strip()
            if city and temp:
                return f"The current weather in {city} is {temp} C" + (f" with {desc}." if desc else ".")
        return ""

    @staticmethod
    def _build_turn_journal(state: TurnState, result: AgentTurnResult) -> Dict[str, Any]:
        return {
            "status": result.status,
            "error": result.error or "",
            "selected_skills": [getattr(skill, "id", "") for skill in state.selected],
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
        }

    def _record_tool_effects(self, state: TurnState, call: ToolCall, result: Dict[str, Any]) -> None:
        state.tool_counts[call.name] = state.tool_counts.get(call.name, 0) + 1
        state.evidence.append(ToolEvidence(name=call.name, args=dict(call.arguments), result=result))
        if not state.search_mode:
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
        ctx = self._build_skill_context(user_input, branch_labels, attachments)
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

            system_content = self._compose_system_content(state.selected, state.ctx)
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
            system_messages = [{"role": "system", "content": system_content}]

            model_messages = self.context_mgr.prune(system_messages + state.dynamic_history, self.context_budget_max_tokens)
            tools = self.skill_runtime.tools_for_skills(state.selected)
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
                if (
                    not state.search_mode
                    and len(stream_result.tool_calls) == 1
                    and all_tool_success
                    and len(direct_answers) == 1
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

            final = stream_result.content
            if self._contains_tool_markup(final):
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
