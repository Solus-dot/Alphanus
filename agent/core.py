from __future__ import annotations

import json
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
        self.auth_header = agent_cfg.get("auth_header")
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

    def _compose_system_content(self, selected: List[Any], ctx: SkillContext) -> str:
        skill_block = self.skill_runtime.compose_skill_block(
            selected,
            ctx,
            context_limit=self.context_mgr.context_limit,
        )
        if not skill_block:
            return self.system_prompt
        return f"{self.system_prompt}\n\nActive skill guidance:\n\n{skill_block}"

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
        selected = self.skill_runtime.select_skills(ctx)

        dynamic_history = list(history_messages)
        skill_exchanges: List[Dict[str, Any]] = []
        action_depth = 0
        full_reasoning = ""
        pass_index = 0

        while True:
            pass_index += 1
            pass_id = f"pass_{pass_index}"
            if stop_event is not None and stop_event.is_set():
                return AgentTurnResult(
                    status="cancelled",
                    content="",
                    reasoning=full_reasoning,
                    skill_exchanges=skill_exchanges,
                )

            system_content = self._compose_system_content(selected, ctx)
            system_messages = [{"role": "system", "content": system_content}]

            model_messages = system_messages + dynamic_history
            model_messages = self.context_mgr.prune(model_messages, self.context_budget_max_tokens)
            tools = self.skill_runtime.tools_for_skills(selected)
            payload = self._build_payload(model_messages, thinking=thinking, tools=tools or None)

            try:
                stream_result = self._call_with_retry(payload, stop_event, on_event, pass_id=pass_id)
            except Exception as exc:
                message = str(exc)
                self._emit(on_event, {"type": "error", "text": message})
                return AgentTurnResult(
                    status="error",
                    content="",
                    reasoning=full_reasoning,
                    skill_exchanges=skill_exchanges,
                    error=message,
                )

            if stream_result.finish_reason == "cancelled":
                return AgentTurnResult(
                    status="cancelled",
                    content=stream_result.content,
                    reasoning=self._append_reasoning(full_reasoning, stream_result.reasoning),
                    skill_exchanges=skill_exchanges,
                )

            full_reasoning = self._append_reasoning(full_reasoning, stream_result.reasoning)
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
                    return AgentTurnResult(
                        status="error",
                        content="",
                        reasoning=full_reasoning,
                        skill_exchanges=skill_exchanges,
                        error="finish_reason tool_calls without tool calls",
                    )

                assistant_tool_calls = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": self._safe_json_dumps(call.arguments),
                        },
                    }
                    for call in stream_result.tool_calls
                ]

                assistant_msg = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": assistant_tool_calls,
                }
                dynamic_history.append(assistant_msg)
                skill_exchanges.append(assistant_msg)

                for call in stream_result.tool_calls:
                    action_depth += 1
                    if action_depth > self.max_action_depth:
                        return AgentTurnResult(
                            status="error",
                            content="",
                            reasoning=full_reasoning,
                            skill_exchanges=skill_exchanges,
                            error=f"Max skill action depth ({self.max_action_depth}) exceeded",
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

                    result = self.skill_runtime.execute_tool_call(
                        call.name,
                        call.arguments,
                        selected=selected,
                        ctx=ctx,
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

                    history_result = self._tool_result_for_history(call.name, result)
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": self._safe_json_dumps(history_result),
                    }
                    dynamic_history.append(tool_message)
                    skill_exchanges.append(tool_message)

                continue

            # stop or length: final answer produced
            final = stream_result.content
            if not final.strip():
                finalize_system = (
                    system_content
                    + "\n\nFinalization rule:\n"
                    + "- Provide only the final user-facing answer using prior tool results.\n"
                    + "- Do not call tools.\n"
                    + "- Do not include hidden reasoning."
                )
                finalize_messages = [{"role": "system", "content": finalize_system}] + dynamic_history
                finalize_messages = self.context_mgr.prune(finalize_messages, self.context_budget_max_tokens)
                finalize_payload = self._build_payload(finalize_messages, thinking=False, tools=None)
                try:
                    finalize_result = self._call_with_retry(
                        finalize_payload,
                        stop_event,
                        on_event,
                        pass_id=f"{pass_id}_final",
                    )
                except Exception as exc:
                    message = str(exc)
                    self._emit(on_event, {"type": "error", "text": message})
                    return AgentTurnResult(
                        status="error",
                        content="",
                        reasoning=full_reasoning,
                        skill_exchanges=skill_exchanges,
                        error=message,
                    )

                if finalize_result.finish_reason == "cancelled":
                    return AgentTurnResult(
                        status="cancelled",
                        content="",
                        reasoning=self._append_reasoning(full_reasoning, finalize_result.reasoning),
                        skill_exchanges=skill_exchanges,
                    )
                if finalize_result.finish_reason == "tool_calls":
                    return AgentTurnResult(
                        status="error",
                        content="",
                        reasoning=self._append_reasoning(full_reasoning, finalize_result.reasoning),
                        skill_exchanges=skill_exchanges,
                        error="Finalization pass unexpectedly returned tool calls",
                    )
                full_reasoning = self._append_reasoning(full_reasoning, finalize_result.reasoning)
                final = finalize_result.content

            self.skill_runtime.post_response(selected, ctx, final)
            return AgentTurnResult(
                status="done",
                content=final,
                reasoning=full_reasoning,
                skill_exchanges=skill_exchanges,
            )
