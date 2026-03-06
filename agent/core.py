from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agent.context import ContextWindowManager
from agent.prompts import build_system_prompt
from core.skills import SkillContext, SkillRuntime
from core.streaming import build_ssl_context, should_retry, stream_chat_completions


@dataclass
class ToolCall:
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
    def __init__(self) -> None:
        self._items: Dict[int, Dict[str, Any]] = {}

    def ingest(self, deltas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        updates: List[Dict[str, Any]] = []
        for delta in deltas:
            index = int(delta.get("index", 0))
            item = self._items.setdefault(
                index,
                {
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
        self.fast_tool_finalize = bool(agent_cfg.get("fast_tool_finalize", True))
        raw_fast_tools = agent_cfg.get(
            "fast_finalize_tools",
            ["create_file", "edit_file", "delete_file"],
        )
        if isinstance(raw_fast_tools, list):
            self.fast_finalize_tools = {str(name).strip() for name in raw_fast_tools if str(name).strip()}
        else:
            self.fast_finalize_tools = {"create_file", "edit_file", "delete_file"}
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

    def ensure_ready(self) -> bool:
        deadline = time.monotonic() + self.readiness_timeout_s
        headers = self._headers()
        req = urllib.request.Request(self.models_endpoint, headers=headers, method="GET")

        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(req, timeout=self.connect_timeout_s, context=self.ssl_context) as resp:
                    if 200 <= resp.status < 300:
                        self._ready_checked = True
                        return True
            except Exception:
                time.sleep(self.readiness_poll_s)
                continue
        self._ready_checked = False
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

    def _build_fast_finalize_response(self, calls: List[ToolCall], results: List[Dict[str, Any]]) -> Optional[str]:
        if not self.fast_tool_finalize:
            return None
        if not calls or len(calls) != len(results):
            return None

        lines: List[str] = []
        for call, result in zip(calls, results):
            if call.name not in self.fast_finalize_tools:
                return None
            if not result.get("ok"):
                return None

            filepath = ""
            if isinstance(result.get("data"), dict):
                filepath = str(result["data"].get("filepath", "") or "")

            if call.name == "create_file":
                verb = "Created"
            elif call.name == "edit_file":
                verb = "Updated"
            elif call.name == "delete_file":
                verb = "Deleted"
            else:
                verb = "Completed"

            if filepath:
                lines.append(f"{verb} `{filepath}`.")
            else:
                lines.append(f"{verb} via `{call.name}`.")

        if len(lines) == 1:
            return lines[0]
        return "Applied requested file changes:\n" + "\n".join(f"- {line}" for line in lines)

    def _stream_one_pass(
        self,
        payload: Dict[str, Any],
        stop_event,
        on_event: Optional[Callable[[Dict[str, Any]], None]],
    ) -> StreamPassResult:
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        finish_reason = "stop"
        tool_acc = ToolCallAccumulator()
        tool_phase_started = False

        for chunk in stream_chat_completions(
            endpoint=self.model_endpoint,
            payload=payload,
            timeout_s=self.request_timeout_s,
            headers=self._headers(),
            ssl_context=self.ssl_context,
            stop_event=stop_event,
        ):
            if stop_event is not None and stop_event.is_set():
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
                            "stream_id": f"call_{update['index']}",
                            "id": update.get("id") or "",
                            "name": update.get("name") or "",
                            "raw_arguments": update.get("raw_arguments") or "",
                        },
                    )

            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])

        return StreamPassResult(
            finish_reason=finish_reason,
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            tool_calls=tool_acc.finalize(),
        )

    def _call_with_retry(self, payload: Dict[str, Any], stop_event, on_event):
        attempt = 0
        while True:
            try:
                return self._stream_one_pass(payload, stop_event, on_event)
            except Exception as exc:
                retryable = should_retry(exc)
                if retryable and attempt < self.per_turn_retries:
                    attempt += 1
                    self._emit(on_event, {"type": "info", "text": f"Retrying request ({attempt}/{self.per_turn_retries})..."})
                    time.sleep(self.retry_backoff_s)
                    self.ensure_ready()
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

        if not self._ready_checked and not self.ensure_ready():
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

        while True:
            if stop_event is not None and stop_event.is_set():
                return AgentTurnResult(
                    status="cancelled",
                    content="",
                    reasoning=full_reasoning,
                    skill_exchanges=skill_exchanges,
                )

            skill_block = self.skill_runtime.compose_skill_block(
                selected,
                ctx,
                context_limit=self.context_mgr.context_limit,
            )

            system_content = self.system_prompt
            if skill_block:
                system_content = (
                    f"{self.system_prompt}\n\n"
                    f"Active skill guidance:\n\n{skill_block}"
                )
            system_messages = [{"role": "system", "content": system_content}]

            model_messages = system_messages + dynamic_history
            model_messages = self.context_mgr.prune(model_messages, self.context_budget_max_tokens)

            payload = {
                "messages": model_messages,
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": bool(thinking)},
            }
            if self.default_max_tokens is not None:
                payload["max_tokens"] = self.default_max_tokens

            tools = self.skill_runtime.tools_for_skills(selected)
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            try:
                stream_result = self._call_with_retry(payload, stop_event, on_event)
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
                    reasoning=full_reasoning + stream_result.reasoning,
                    skill_exchanges=skill_exchanges,
                )

            full_reasoning += stream_result.reasoning

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
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
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

                batch_results: List[Dict[str, Any]] = []
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
                            "stream_id": f"call_{call.index}",
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
                    batch_results.append(result)

                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                    dynamic_history.append(tool_message)
                    skill_exchanges.append(tool_message)

                fast_reply = self._build_fast_finalize_response(stream_result.tool_calls, batch_results)
                if fast_reply is not None:
                    self.skill_runtime.post_response(selected, ctx, fast_reply)
                    return AgentTurnResult(
                        status="done",
                        content=fast_reply,
                        reasoning=full_reasoning,
                        skill_exchanges=skill_exchanges,
                    )

                continue

            # stop or length: final answer produced
            final = stream_result.content
            self.skill_runtime.post_response(selected, ctx, final)
            return AgentTurnResult(
                status="done",
                content=final,
                reasoning=full_reasoning,
                skill_exchanges=skill_exchanges,
            )
