from __future__ import annotations

from typing import cast

from core.endpoint_modes import CONCRETE_ENDPOINT_MODES, ENDPOINT_MODE_CHAT, ENDPOINT_MODE_RESPONSES
from core.message_types import JSONValue
from core.types import JsonObject


class ProviderPayloadAdapter:
    @staticmethod
    def chat_tools_to_responses(tools: list[JsonObject]) -> list[JsonObject]:
        converted: list[JsonObject] = []
        for item in tools:
            if not isinstance(item, dict):
                continue
            function = item.get("function")
            if isinstance(function, dict):
                converted.append(
                    {
                        "type": "function",
                        "name": str(function.get("name", "")).strip(),
                        "description": str(function.get("description", "")).strip(),
                        "parameters": function.get("parameters") if isinstance(function.get("parameters"), dict) else {},
                    }
                )
                continue
            if item.get("type") == "function" and isinstance(item.get("name"), str):
                converted.append(item)
        return converted

    @staticmethod
    def responses_tools_to_chat(tools: list[JsonObject]) -> list[JsonObject]:
        converted: list[JsonObject] = []
        for item in tools:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("function"), dict):
                converted.append(item)
                continue
            if str(item.get("type", "")).strip() != "function":
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(item.get("description", "")).strip(),
                        "parameters": item.get("parameters") if isinstance(item.get("parameters"), dict) else {},
                    },
                }
            )
        return converted

    def payload_to_mode(
        self,
        payload: dict[str, object],
        mode: str,
        *,
        default_max_tokens: int | None,
    ) -> JsonObject:
        if mode not in CONCRETE_ENDPOINT_MODES:
            mode = ENDPOINT_MODE_CHAT
        messages = payload.get("messages")
        if not isinstance(messages, list):
            messages = payload.get("input")
        message_list = [item for item in messages] if isinstance(messages, list) else []
        tools_raw = payload.get("tools")
        tools = [item for item in tools_raw] if isinstance(tools_raw, list) else None
        thinking = False
        template = payload.get("chat_template_kwargs")
        if isinstance(template, dict):
            thinking = bool(template.get("enable_thinking"))
        max_tokens_override = None
        max_output_tokens = payload.get("max_output_tokens")
        max_tokens = payload.get("max_tokens")
        if isinstance(max_output_tokens, (int, float)):
            max_tokens_override = int(max_output_tokens)
        elif isinstance(max_tokens, (int, float)):
            max_tokens_override = int(max_tokens)
        model_override = str(payload.get("model", "")).strip()
        converted_tools: list[JsonObject] | None = None
        if tools is not None:
            converted_tools = self.chat_tools_to_responses(tools) if mode == ENDPOINT_MODE_RESPONSES else self.responses_tools_to_chat(tools)
        return self.build_payload(
            model_messages=cast(list[JsonObject], message_list),
            thinking=thinking,
            tools=converted_tools,
            max_tokens_override=max_tokens_override,
            model_override=model_override,
            mode=mode,
            default_max_tokens=default_max_tokens,
        )

    def build_payload(
        self,
        *,
        model_messages: list[JsonObject],
        thinking: bool,
        tools: list[JsonObject] | None = None,
        max_tokens_override: int | None = None,
        model_override: str = "",
        mode: str,
        default_max_tokens: int | None,
    ) -> JsonObject:
        selected_mode = mode if mode in CONCRETE_ENDPOINT_MODES else ENDPOINT_MODE_CHAT
        limit = default_max_tokens if max_tokens_override is None else max_tokens_override
        if selected_mode == ENDPOINT_MODE_RESPONSES:
            payload: JsonObject = {
                "input": cast(JSONValue, model_messages),
                "stream": True,
            }
            if limit is not None and int(limit) > 0:
                payload["max_output_tokens"] = int(limit)
            if tools:
                payload["tools"] = cast(JSONValue, self.chat_tools_to_responses(tools))
                payload["tool_choice"] = "auto"
            if model_override.strip():
                payload["model"] = model_override.strip()
            if thinking:
                payload["reasoning"] = {"summary": "auto"}
            return payload

        payload: JsonObject = {
            "messages": cast(JSONValue, model_messages),
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        }
        if limit is not None and int(limit) > 0:
            payload["max_tokens"] = int(limit)
        if tools:
            payload["tools"] = cast(JSONValue, tools)
            payload["tool_choice"] = "auto"
        if model_override.strip():
            payload["model"] = model_override.strip()
        return payload
