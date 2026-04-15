from __future__ import annotations

from typing import Callable, Literal, TypeAlias, TypedDict

JSONPrimitive = str | int | float | bool | None
JSONValue: TypeAlias = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]
JsonObject: TypeAlias = dict[str, JSONValue]
ShellConfirmationFn: TypeAlias = Callable[[str], bool]
UserInputRequestFn: TypeAlias = Callable[[JsonObject], JsonObject]


class ToolFunctionCall(TypedDict, total=False):
    name: str
    arguments: str


class ToolCallDelta(TypedDict, total=False):
    index: int
    id: str
    type: Literal["function"]
    function: ToolFunctionCall


class ToolCallUpdate(TypedDict):
    index: int
    stream_id: str
    id: str
    name: str
    raw_arguments: str


class MessageContentPart(TypedDict, total=False):
    type: str
    text: str
    image_url: dict[str, JSONValue]
    video_url: dict[str, JSONValue]


class ChatMessage(TypedDict, total=False):
    role: str
    content: JSONValue | list[MessageContentPart]
    name: str
    tool_call_id: str
    tool_calls: list[ToolCallDelta]
