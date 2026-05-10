from __future__ import annotations

from typing import NotRequired, TypedDict

from core.message_types import JSONValue


class ToolError(TypedDict):
    code: str
    message: str


class ToolResult(TypedDict):
    ok: bool
    data: JSONValue
    error: ToolError | None
    meta: dict[str, JSONValue]


class ToolResultMeta(TypedDict):
    duration_ms: NotRequired[int]


def ok_result(data: JSONValue, *, duration_ms: int | None = None) -> ToolResult:
    meta: dict[str, JSONValue] = {}
    if duration_ms is not None:
        meta["duration_ms"] = duration_ms
    return {"ok": True, "data": data, "error": None, "meta": meta}


def error_result(code: str, message: str, *, duration_ms: int | None = None) -> ToolResult:
    meta: dict[str, JSONValue] = {}
    if duration_ms is not None:
        meta["duration_ms"] = duration_ms
    return {
        "ok": False,
        "data": None,
        "error": {"code": code, "message": message},
        "meta": meta,
    }
