from __future__ import annotations

from typing import cast

from core.message_types import JSONValue


def coerce_int(
    value: object,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    parsed: int
    if isinstance(value, bool):
        parsed = default
    elif isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        parsed = int(value)
    elif isinstance(value, str):
        try:
            parsed = int(value.strip())
        except Exception:
            parsed = default
    else:
        parsed = default

    if minimum is not None and parsed < minimum:
        parsed = minimum
    if maximum is not None and parsed > maximum:
        parsed = maximum
    return parsed


def as_json_object(value: object) -> dict[str, JSONValue]:
    if isinstance(value, dict):
        return cast(dict[str, JSONValue], value)
    return {}


def get_json_object(source: dict[str, JSONValue], key: str) -> dict[str, JSONValue]:
    return as_json_object(source.get(key))
