from __future__ import annotations

from typing import Final

TRUE_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


def parse_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in TRUE_VALUES:
            return True
        if lowered in FALSE_VALUES:
            return False
    return None


def coerce_bool(value: object, default: bool) -> bool:
    parsed = parse_bool(value)
    return default if parsed is None else parsed
