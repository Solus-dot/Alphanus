from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import Final

DEFAULT_THEME_ID: Final[str] = "catppuccin-mocha"

THEME_ALIASES: Final[dict[str, str]] = {
    "catppuccin": "catppuccin-mocha",
    "rose-pine": "rose-pine-moon",
    "onedark": "one-dark-pro",
    "one-dark": "one-dark-pro",
}


def normalize_theme_id(raw: str, *, default: str = DEFAULT_THEME_ID, available: Iterable[str] | None = None) -> tuple[str, bool]:
    text = str(raw or "").strip().lower()
    if not text:
        return default, True
    aliased = THEME_ALIASES.get(text, text)
    available_set: Collection[str] = {str(item).strip().lower() for item in available or ()}
    return (aliased, False) if not available_set or aliased in available_set else (default, True)
