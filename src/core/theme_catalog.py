from __future__ import annotations

from collections.abc import Iterable
from typing import Final

DEFAULT_THEME_ID: Final[str] = "catppuccin-mocha"

BUILTIN_THEME_IDS: Final[tuple[str, ...]] = (
    "classic",
    "soft",
    "catppuccin-mocha",
    "catppuccin-macchiato",
    "tokyonight-moon",
    "gruvbox-dark-soft",
    "dracula",
    "nord",
    "rose-pine-moon",
    "ayu-dark",
    "one-dark-pro",
)

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
    if available is not None:
        available_set = {str(item).strip().lower() for item in available}
        return (aliased, False) if aliased in available_set else (default, True)
    if aliased in BUILTIN_THEME_IDS:
        return aliased, False
    return default, True
