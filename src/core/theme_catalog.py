from __future__ import annotations

from typing import Final

DEFAULT_THEME_ID: Final[str] = "catppuccin-mocha"

BUILTIN_THEME_IDS: Final[tuple[str, ...]] = (
    "classic",
    "soft",
    "catppuccin-mocha",
    "catppuccin-macchiato",
    "tokyonight-moon",
    "gruvbox-dark-soft",
)

THEME_ALIASES: Final[dict[str, str]] = {
    "catppuccin": "catppuccin-mocha",
}


def normalize_theme_id(raw: str, *, default: str = DEFAULT_THEME_ID) -> tuple[str, bool]:
    text = str(raw or "").strip().lower()
    if not text:
        return default, True
    aliased = THEME_ALIASES.get(text, text)
    if aliased in BUILTIN_THEME_IDS:
        return aliased, False
    return default, True
