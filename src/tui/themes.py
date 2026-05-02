from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from textual.theme import Theme

from core.theme_catalog import BUILTIN_THEME_IDS, DEFAULT_THEME_ID, normalize_theme_id


@dataclass(frozen=True, slots=True)
class ThemeSpec:
    id: str
    title: str
    description: str
    theme: Theme
    syntax_theme: str
    text_area_theme: str
    colors: dict[str, str]


def _theme_variables(*, muted: str, subtle: str, border: str, chip_bg: str, chip_text: str, selection_bg: str) -> dict[str, str]:
    return {
        "app-muted": muted,
        "app-subtle": subtle,
        "app-border": border,
        "app-chip-bg": chip_bg,
        "app-chip-text": chip_text,
        "app-selection-bg": selection_bg,
    }


THEME_SPECS: Final[dict[str, ThemeSpec]] = {
    "classic": ThemeSpec(
        id="classic",
        title="Classic",
        description="Original high-contrast Alphanus dark theme",
        theme=Theme(
            name="classic",
            primary="#6366f1",
            secondary="#818cf8",
            accent="#6366f1",
            foreground="#e4e4e7",
            background="#09090b",
            surface="#18181b",
            panel="#000000",
            success="#10b981",
            warning="#f59e0b",
            error="#f43f5e",
            dark=True,
            variables=_theme_variables(
                muted="#a1a1aa",
                subtle="#71717a",
                border="#52525b",
                chip_bg="#1a1730",
                chip_text="#f4f4f5",
                selection_bg="#1a1730",
            ),
        ),
        syntax_theme="github-dark",
        text_area_theme="dracula",
        colors={
            "accent": "#6366f1",
            "text": "#f4f4f5",
            "muted": "#a1a1aa",
            "subtle": "#71717a",
            "success": "#10b981",
            "warning": "#f59e0b",
            "error": "#f87171",
            "user_bar": "#10b981",
            "assistant_bar": "#6366f1",
            "chip_bg": "#1a1730",
            "chip_text": "#f4f4f5",
            "panel_bg": "#09090b",
            "panel_border": "#27272a",
        },
    ),
    "soft": ThemeSpec(
        id="soft",
        title="Soft",
        description="Low-glare forest slate with warm text",
        theme=Theme(
            name="soft",
            primary="#7fbbb3",
            secondary="#a7c080",
            accent="#7fbbb3",
            foreground="#d3c6aa",
            background="#232a2e",
            surface="#2d353b",
            panel="#1f2528",
            success="#a7c080",
            warning="#dbbc7f",
            error="#e67e80",
            dark=True,
            variables=_theme_variables(
                muted="#9da9a0",
                subtle="#859289",
                border="#4f5b58",
                chip_bg="#34403d",
                chip_text="#d3c6aa",
                selection_bg="#3d4a46",
            ),
        ),
        syntax_theme="github-dark",
        text_area_theme="dracula",
        colors={
            "accent": "#7fbbb3",
            "text": "#d3c6aa",
            "muted": "#9da9a0",
            "subtle": "#859289",
            "success": "#a7c080",
            "warning": "#dbbc7f",
            "error": "#e67e80",
            "user_bar": "#a7c080",
            "assistant_bar": "#7fbbb3",
            "chip_bg": "#34403d",
            "chip_text": "#d3c6aa",
            "panel_bg": "#1f2528",
            "panel_border": "#4f5b58",
        },
    ),
    "catppuccin-mocha": ThemeSpec(
        id="catppuccin-mocha",
        title="Catppuccin Mocha",
        description="Official Catppuccin Mocha flavor",
        theme=Theme(
            name="catppuccin-mocha",
            primary="#cba6f7",
            secondary="#b4befe",
            accent="#cba6f7",
            foreground="#cdd6f4",
            background="#1e1e2e",
            surface="#313244",
            panel="#181825",
            success="#a6e3a1",
            warning="#f9e2af",
            error="#f38ba8",
            dark=True,
            variables=_theme_variables(
                muted="#a6adc8",
                subtle="#7f849c",
                border="#6c7086",
                chip_bg="#313244",
                chip_text="#cdd6f4",
                selection_bg="#45475a",
            ),
        ),
        syntax_theme="dracula",
        text_area_theme="dracula",
        colors={
            "accent": "#cba6f7",
            "text": "#cdd6f4",
            "muted": "#a6adc8",
            "subtle": "#7f849c",
            "success": "#a6e3a1",
            "warning": "#f9e2af",
            "error": "#f38ba8",
            "user_bar": "#a6e3a1",
            "assistant_bar": "#cba6f7",
            "chip_bg": "#313244",
            "chip_text": "#cdd6f4",
            "panel_bg": "#181825",
            "panel_border": "#6c7086",
        },
    ),
    "catppuccin-macchiato": ThemeSpec(
        id="catppuccin-macchiato",
        title="Catppuccin Macchiato",
        description="Official Catppuccin Macchiato flavor",
        theme=Theme(
            name="catppuccin-macchiato",
            primary="#c6a0f6",
            secondary="#b7bdf8",
            accent="#c6a0f6",
            foreground="#cad3f5",
            background="#24273a",
            surface="#363a4f",
            panel="#1e2030",
            success="#a6da95",
            warning="#eed49f",
            error="#ed8796",
            dark=True,
            variables=_theme_variables(
                muted="#a5adcb",
                subtle="#8087a2",
                border="#6e738d",
                chip_bg="#363a4f",
                chip_text="#cad3f5",
                selection_bg="#494d64",
            ),
        ),
        syntax_theme="dracula",
        text_area_theme="dracula",
        colors={
            "accent": "#c6a0f6",
            "text": "#cad3f5",
            "muted": "#a5adcb",
            "subtle": "#8087a2",
            "success": "#a6da95",
            "warning": "#eed49f",
            "error": "#ed8796",
            "user_bar": "#a6da95",
            "assistant_bar": "#c6a0f6",
            "chip_bg": "#363a4f",
            "chip_text": "#cad3f5",
            "panel_bg": "#1e2030",
            "panel_border": "#6e738d",
        },
    ),
    "tokyonight-moon": ThemeSpec(
        id="tokyonight-moon",
        title="Tokyo Night Moon",
        description="Cool moonlit dark with restrained electric accents",
        theme=Theme(
            name="tokyonight-moon",
            primary="#82aaff",
            secondary="#c099ff",
            accent="#82aaff",
            foreground="#c8d3f5",
            background="#1e2030",
            surface="#2a2e45",
            panel="#1a1b28",
            success="#86e1fc",
            warning="#ffc777",
            error="#ff757f",
            dark=True,
            variables=_theme_variables(
                muted="#9aa7de",
                subtle="#7a88cf",
                border="#3a4165",
                chip_bg="#2b3560",
                chip_text="#d5deff",
                selection_bg="#3c4b83",
            ),
        ),
        syntax_theme="github-dark",
        text_area_theme="dracula",
        colors={
            "accent": "#82aaff",
            "text": "#d5deff",
            "muted": "#9aa7de",
            "subtle": "#7a88cf",
            "success": "#86e1fc",
            "warning": "#ffc777",
            "error": "#ff757f",
            "user_bar": "#86e1fc",
            "assistant_bar": "#82aaff",
            "chip_bg": "#2b3560",
            "chip_text": "#d5deff",
            "panel_bg": "#1a1b28",
            "panel_border": "#3a4165",
        },
    ),
    "gruvbox-dark-soft": ThemeSpec(
        id="gruvbox-dark-soft",
        title="Gruvbox Dark Soft",
        description="Warm low-contrast earthy dark palette",
        theme=Theme(
            name="gruvbox-dark-soft",
            primary="#83a598",
            secondary="#d3869b",
            accent="#fabd2f",
            foreground="#ebdbb2",
            background="#32302f",
            surface="#3c3836",
            panel="#282828",
            success="#b8bb26",
            warning="#fabd2f",
            error="#fb4934",
            dark=True,
            variables=_theme_variables(
                muted="#bdae93",
                subtle="#a89984",
                border="#504945",
                chip_bg="#4a443f",
                chip_text="#f2e5bc",
                selection_bg="#665c54",
            ),
        ),
        syntax_theme="github-dark",
        text_area_theme="dracula",
        colors={
            "accent": "#fabd2f",
            "text": "#f2e5bc",
            "muted": "#bdae93",
            "subtle": "#a89984",
            "success": "#b8bb26",
            "warning": "#fabd2f",
            "error": "#fb4934",
            "user_bar": "#8ec07c",
            "assistant_bar": "#fabd2f",
            "chip_bg": "#4a443f",
            "chip_text": "#f2e5bc",
            "panel_bg": "#282828",
            "panel_border": "#504945",
        },
    ),
}

_FALLBACK_THEME_ID: Final[str] = "classic"
_FALLBACK_THEME_SPEC: Final[ThemeSpec] = THEME_SPECS[_FALLBACK_THEME_ID]
_FALLBACK_THEME_VARIABLES: Final[dict[str, str]] = {key: str(value) for key, value in (_FALLBACK_THEME_SPEC.theme.variables or {}).items()}
_DEFAULT_THEME_SPEC: Final[ThemeSpec] = THEME_SPECS[DEFAULT_THEME_ID]

FALLBACK_COLORS: Final[dict[str, str]] = {
    **{key: str(value) for key, value in _FALLBACK_THEME_SPEC.colors.items()},
    "badge_bg": str(_FALLBACK_THEME_SPEC.colors["chip_bg"]),
}


def available_theme_ids() -> list[str]:
    return list(BUILTIN_THEME_IDS)


def theme_spec(theme_id: str) -> ThemeSpec:
    resolved, _ = normalize_theme_id(theme_id, default=DEFAULT_THEME_ID)
    return THEME_SPECS[resolved]


def fallback_color(key: str, default: str = "") -> str:
    return str(FALLBACK_COLORS.get(key, default))


def fallback_theme_variables() -> dict[str, str]:
    return dict(_FALLBACK_THEME_VARIABLES)


def default_theme_variables() -> dict[str, str]:
    variables = _DEFAULT_THEME_SPEC.theme.variables or {}
    merged = fallback_theme_variables()
    merged.update({key: str(value) for key, value in variables.items()})
    return merged
