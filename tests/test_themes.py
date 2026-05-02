from __future__ import annotations

from tui.themes import theme_spec


def test_catppuccin_themes_use_official_palette_values() -> None:
    mocha = theme_spec("catppuccin-mocha")
    macchiato = theme_spec("catppuccin-macchiato")

    assert mocha.theme.primary == "#cba6f7"
    assert mocha.theme.secondary == "#b4befe"
    assert mocha.theme.foreground == "#cdd6f4"
    assert mocha.theme.background == "#1e1e2e"
    assert mocha.theme.surface == "#313244"
    assert mocha.theme.panel == "#181825"
    assert mocha.theme.variables["app-selection-bg"] == "#45475a"
    assert mocha.colors["panel_border"] == "#6c7086"

    assert macchiato.theme.primary == "#c6a0f6"
    assert macchiato.theme.secondary == "#b7bdf8"
    assert macchiato.theme.foreground == "#cad3f5"
    assert macchiato.theme.background == "#24273a"
    assert macchiato.theme.surface == "#363a4f"
    assert macchiato.theme.panel == "#1e2030"
    assert macchiato.theme.variables["app-selection-bg"] == "#494d64"
    assert macchiato.colors["panel_border"] == "#6e738d"
