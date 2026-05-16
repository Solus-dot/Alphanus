from __future__ import annotations

from types import SimpleNamespace

from tui.themes import theme_spec


def _write_custom_theme(theme_dir, theme_id: str = "solar-test") -> None:
    theme_dir.mkdir()
    (theme_dir / f"{theme_id}.json").write_text(
        f"""
{{
  "id": "{theme_id}",
  "title": "Solar Test",
  "description": "Custom test theme",
  "syntax_theme": "github-dark",
  "text_area_theme": "dracula",
  "theme": {{
    "primary": "#111111",
    "secondary": "#222222",
    "accent": "#333333",
    "foreground": "#444444",
    "background": "#050505",
    "surface": "#101010",
    "panel": "#000000",
    "success": "#00ff00",
    "warning": "#ffff00",
    "error": "#ff0000",
    "dark": true,
    "variables": {{
      "app-muted": "#777777",
      "app-selection-bg": "#222222"
    }}
  }},
  "colors": {{
    "accent": "#333333",
    "text": "#444444",
    "muted": "#777777",
    "subtle": "#666666",
    "success": "#00ff00",
    "warning": "#ffff00",
    "error": "#ff0000",
    "user_bar": "#00ff00",
    "assistant_bar": "#333333",
    "chip_bg": "#111111",
    "chip_text": "#444444",
    "panel_bg": "#000000",
    "panel_border": "#555555"
  }}
}}
""".strip(),
        encoding="utf-8",
    )


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


def test_custom_theme_json_loads_from_theme_path(tmp_path, monkeypatch) -> None:
    theme_dir = tmp_path / "themes"
    _write_custom_theme(theme_dir)
    monkeypatch.setenv("ALPHANUS_THEME_PATHS", str(theme_dir))

    from tui import themes

    themes.reload_theme_specs()
    try:
        spec = themes.theme_spec("solar-test")
        assert "solar-test" in themes.available_theme_ids()
        assert spec.title == "Solar Test"
        assert spec.theme.primary == "#111111"
        assert spec.colors["panel_border"] == "#555555"
    finally:
        themes.reload_theme_specs()


def test_unknown_theme_falls_back_to_default() -> None:
    from tui import themes

    themes.reload_theme_specs()
    assert themes.theme_spec("missing-theme").id == "catppuccin-mocha"


def test_runtime_config_preserves_loadable_custom_theme(tmp_path, monkeypatch) -> None:
    theme_dir = tmp_path / "themes"
    _write_custom_theme(theme_dir)
    monkeypatch.setenv("ALPHANUS_THEME_PATHS", str(theme_dir))

    from core.runtime_config import UiRuntimeConfig
    from tui import themes

    themes.reload_theme_specs()
    try:
        ui = UiRuntimeConfig.from_config({"tui": {"theme": "solar-test"}})
        assert ui.theme == "solar-test"
    finally:
        themes.reload_theme_specs()


def test_theme_controller_applies_loadable_custom_theme(tmp_path, monkeypatch) -> None:
    theme_dir = tmp_path / "themes"
    _write_custom_theme(theme_dir)
    monkeypatch.setenv("ALPHANUS_THEME_PATHS", str(theme_dir))

    from tui import themes
    from tui.theme_controller import ThemeController

    class AppStub:
        def __init__(self) -> None:
            self._themes_registered = False
            self._live_preview = SimpleNamespace(set_theme_colors=lambda **_kwargs: None)
            self._ui_config = SimpleNamespace(theme="solar-test")
            self.streaming = False
            self.registered: list[str] = []

        def register_theme(self, theme) -> None:
            self.registered.append(theme.name)

        def query_one(self, *_args, **_kwargs):
            raise LookupError

        def refresh_css(self, *_args, **_kwargs) -> None:
            return None

        def __getattr__(self, name: str):
            if name == "_apply_focus_classes" or name.startswith("_update_") or name.startswith("_refresh_"):
                return lambda *_args, **_kwargs: None
            raise AttributeError(name)

    themes.reload_theme_specs()
    try:
        app = AppStub()
        controller = ThemeController(app)
        assert controller.apply_theme("solar-test") == "solar-test"
        assert controller.theme_id() == "solar-test"
        assert app.theme == "solar-test"
        assert "solar-test" in app.registered
    finally:
        themes.reload_theme_specs()
