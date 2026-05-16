from __future__ import annotations

from textual.css.query import NoMatches

from core.runtime_config import UiRuntimeConfig
from core.theme_catalog import DEFAULT_THEME_ID, normalize_theme_id
from tui.chat_input import ChatInput, _PasteTokenHighlighter
from tui.themes import ThemeSpec, available_theme_ids, default_theme_variables, fallback_color, theme_spec

DEFAULT_ACCENT_COLOR = fallback_color("accent")
DEFAULT_PANEL_BG = fallback_color("panel_bg")
DEFAULT_SUBTLE_COLOR = fallback_color("subtle")
DEFAULT_MUTED_COLOR = fallback_color("muted")


class ThemeController:
    def __init__(self, app) -> None:
        self.app = app

    @staticmethod
    def variable_defaults() -> dict[str, str]:
        return default_theme_variables()

    @staticmethod
    def _normalize_theme_id(raw_theme_id: str, *, default: str = DEFAULT_THEME_ID) -> str:
        resolved, _ = normalize_theme_id(raw_theme_id, default=default, available=available_theme_ids())
        return resolved

    def timing_config(self):
        timing = getattr(self.app, "_ui_timing", None)
        if timing is None:
            timing = UiRuntimeConfig.from_config({}).timing
            self.app._ui_timing = timing
        return timing

    def theme_id(self) -> str:
        current = str(getattr(self.app, "_active_theme_id", "") or "").strip().lower()
        if current:
            return self._normalize_theme_id(current)
        ui_cfg = getattr(self.app, "_ui_config", None)
        configured = str(getattr(ui_cfg, "theme", "") or "").strip().lower()
        return self._normalize_theme_id(configured)

    def theme_spec(self) -> ThemeSpec:
        return theme_spec(self.theme_id())

    def theme_color(self, key: str, default: str) -> str:
        spec = self.theme_spec()
        return str(spec.colors.get(key, default))

    def register_themes(self) -> None:
        if getattr(self.app, "_themes_registered", False):
            return
        for theme_id in available_theme_ids():
            self.app.register_theme(theme_spec(theme_id).theme)
        self.app._themes_registered = True

    def apply_theme(self, raw_theme_id: str) -> str:
        resolved = self._normalize_theme_id(raw_theme_id)
        self.register_themes()
        self.app.theme = resolved
        self.app._active_theme_id = resolved
        style = f"bold {self.theme_color('accent', DEFAULT_ACCENT_COLOR)} on {self.theme_color('panel_bg', DEFAULT_PANEL_BG)}"
        _PasteTokenHighlighter.STYLE = style
        ChatInput.PASTE_TOKEN_STYLE = style
        set_preview_theme = getattr(self.app._live_preview, "set_theme_colors", None)
        if callable(set_preview_theme):
            set_preview_theme(
                label_color=self.theme_color("subtle", DEFAULT_SUBTLE_COLOR),
                muted_color=self.theme_color("muted", DEFAULT_MUTED_COLOR),
            )
        try:
            chat_input = self.app.query_one(ChatInput)
        except Exception:
            chat_input = None
        if chat_input is not None:
            highlighter = getattr(chat_input, "highlighter", None)
            if isinstance(highlighter, _PasteTokenHighlighter):
                highlighter.STYLE = style
            chat_input.sync_paste_placeholders(chat_input.value)
        try:
            self.app.refresh_css(animate=False)
            self.app._apply_focus_classes()
            self.app._update_topbar()
            self.app._update_status1()
            self.app._update_status2()
            self.app._update_footer_separator()
            self.app._update_sidebar()
            self.app._update_pending_attachments()
            if self.app.streaming:
                self.app._refresh_deferred_partial()
                self.app._refresh_live_preview_partial()
            self.app._refresh_themed_transcript_entries()
            self.app._refresh_command_popup_for_resize()
        except NoMatches:
            pass
        return resolved

    def apply_theme_from_config(self) -> str:
        configured = getattr(self.app, "_ui_config", None)
        configured_theme = str(getattr(configured, "theme", DEFAULT_THEME_ID))
        try:
            return self.apply_theme(configured_theme)
        except Exception:
            resolved = self._normalize_theme_id(configured_theme)
            self.app._active_theme_id = resolved
            return resolved
