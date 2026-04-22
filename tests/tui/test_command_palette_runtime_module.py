from __future__ import annotations

from types import SimpleNamespace

from tui.command_palette_runtime import refresh_command_popup


class _Styles:
    def __init__(self) -> None:
        self.height = 0
        self.width = 0


class _Popup:
    def __init__(self) -> None:
        self.display = False
        self.styles = _Styles()
        self.offset = (0, 0)


class _Options:
    def __init__(self) -> None:
        self.styles = _Styles()
        self.highlighted = None
        self.items = []

    def clear_options(self) -> None:
        self.items = []

    def add_options(self, rendered) -> None:
        self.items.extend(rendered)


class _App:
    def __init__(self, *, separator_y: int) -> None:
        self.streaming = False
        self._await_shell_confirm = False
        self._command_matches = []
        self._popup = _Popup()
        self._options = _Options()
        self.chat_input = SimpleNamespace(
            value="/",
            cursor_position=1,
            region=SimpleNamespace(x=2, y=12, width=58, height=1),
            size=SimpleNamespace(width=58, height=1),
        )
        self.separator = SimpleNamespace(region=SimpleNamespace(y=separator_y))

    def _theme_color(self, _token: str, default: str) -> str:
        return default

    def _command_popup(self) -> _Popup:
        return self._popup

    def _command_options(self) -> _Options:
        return self._options

    def query_one(self, selector, _widget_type=None):
        if selector == "#footer-sep":
            return self.separator
        return self.chat_input


def test_refresh_command_popup_keeps_bottom_at_separator() -> None:
    app = _App(separator_y=18)

    refresh_command_popup(app, "/", chat_input_cls=object)

    assert app._popup.display is True
    assert app._popup.styles.height == 13
    assert app._popup.offset == (2, 5)
    assert app._popup.offset[1] + app._popup.styles.height == 18


def test_refresh_command_popup_clamps_height_when_vertical_space_is_tight() -> None:
    app = _App(separator_y=4)

    refresh_command_popup(app, "/", chat_input_cls=object)

    assert app._popup.display is True
    assert app._popup.styles.height == 3
    assert app._popup.offset == (2, 1)
    assert app._options.styles.height == 1
