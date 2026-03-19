import json
from pathlib import Path
from typing import Any, Dict, Optional

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea


class CodeViewerModal(ModalScreen[None]):
    CSS = """
    CodeViewerModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.72);
    }

    #code-modal {
        width: 88%;
        height: 88%;
        background: #121214;
        border: panel #27272a;
        padding: 1 2;
    }

    #code-modal-title {
        color: #e4e4e7;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #code-modal-editor {
        width: 1fr;
        height: 1fr;
        background: #18181b;
        color: #e4e4e7;
        border: solid #27272a;
    }

    #code-modal-footer {
        height: auto;
        align-horizontal: right;
        padding: 1 0 0 0;
    }

    #code-modal-hint {
        width: 1fr;
        color: #a1a1aa;
        padding-top: 1;
    }

    #code-copy {
        background: #6366f1;
        color: #ffffff;
        border: none;
        margin-right: 1;
    }

    #code-close {
        background: #18181b;
        color: #e4e4e7;
        border: none;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", show=False),
        Binding("ctrl+shift+c", "copy_all", show=False),
    ]

    def __init__(self, code: str, language: Optional[str], title: str = "Code Block") -> None:
        super().__init__()
        self._code = code
        self._language = language or "text"
        self._title = title

    def compose(self) -> ComposeResult:
        with Vertical(id="code-modal"):
            yield Static(self._title, id="code-modal-title")
            yield TextArea(
                self._code,
                language=self._language,
                theme="dracula",
                read_only=True,
                show_line_numbers=False,
                compact=True,
                id="code-modal-editor",
            )
            with Horizontal(id="code-modal-footer"):
                yield Static("select text or ctrl+shift+c to copy all", id="code-modal-hint")
                yield Button("Copy All", id="code-copy", variant="primary")
                yield Button("Close", id="code-close")

    def on_mount(self) -> None:
        self.query_one("#code-modal-editor", TextArea).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_copy_all(self) -> None:
        self.app.copy_to_clipboard(self._code)

    @on(Button.Pressed, "#code-copy")
    def _copy_all(self) -> None:
        self.action_copy_all()

    @on(Button.Pressed, "#code-close")
    def _close(self) -> None:
        self.action_cancel()


class ConfigEditorModal(ModalScreen[Optional[Dict[str, Any]]]):
    CSS = """
    ConfigEditorModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.74);
    }

    #config-modal {
        width: 90%;
        height: 90%;
        background: #121214;
        border: panel #27272a;
        padding: 1 2;
    }

    #config-modal-title {
        color: #e4e4e7;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #config-modal-subtitle {
        color: #a1a1aa;
        padding: 0 0 1 0;
    }

    #config-modal-note {
        color: #c4b5fd;
        padding: 0 0 1 0;
    }

    #config-modal-editor {
        width: 1fr;
        height: 1fr;
        background: #18181b;
        color: #e4e4e7;
        border: solid #27272a;
    }

    #config-modal-error {
        min-height: 1;
        color: #f87171;
        padding: 1 0 0 0;
    }

    #config-modal-footer {
        height: auto;
        align-horizontal: right;
        padding: 1 0 0 0;
    }

    #config-save {
        background: #10b981;
        color: #ffffff;
        border: none;
        margin-right: 1;
    }

    #config-cancel {
        background: #18181b;
        color: #e4e4e7;
        border: none;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", show=False),
        Binding("ctrl+s", "save", show=False),
    ]

    def __init__(self, config_path: Path, initial_text: str) -> None:
        super().__init__()
        self._config_path = config_path
        self._initial_text = initial_text

    def compose(self) -> ComposeResult:
        with Vertical(id="config-modal"):
            yield Static("Global Config", id="config-modal-title")
            yield Static(str(self._config_path), id="config-modal-subtitle")
            yield Static(
                "Secrets are omitted here. Use environment variables such as TAVILY_API_KEY or BRAVE_SEARCH_API_KEY.",
                id="config-modal-note",
            )
            yield TextArea(
                self._initial_text,
                language="json",
                theme="dracula",
                read_only=False,
                show_line_numbers=True,
                compact=True,
                id="config-modal-editor",
            )
            yield Static("", id="config-modal-error")
            with Horizontal(id="config-modal-footer"):
                yield Button("Save", id="config-save", variant="success")
                yield Button("Cancel", id="config-cancel")

    def on_mount(self) -> None:
        editor = self.query_one("#config-modal-editor", TextArea)
        editor.focus()
        editor.cursor_location = (0, 0)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_save(self) -> None:
        self._save()

    @on(Button.Pressed, "#config-save")
    def _save_button(self) -> None:
        self.action_save()

    @on(Button.Pressed, "#config-cancel")
    def _cancel_button(self) -> None:
        self.action_cancel()

    def _save(self) -> None:
        editor = self.query_one("#config-modal-editor", TextArea)
        raw = editor.text
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.query_one("#config-modal-error", Static).update(
                f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
            )
            return

        if not isinstance(parsed, dict):
            self.query_one("#config-modal-error", Static).update("Global config must be a JSON object.")
            return

        formatted = json.dumps(parsed, indent=2) + "\n"
        self.dismiss({"config": parsed, "text": formatted})
