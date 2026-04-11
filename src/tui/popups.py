import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional

from rich.markup import escape as esc
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from core.sessions import SessionSummary


class CodeViewerModal(ModalScreen[None]):
    CSS = """
    CodeViewerModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.72);
    }

    #code-modal {
        width: 88%;
        height: 88%;
        background: #000000;
        border: panel #52525b;
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
        background: #000000;
        color: #e4e4e7;
        border: solid #52525b;
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
        background: #000000;
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
        background: #000000;
        border: panel #52525b;
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
        color: #6366f1;
        padding: 0 0 1 0;
    }

    #config-modal-editor {
        width: 1fr;
        height: 1fr;
        background: #000000;
        color: #e4e4e7;
        border: solid #52525b;
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
        background: #000000;
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


class SessionManagerModal(ModalScreen[Optional[Dict[str, str]]]):
    CSS = """
    SessionManagerModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.76);
    }

    #session-modal {
        width: 68;
        max-width: 92%;
        height: auto;
        background: #09090b;
        border: solid #52525b;
        padding: 0 1 1 2;
    }

    #session-modal-kicker {
        color: #71717a;
        padding: 0;
    }

    #session-modal-subtitle {
        color: #a1a1aa;
        padding: 0 0 1 0;
    }

    #session-modal-hint {
        color: #71717a;
        padding: 0 0 1 0;
    }

    #session-modal-list-label,
    #session-modal-new-label {
        color: #6366f1;
        padding: 0 0 1 0;
    }

    #session-modal-list {
        width: 1fr;
        height: auto;
        max-height: 10;
        background: #000000;
        border: solid #52525b;
        margin: 0 0 1 0;
        padding: 0;
    }

    #session-modal-list > .option-list--option-highlighted {
        color: #ffffff;
        background: #1a1730;
    }

    #session-modal-name {
        width: 1fr;
        margin: 0 0 1 0;
        background: #000000;
        color: #e4e4e7;
        border: round #63636b;
    }

    #session-modal-footer {
        width: 1fr;
        height: auto;
        align-horizontal: left;
        padding: 0;
    }

    #session-open,
    #session-delete,
    #session-new {
        width: 1fr;
        min-width: 0;
        border: none;
        content-align: center middle;
        margin-right: 1;
    }

    #session-open {
        background: #1a1730;
        color: #c7d2fe;
    }

    #session-delete {
        background: #26181a;
        color: #f0c4c8;
    }

    #session-new {
        background: #15241f;
        color: #b9ead9;
        margin-right: 0;
    }

    """

    BINDINGS = [
        Binding("escape", "cancel", show=False),
        Binding("enter", "open_selected", show=False),
        Binding("1", "open_selected", show=False),
        Binding("2", "delete_selected", show=False),
        Binding("3", "create_new", show=False),
    ]

    def __init__(self, sessions: list[SessionSummary], active_session_id: str) -> None:
        super().__init__()
        self._sessions = sessions
        self._active_session_id = active_session_id
        self._pending_delete_session_id = ""

    def compose(self) -> ComposeResult:
        with Vertical(id="session-modal"):
            yield Static("SESSIONS", id="session-modal-kicker")
            yield Static("Open, create, or delete sessions from one place.", id="session-modal-subtitle")
            yield Static("Shortcuts: 1. Open · 2. Delete · 3. New", id="session-modal-hint")
            if self._sessions:
                yield Static("Saved Sessions", id="session-modal-list-label")
                yield OptionList(*self._session_options(), id="session-modal-list")
            yield Static("New Session", id="session-modal-new-label")
            yield Input(
                placeholder="Leave blank to use the next session name",
                value="",
                id="session-modal-name",
            )
            with Horizontal(id="session-modal-footer"):
                yield Button("1. Open Selected", id="session-open", variant="primary", disabled=not self._sessions)
                yield Button("2. Delete Selected", id="session-delete", variant="default", disabled=not self._sessions)
                yield Button("3. New Session", id="session-new", variant="success")

    def _session_options(self) -> list[Option]:
        options: list[Option] = []
        for index, session in enumerate(self._sessions, start=1):
            marker = (
                "[bold #6366f1]active[/bold #6366f1]"
                if session.is_active
                else "[#71717a]saved[/#71717a]"
            )
            title = (
                f"[bold #6366f1]{esc(session.title)}[/bold #6366f1]"
                if session.is_active
                else f"[#f4f4f5]{esc(session.title)}[/#f4f4f5]"
            )
            prompt = (
                f"[bold #6366f1]{index}.[/bold #6366f1] "
                f"{marker} {title} "
                f"[dim]{session.turn_count} turns · {session.branch_count} branches[/dim]"
            )
            options.append(Option(prompt, id=session.id))
        return options

    def on_mount(self) -> None:
        if self._sessions:
            options = self.query_one("#session-modal-list", OptionList)
            highlighted = next(
                (idx for idx, session in enumerate(self._sessions) if session.id == self._active_session_id),
                0,
            )
            options.highlighted = highlighted
            options.focus()
        else:
            self.query_one("#session-modal-name", Input).focus()
        self._sync_delete_button()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _selected_session_id(self) -> str:
        if not self._sessions:
            return ""
        options = self.query_one("#session-modal-list", OptionList)
        selected = options.highlighted_option
        if selected is None or selected.id is None:
            selected = self._session_options()[0]
        return str(selected.id)

    def _clear_delete_confirmation(self) -> None:
        if not self._pending_delete_session_id:
            return
        self._pending_delete_session_id = ""
        self._sync_delete_button()

    def _sync_delete_button(self) -> None:
        if not self._sessions:
            return
        button = self.query_one("#session-delete", Button)
        confirming = bool(
            self._pending_delete_session_id and self._pending_delete_session_id == self._selected_session_id()
        )
        button.label = "2. Confirm Delete" if confirming else "2. Delete Selected"
        button.variant = "error" if confirming else "default"

    def action_open_selected(self) -> None:
        session_id = self._selected_session_id()
        if not session_id:
            return
        self.dismiss({"action": "open", "session_id": session_id})

    def action_delete_selected(self) -> None:
        session_id = self._selected_session_id()
        if not session_id:
            return
        if self._pending_delete_session_id != session_id:
            self._pending_delete_session_id = session_id
            self._sync_delete_button()
            return
        self.dismiss({"action": "delete", "session_id": session_id})

    def action_create_new(self) -> None:
        self._clear_delete_confirmation()
        name = self.query_one("#session-modal-name", Input).value.strip()
        self.dismiss({"action": "create", "title": name})

    @on(Button.Pressed, "#session-open")
    def _open_button(self) -> None:
        self._clear_delete_confirmation()
        self.action_open_selected()

    @on(Button.Pressed, "#session-delete")
    def _delete_button(self) -> None:
        self.action_delete_selected()

    @on(Button.Pressed, "#session-new")
    def _new_button(self) -> None:
        self.action_create_new()

    @on(Input.Submitted, "#session-modal-name")
    def _new_name_submitted(self, event: Input.Submitted) -> None:
        self._clear_delete_confirmation()
        self.dismiss({"action": "create", "title": event.value.strip()})

    @on(OptionList.OptionSelected, "#session-modal-list")
    def _session_selected(self, _event: OptionList.OptionSelected) -> None:
        self._clear_delete_confirmation()
        self._sync_delete_button()

    @on(OptionList.OptionHighlighted, "#session-modal-list")
    def _session_highlighted(self, _event: OptionList.OptionHighlighted) -> None:
        self._clear_delete_confirmation()
        self._sync_delete_button()


@dataclass(frozen=True)
class PickerItem:
    id: str
    prompt: str


class SelectionPickerModal(ModalScreen[Optional[Dict[str, str]]]):
    CSS = """
    SelectionPickerModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.76);
    }

    #picker-modal {
        width: 68;
        max-width: 92%;
        height: auto;
        background: #09090b;
        border: solid #52525b;
        padding: 0 1 1 2;
    }

    #picker-modal-kicker {
        color: #71717a;
        padding: 0 0 1 0;
    }

    #picker-modal-title {
        color: #e4e4e7;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #picker-modal-subtitle {
        color: #a1a1aa;
        padding: 0 0 1 0;
    }

    #picker-modal-hint {
        color: #71717a;
        padding: 0 0 1 0;
    }

    #picker-modal-list-label {
        color: #6366f1;
        padding: 0 0 1 0;
    }

    #picker-modal-list {
        width: 1fr;
        height: auto;
        max-height: 10;
        background: #000000;
        border: solid #52525b;
        margin: 0 0 1 0;
        padding: 0;
    }

    #picker-modal-list > .option-list--option-highlighted {
        color: #ffffff;
        background: #1a1730;
    }

    #picker-modal-empty {
        color: #a1a1aa;
        padding: 0 0 1 0;
    }

    #picker-modal-footer {
        width: 1fr;
        height: auto;
        align-horizontal: left;
        padding: 0;
    }

    #picker-confirm,
    #picker-cancel {
        width: 1fr;
        min-width: 0;
        border: none;
        content-align: center middle;
        margin-right: 1;
    }

    #picker-confirm {
        background: #1a1730;
        color: #c7d2fe;
    }

    #picker-cancel {
        background: #23252a;
        color: #d4d4d8;
        margin-right: 0;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", show=False),
        Binding("enter", "confirm", show=False),
        Binding("1", "confirm", show=False),
        Binding("2", "cancel", show=False),
    ]

    def __init__(
        self,
        *,
        title: str,
        subtitle: str,
        confirm_label: str,
        empty_text: str,
        items: list[PickerItem],
    ) -> None:
        super().__init__()
        self._title = title
        self._subtitle = subtitle
        self._confirm_label = confirm_label
        self._empty_text = empty_text
        self._items = items

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-modal"):
            yield Static("FILE SELECTOR", id="picker-modal-kicker")
            yield Static(self._title, id="picker-modal-title")
            yield Static(self._subtitle, id="picker-modal-subtitle")
            yield Static(f"Shortcuts: 1. {self._confirm_label} · 2. Cancel", id="picker-modal-hint")
            if self._items:
                yield Static("Available Files", id="picker-modal-list-label")
                yield OptionList(*[Option(item.prompt, id=item.id) for item in self._items], id="picker-modal-list")
            else:
                yield Static(self._empty_text, id="picker-modal-empty")
            with Horizontal(id="picker-modal-footer"):
                yield Button(f"1. {self._confirm_label}", id="picker-confirm", variant="primary", disabled=not self._items)
                yield Button("2. Cancel", id="picker-cancel")

    def on_mount(self) -> None:
        if self._items:
            options = self.query_one("#picker-modal-list", OptionList)
            options.highlighted = 0
            options.focus()
        else:
            self.query_one("#picker-cancel", Button).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_confirm(self) -> None:
        if not self._items:
            return
        options = self.query_one("#picker-modal-list", OptionList)
        selected = options.highlighted_option
        if selected is None or selected.id is None:
            selected = Option(self._items[0].prompt, id=self._items[0].id)
        self.dismiss({"id": str(selected.id)})

    @on(Button.Pressed, "#picker-confirm")
    def _confirm_button(self) -> None:
        self.action_confirm()

    @on(Button.Pressed, "#picker-cancel")
    def _cancel_button(self) -> None:
        self.action_cancel()

    @on(OptionList.OptionSelected, "#picker-modal-list")
    def _option_selected(self, _event: OptionList.OptionSelected) -> None:
        self.action_confirm()
