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

from core.sessions import ExportSummary, SessionSummary


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


class SessionPickerModal(ModalScreen[Optional[Dict[str, str]]]):
    CSS = """
    SessionPickerModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.76);
    }

    #session-modal {
        width: 72;
        max-width: 92%;
        height: auto;
        background: #121214;
        border: panel #27272a;
        padding: 1 2;
    }

    #session-modal-title {
        color: #e4e4e7;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #session-modal-subtitle {
        color: #a1a1aa;
        padding: 0 0 1 0;
    }

    #session-modal-list-label,
    #session-modal-new-label {
        color: #c4b5fd;
        padding: 1 0 0 0;
    }

    #session-modal-list {
        width: 1fr;
        height: auto;
        max-height: 10;
        background: #18181b;
        border: solid #27272a;
        margin: 0 0 1 0;
    }

    #session-modal-list > .option-list--option-highlighted {
        color: #ffffff;
        background: #312e81;
    }

    #session-modal-name {
        width: 1fr;
        margin: 0 0 1 0;
        background: #18181b;
        color: #e4e4e7;
        border: round #3f3f46;
    }

    #session-modal-footer {
        height: auto;
        align-horizontal: right;
        padding: 1 0 0 0;
    }

    #session-load {
        background: #6366f1;
        color: #ffffff;
        border: none;
        margin-right: 1;
    }

    #session-new {
        background: #10b981;
        color: #ffffff;
        border: none;
        margin-right: 1;
    }

    #session-continue {
        background: #18181b;
        color: #e4e4e7;
        border: none;
    }
    """

    BINDINGS = [
        Binding("escape", "continue_current", show=False),
        Binding("enter", "load_selected", show=False),
    ]

    def __init__(self, sessions: list[SessionSummary], active_session_id: str, active_session_title: str) -> None:
        super().__init__()
        self._sessions = sessions
        self._active_session_id = active_session_id
        self._active_session_title = active_session_title

    def compose(self) -> ComposeResult:
        with Vertical(id="session-modal"):
            yield Static("Choose Session", id="session-modal-title")
            yield Static(
                "Load an existing session or start a new one before entering the chat.",
                id="session-modal-subtitle",
            )
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
                yield Button("Load Selected", id="session-load", variant="primary", disabled=not self._sessions)
                yield Button("New Session", id="session-new", variant="success")
                yield Button(f"Continue {self._active_session_title}", id="session-continue")

    def _session_options(self) -> list[Option]:
        options: list[Option] = []
        for index, session in enumerate(self._sessions, start=1):
            marker = (
                "[bold #8b5cf6]active[/bold #8b5cf6]"
                if session.is_active
                else "[#71717a]saved[/#71717a]"
            )
            title = (
                f"[bold #c4b5fd]{esc(session.title)}[/bold #c4b5fd]"
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

    def action_continue_current(self) -> None:
        self.dismiss({"action": "continue"})

    def action_load_selected(self) -> None:
        if not self._sessions:
            return
        options = self.query_one("#session-modal-list", OptionList)
        selected = options.highlighted_option
        if selected is None or selected.id is None:
            selected = self._session_options()[0]
        self.dismiss({"action": "load", "selector": str(selected.id)})

    @on(Button.Pressed, "#session-load")
    def _load_button(self) -> None:
        self.action_load_selected()

    @on(Button.Pressed, "#session-new")
    def _new_button(self) -> None:
        name = self.query_one("#session-modal-name", Input).value.strip()
        self.dismiss({"action": "new", "title": name})

    @on(Input.Submitted, "#session-modal-name")
    def _new_name_submitted(self, event: Input.Submitted) -> None:
        self.dismiss({"action": "new", "title": event.value.strip()})

    @on(Button.Pressed, "#session-continue")
    def _continue_button(self) -> None:
        self.action_continue_current()

    @on(OptionList.OptionSelected, "#session-modal-list")
    def _session_selected(self, _event: OptionList.OptionSelected) -> None:
        self.action_load_selected()


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
        width: 72;
        max-width: 92%;
        height: auto;
        background: #121214;
        border: panel #27272a;
        padding: 1 2;
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

    #picker-modal-list {
        width: 1fr;
        height: auto;
        max-height: 12;
        background: #18181b;
        border: solid #27272a;
        margin: 0 0 1 0;
    }

    #picker-modal-list > .option-list--option-highlighted {
        color: #ffffff;
        background: #312e81;
    }

    #picker-modal-empty {
        color: #a1a1aa;
        padding: 0 0 1 0;
    }

    #picker-modal-footer {
        height: auto;
        align-horizontal: right;
        padding: 1 0 0 0;
    }

    #picker-confirm {
        background: #6366f1;
        color: #ffffff;
        border: none;
        margin-right: 1;
    }

    #picker-cancel {
        background: #18181b;
        color: #e4e4e7;
        border: none;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", show=False),
        Binding("enter", "confirm", show=False),
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
            yield Static(self._title, id="picker-modal-title")
            yield Static(self._subtitle, id="picker-modal-subtitle")
            if self._items:
                yield OptionList(*[Option(item.prompt, id=item.id) for item in self._items], id="picker-modal-list")
            else:
                yield Static(self._empty_text, id="picker-modal-empty")
            with Horizontal(id="picker-modal-footer"):
                yield Button(self._confirm_label, id="picker-confirm", variant="primary", disabled=not self._items)
                yield Button("Cancel", id="picker-cancel")

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


def session_picker_items(sessions: list[SessionSummary]) -> list[PickerItem]:
    items: list[PickerItem] = []
    for index, session in enumerate(sessions, start=1):
        marker = (
            "[bold #8b5cf6]active[/bold #8b5cf6]"
            if session.is_active
            else "[#71717a]saved[/#71717a]"
        )
        title = (
            f"[bold #c4b5fd]{esc(session.title)}[/bold #c4b5fd]"
            if session.is_active
            else f"[#f4f4f5]{esc(session.title)}[/#f4f4f5]"
        )
        prompt = (
            f"[bold #6366f1]{index}.[/bold #6366f1] "
            f"{marker} {title} "
            f"[dim]{session.turn_count} turns · {session.branch_count} branches[/dim]"
        )
        items.append(PickerItem(id=session.id, prompt=prompt))
    return items


def export_picker_items(exports: list[ExportSummary]) -> list[PickerItem]:
    items: list[PickerItem] = []
    for index, exported in enumerate(exports, start=1):
        prompt = (
            f"[bold #6366f1]{index}.[/bold #6366f1] "
            f"[#f4f4f5]{esc(exported.title)}[/#f4f4f5] "
            f"[dim]{exported.turn_count} turns · {exported.branch_count} branches · {esc(exported.filename)}[/dim]"
        )
        items.append(PickerItem(id=exported.filename, prompt=prompt))
    return items
