import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional

from rich.markup import escape as esc
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from core.sessions import SessionSummary
from tui.themes import fallback_color

DEFAULT_ACCENT_COLOR = fallback_color("accent")
DEFAULT_SUBTLE_COLOR = fallback_color("subtle")
DEFAULT_TEXT_COLOR = fallback_color("text")


class CodeViewerModal(ModalScreen[None]):
    CSS = """
    CodeViewerModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.72);
    }

    #code-modal {
        width: 88%;
        height: 88%;
        background: $panel;
        border: panel $app-border;
        padding: 1 2;
    }

    #code-modal-title {
        color: $foreground;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #code-modal-editor {
        width: 1fr;
        height: 1fr;
        background: $panel;
        color: $foreground;
        border: solid $app-border;
    }

    #code-modal-footer {
        height: auto;
        align-horizontal: right;
        padding: 1 0 0 0;
    }

    #code-modal-hint {
        width: 1fr;
        color: $app-muted;
        padding-top: 1;
    }

    #code-copy {
        background: $accent;
        color: $foreground;
        border: none;
        margin-right: 1;
    }

    #code-close {
        background: $panel;
        color: $foreground;
        border: none;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", show=False),
        Binding("ctrl+shift+c", "copy_all", show=False),
    ]

    def __init__(
        self,
        code: str,
        language: Optional[str],
        title: str = "Code Block",
        *,
        syntax_theme: str = "dracula",
    ) -> None:
        super().__init__()
        self._code = code
        self._language = language or "text"
        self._title = title
        self._syntax_theme = syntax_theme

    def compose(self) -> ComposeResult:
        with Vertical(id="code-modal"):
            yield Static(self._title, id="code-modal-title")
            yield TextArea(
                self._code,
                language=self._language,
                theme=self._syntax_theme,
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
        background: $panel;
        border: panel $app-border;
        padding: 1 2;
    }

    #config-modal-title {
        color: $foreground;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #config-modal-subtitle {
        color: $app-muted;
        padding: 0 0 1 0;
    }

    #config-modal-note {
        color: $accent;
        padding: 0 0 1 0;
    }

    #config-modal-editor {
        width: 1fr;
        height: 1fr;
        background: $panel;
        color: $foreground;
        border: solid $app-border;
    }

    #config-modal-error {
        min-height: 1;
        color: $error;
        padding: 1 0 0 0;
    }

    #config-modal-footer {
        height: auto;
        align-horizontal: right;
        padding: 1 0 0 0;
    }

    #config-save {
        background: $success;
        color: $foreground;
        border: none;
        margin-right: 1;
    }

    #config-cancel {
        background: $panel;
        color: $foreground;
        border: none;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", show=False),
        Binding("ctrl+s", "save", show=False),
    ]

    def __init__(self, config_path: Path, initial_text: str, *, syntax_theme: str = "dracula") -> None:
        super().__init__()
        self._config_path = config_path
        self._initial_text = initial_text
        self._syntax_theme = syntax_theme

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
                theme=self._syntax_theme,
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
        background: $background;
        border: solid $app-border;
        padding: 0 1;
    }

    #session-modal-kicker {
        color: $app-subtle;
        padding: 0;
    }

    #session-modal-subtitle {
        color: $app-muted;
        padding: 0;
    }

    #session-modal-hint {
        color: $app-subtle;
        padding: 0;
    }

    #session-modal-list-label,
    #session-modal-action-label {
        color: $accent;
        padding: 0;
    }

    #session-modal-list {
        width: 1fr;
        height: auto;
        max-height: 10;
        background: $panel;
        border: solid $app-border;
        margin: 0;
        padding: 0;
    }

    #session-modal-list > .option-list--option-highlighted {
        color: $foreground;
        background: $app-selection-bg;
    }

    #session-modal-footer {
        width: 1fr;
        height: auto;
        layout: vertical;
        align-horizontal: left;
        padding: 0;
    }

    #session-open,
    #session-delete,
    #session-new {
        width: 1fr;
        border: none;
        content-align: left middle;
        height: 1;
        padding: 0 1 0 1;
        margin: 0;
    }

    #session-open {
        background: $surface;
        color: $foreground;
    }

    #session-delete {
        background: $surface;
        color: $foreground;
    }

    #session-new {
        background: $surface;
        color: $foreground;
    }

    """

    BINDINGS = [
        Binding("escape", "cancel", show=False),
        Binding("enter", "open_selected", show=False),
        Binding("return", "open_selected", show=False),
        Binding("d", "delete_selected", show=False),
        Binding("n", "create_new", show=False),
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
            yield Static(
                Text("Shortcuts: [Enter] opens selected · [D] deletes selected · [N] creates new · [Esc] closes"),
                id="session-modal-hint",
            )
            if self._sessions:
                yield Static("Saved Sessions", id="session-modal-list-label")
                yield OptionList(*self._session_options(), id="session-modal-list")
            yield Static("Actions", id="session-modal-action-label")
            with Vertical(id="session-modal-footer"):
                yield Button(Text("Open Selected Session [Enter]"), id="session-open", variant="primary", disabled=not self._sessions)
                yield Button(Text("Delete Selected Session [D]"), id="session-delete", variant="default", disabled=not self._sessions)
                yield Button(Text("New Session [N]"), id="session-new", variant="default")

    def _session_options(self) -> list[Option]:
        theme_color = getattr(self.app, "_theme_color", None)
        accent = str(theme_color("accent", DEFAULT_ACCENT_COLOR)) if callable(theme_color) else DEFAULT_ACCENT_COLOR
        subtle = str(theme_color("subtle", DEFAULT_SUBTLE_COLOR)) if callable(theme_color) else DEFAULT_SUBTLE_COLOR
        text_color = str(theme_color("text", DEFAULT_TEXT_COLOR)) if callable(theme_color) else DEFAULT_TEXT_COLOR
        options: list[Option] = []
        for index, session in enumerate(self._sessions, start=1):
            marker = (
                f"[bold {accent}]active[/bold {accent}]"
                if session.is_active
                else f"[{subtle}]saved[/{subtle}]"
            )
            title = (
                f"[bold {accent}]{esc(session.title)}[/bold {accent}]"
                if session.is_active
                else f"[{text_color}]{esc(session.title)}[/{text_color}]"
            )
            prompt = (
                f"[bold {accent}]{index}.[/bold {accent}] "
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
            self.query_one("#session-new", Button).focus()
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
        button.label = Text("Confirm Delete [D]") if confirming else Text("Delete Selected Session [D]")
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
        self.dismiss({"action": "new"})

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

    @on(OptionList.OptionSelected, "#session-modal-list")
    def _session_selected(self, _event: OptionList.OptionSelected) -> None:
        self._clear_delete_confirmation()
        self.action_open_selected()

    @on(OptionList.OptionHighlighted, "#session-modal-list")
    def _session_highlighted(self, _event: OptionList.OptionHighlighted) -> None:
        self._clear_delete_confirmation()
        self._sync_delete_button()


class SessionNameModal(ModalScreen[Optional[Dict[str, str]]]):
    CSS = """
    SessionNameModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.76);
    }

    #session-name-modal {
        width: 58;
        max-width: 92%;
        height: auto;
        background: $background;
        border: solid $app-border;
        padding: 0 1;
    }

    #session-name-modal-kicker {
        color: $app-subtle;
        padding: 0;
    }

    #session-name-modal-title {
        color: $foreground;
        text-style: bold;
        padding: 0;
    }

    #session-name-modal-subtitle {
        color: $app-muted;
        padding: 0;
    }

    #session-name-modal-hint {
        color: $app-subtle;
        padding: 0;
    }

    #session-name-modal-input {
        width: 1fr;
        margin: 0;
        background: $panel;
        color: $foreground;
        border: round $app-border;
    }

    #session-name-modal-footer {
        width: 1fr;
        height: auto;
        layout: vertical;
        align-horizontal: left;
        padding: 0;
    }

    #session-name-create,
    #session-name-cancel {
        width: 1fr;
        border: none;
        content-align: left middle;
        height: 1;
        padding: 0 1 0 1;
        margin: 0;
    }

    #session-name-create {
        background: $surface;
        color: $foreground;
    }

    #session-name-cancel {
        background: $surface;
        color: $foreground;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", show=False),
        Binding("enter", "create", show=False),
        Binding("return", "create", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="session-name-modal"):
            yield Static("NEW SESSION", id="session-name-modal-kicker")
            yield Static("Name your session", id="session-name-modal-title")
            yield Static(
                "Leave blank to auto-generate the next session name.",
                id="session-name-modal-subtitle",
            )
            yield Static(Text("Shortcuts: [Enter] creates · [Esc] cancels"), id="session-name-modal-hint")
            yield Input(placeholder="Session name (optional)", value="", id="session-name-modal-input")
            with Vertical(id="session-name-modal-footer"):
                yield Button(Text("Create Session [Enter]"), id="session-name-create", variant="primary")
                yield Button(Text("Cancel [Esc]"), id="session-name-cancel", variant="default")

    def on_mount(self) -> None:
        self.query_one("#session-name-modal-input", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_create(self) -> None:
        title = self.query_one("#session-name-modal-input", Input).value.strip()
        self.dismiss({"action": "create", "title": title})

    @on(Button.Pressed, "#session-name-create")
    def _create_button(self) -> None:
        self.action_create()

    @on(Button.Pressed, "#session-name-cancel")
    def _cancel_button(self) -> None:
        self.action_cancel()

    @on(Input.Submitted, "#session-name-modal-input")
    def _name_submitted(self, _event: Input.Submitted) -> None:
        self.action_create()

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
        background: $background;
        border: solid $app-border;
        padding: 0 1;
    }

    #picker-modal-kicker {
        color: $app-subtle;
        padding: 0;
    }

    #picker-modal-title {
        color: $foreground;
        text-style: bold;
        padding: 0;
    }

    #picker-modal-subtitle {
        color: $app-muted;
        padding: 0;
    }

    #picker-modal-hint {
        color: $app-subtle;
        padding: 0;
    }

    #picker-modal-list-label {
        color: $accent;
        padding: 0;
    }

    #picker-modal-list {
        width: 1fr;
        height: auto;
        max-height: 10;
        background: $panel;
        border: solid $app-border;
        margin: 0;
        padding: 0;
    }

    #picker-modal-list > .option-list--option-highlighted {
        color: $foreground;
        background: $app-selection-bg;
    }

    #picker-modal-empty {
        color: $app-muted;
        padding: 0;
    }

    #picker-modal-footer {
        width: 1fr;
        height: auto;
        layout: vertical;
        align-horizontal: left;
        padding: 0;
    }

    #picker-confirm,
    #picker-cancel {
        width: 1fr;
        border: none;
        content-align: left middle;
        height: 1;
        padding: 0 1 0 1;
        margin: 0;
    }

    #picker-confirm {
        background: $surface;
        color: $foreground;
    }

    #picker-cancel {
        background: $surface;
        color: $foreground;
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
        kicker: str = "FILE SELECTOR",
        title: str,
        subtitle: str,
        list_label: str = "Available Files",
        confirm_label: str,
        empty_text: str,
        items: list[PickerItem],
    ) -> None:
        super().__init__()
        self._kicker = kicker
        self._title = title
        self._subtitle = subtitle
        self._list_label = list_label
        self._confirm_label = confirm_label
        self._empty_text = empty_text
        self._items = items

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-modal"):
            yield Static(self._kicker, id="picker-modal-kicker")
            yield Static(self._title, id="picker-modal-title")
            yield Static(self._subtitle, id="picker-modal-subtitle")
            yield Static(
                f"Shortcuts: 1. {self._confirm_label} · 2. Cancel · Enter selects highlighted item",
                id="picker-modal-hint",
            )
            if self._items:
                yield Static(self._list_label, id="picker-modal-list-label")
                yield OptionList(*[Option(item.prompt, id=item.id) for item in self._items], id="picker-modal-list")
            else:
                yield Static(self._empty_text, id="picker-modal-empty")
            with Vertical(id="picker-modal-footer"):
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


@dataclass(frozen=True)
class CommandPaletteItem:
    id: str
    prompt: str
    search_text: str
    rank: int = 0


class CommandPaletteModal(ModalScreen[Optional[Dict[str, str]]]):
    CSS = """
    CommandPaletteModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.78);
    }

    #command-palette-modal {
        width: 84;
        max-width: 94%;
        height: auto;
        background: $background;
        border: solid $app-border;
        padding: 0 1;
    }

    #command-palette-kicker {
        color: $app-subtle;
        padding: 0;
    }

    #command-palette-title {
        color: $foreground;
        text-style: bold;
        padding: 0;
    }

    #command-palette-subtitle {
        color: $app-muted;
        padding: 0;
    }

    #command-palette-query {
        width: 1fr;
        margin: 0;
        background: $panel;
        color: $foreground;
        border: round $app-border;
    }

    #command-palette-list {
        width: 1fr;
        height: auto;
        max-height: 12;
        background: $panel;
        border: solid $app-border;
        margin: 0;
        padding: 0;
    }

    #command-palette-list > .option-list--option-highlighted {
        color: $foreground;
        background: $app-selection-bg;
    }

    #command-palette-empty {
        color: $app-muted;
        padding: 0;
    }

    #command-palette-hint {
        color: $app-subtle;
        padding: 0;
    }

    #command-palette-footer {
        width: 1fr;
        height: auto;
        layout: vertical;
        align-horizontal: left;
        padding: 0;
    }

    #command-palette-open,
    #command-palette-cancel {
        width: 1fr;
        border: none;
        content-align: left middle;
        height: 1;
        padding: 0 1 0 1;
        margin: 0;
    }

    #command-palette-open {
        background: $surface;
        color: $foreground;
    }

    #command-palette-cancel {
        background: $surface;
        color: $foreground;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", show=False),
        Binding("enter", "confirm", show=False),
        Binding("return", "confirm", show=False),
        Binding("down", "move_down", show=False),
        Binding("up", "move_up", show=False),
    ]

    def __init__(self, *, items: list[CommandPaletteItem]) -> None:
        super().__init__()
        self._all_items = list(items)
        self._filtered: list[CommandPaletteItem] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="command-palette-modal"):
            yield Static("QUICK OPEN", id="command-palette-kicker")
            yield Static("Commands · Sessions · Files · Skills", id="command-palette-title")
            yield Static("Type to search. Enter opens highlighted item.", id="command-palette-subtitle")
            yield Input(placeholder="Search (e.g. doctor, session, py, skill)", id="command-palette-query")
            yield OptionList(id="command-palette-list")
            yield Static("", id="command-palette-empty")
            yield Static("Shortcuts: [Enter] open · [Up/Down] move · [Esc] close", id="command-palette-hint")
            with Vertical(id="command-palette-footer"):
                yield Button("Open [Enter]", id="command-palette-open", variant="primary")
                yield Button("Cancel [Esc]", id="command-palette-cancel", variant="default")

    def on_mount(self) -> None:
        self._refresh_options("")
        self.query_one("#command-palette-query", Input).focus()

    def _score_item(self, item: CommandPaletteItem, query: str) -> Optional[tuple[int, int, int, str]]:
        q = query.strip().lower()
        search = item.search_text.lower()
        prompt = item.prompt.lower()
        if not q:
            return (item.rank, 9, len(prompt), prompt)
        tokens = [part for part in q.split() if part]
        if any(token not in search for token in tokens):
            return None
        starts = 0 if prompt.startswith(q) else 1
        contains = 0 if q in prompt else 1
        return (item.rank, starts + contains, len(prompt), prompt)

    def _refresh_options(self, query: str) -> None:
        ranked: list[tuple[tuple[int, int, int, str], CommandPaletteItem]] = []
        for item in self._all_items:
            score = self._score_item(item, query)
            if score is None:
                continue
            ranked.append((score, item))
        ranked.sort(key=lambda pair: pair[0])
        self._filtered = [item for _score, item in ranked[:120]]

        options = self.query_one("#command-palette-list", OptionList)
        options.clear_options()
        if self._filtered:
            options.add_options([Option(item.prompt, id=item.id) for item in self._filtered])
            options.highlighted = 0
        empty = self.query_one("#command-palette-empty", Static)
        empty.update("" if self._filtered else "No matches")
        self.query_one("#command-palette-open", Button).disabled = not self._filtered

    def _shift_selection(self, delta: int) -> None:
        if not self._filtered:
            return
        options = self.query_one("#command-palette-list", OptionList)
        current = 0 if options.highlighted is None else int(options.highlighted)
        options.highlighted = (current + delta) % len(self._filtered)
        options.scroll_to_highlight(top=False)

    def action_move_down(self) -> None:
        self._shift_selection(1)

    def action_move_up(self) -> None:
        self._shift_selection(-1)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_confirm(self) -> None:
        if not self._filtered:
            return
        options = self.query_one("#command-palette-list", OptionList)
        selected = options.highlighted_option
        if selected is None or selected.id is None:
            selected_id = self._filtered[0].id
        else:
            selected_id = str(selected.id)
        self.dismiss({"id": selected_id})

    @on(Input.Changed, "#command-palette-query")
    def _query_changed(self, event: Input.Changed) -> None:
        self._refresh_options(event.value)

    @on(Input.Submitted, "#command-palette-query")
    def _query_submitted(self, _event: Input.Submitted) -> None:
        self.action_confirm()

    @on(Button.Pressed, "#command-palette-open")
    def _open_button(self) -> None:
        self.action_confirm()

    @on(Button.Pressed, "#command-palette-cancel")
    def _cancel_button(self) -> None:
        self.action_cancel()

    @on(OptionList.OptionSelected, "#command-palette-list")
    def _option_selected(self, _event: OptionList.OptionSelected) -> None:
        self.action_confirm()
