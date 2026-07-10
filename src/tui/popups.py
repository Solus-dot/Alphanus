import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.markup import escape as esc
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from core.sessions import SessionSearchResult, SessionSummary
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
        language: str | None,
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


class ConfigEditorModal(ModalScreen[dict[str, Any] | None]):
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
            yield Static("Secrets are omitted here. Use environment variables for model and embedding API keys.", id="config-modal-note")
            yield TextArea(
                self._initial_text,
                language="toml",
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
            parsed = tomllib.loads(raw)
        except tomllib.TOMLDecodeError as exc:
            self.query_one("#config-modal-error", Static).update(f"Invalid TOML: {exc}")
            return

        if not isinstance(parsed, dict):
            self.query_one("#config-modal-error", Static).update("Global config must be a TOML document.")
            return

        self.dismiss({"config": parsed, "text": raw})


def _health_status(ok: bool, *, warn: bool = False) -> str:
    if ok and not warn:
        return "OK"
    return "WARN" if warn else "ERROR"


def _skill_needs_attention(skill: dict[str, Any]) -> bool:
    availability = str(skill.get("availability_code", "ready") or "ready")
    status = str(skill.get("status", "unknown") or "unknown")
    if availability != "ready":
        return True
    if status not in {"on", "off", "ready", "loaded", "available"}:
        return True
    if not bool(skill.get("available", True)):
        return True
    if skill.get("validation_errors"):
        return True
    return False


def health_report_markup(report: dict[str, Any]) -> str:
    agent = report.get("agent", {}) if isinstance(report.get("agent"), dict) else {}
    project = report.get("project", {}) if isinstance(report.get("project"), dict) else {}
    sandbox = report.get("sandbox", {}) if isinstance(report.get("sandbox"), dict) else {}
    search = report.get("search", {}) if isinstance(report.get("search"), dict) else {}
    retrieval = report.get("retrieval", {}) if isinstance(report.get("retrieval"), dict) else {}
    memory = report.get("memory", {}) if isinstance(report.get("memory"), dict) else {}
    skills = report.get("skills", []) if isinstance(report.get("skills"), list) else []

    def line(label: str, status: str, detail: str = "") -> str:
        color = "#10b981" if status == "OK" else ("#f59e0b" if status == "WARN" else "#f87171")
        suffix = f" [dim]{esc(detail)}[/dim]" if detail else ""
        return f"[{color}]{status:<5}[/{color}] [bold]{esc(label)}[/bold]{suffix}"

    skill_errors = []
    for skill in skills:
        if not isinstance(skill, dict):
            continue
        if _skill_needs_attention(skill):
            skill_errors.append(skill)
    endpoint_policy_error = str(agent.get("endpoint_policy_error") or "")
    backend_integrity = str(agent.get("backend_model_integrity", "unknown"))
    search_reason = str(search.get("reason") or "")
    retrieval_reason = str(retrieval.get("reason") or "")
    lines = [
        "[bold]Project Health[/bold]",
        "",
        line("model endpoint", _health_status(bool(agent.get("ready")), warn=bool(endpoint_policy_error)), str(agent.get("endpoint_mode", ""))),
        line("endpoint policy", _health_status(not bool(endpoint_policy_error)), endpoint_policy_error or "valid"),
        line(
            "backend profile",
            _health_status(backend_integrity != "violation", warn=backend_integrity in {"unknown", ""}),
            f"{agent.get('backend_profile_selected', 'unknown')} · integrity {backend_integrity}",
        ),
        line(
            "project",
            _health_status(bool(project.get("exists")) and bool(project.get("writable"))),
            str(project.get("path", "")),
        ),
        line("search", _health_status(bool(search.get("ready"))), search_reason or str(search.get("provider", ""))),
        line("retrieval", _health_status(bool(retrieval.get("ready"))), retrieval_reason or "ready"),
        line("memory", _health_status(True), f"{memory.get('backend', 'lexical')} · {memory.get('count', 0)} items"),
        line(
            "skills",
            _health_status(not skill_errors, warn=bool(skill_errors)),
            f"{len(skills)} installed" + (f" · {len(skill_errors)} need attention" if skill_errors else ""),
        ),
        line(
            "runtime",
            _health_status(True),
            f"{agent.get('permission_mode', 'project-write')} · approvals {agent.get('approvals', 'on-boundary')}",
        ),
        line(
            "sandbox",
            _health_status(bool(sandbox.get("ok"))),
            str(sandbox.get("message") or sandbox.get("backend") or ""),
        ),
    ]
    return "\n".join(lines)


class HealthModal(ModalScreen[None]):
    CSS = """
    HealthModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.76);
    }

    #health-modal {
        width: 76;
        max-width: 94%;
        height: auto;
        background: $background;
        border: solid $app-border;
        padding: 0 1;
    }

    #health-content {
        width: 1fr;
        color: $foreground;
        padding: 0;
    }

    #health-hint {
        color: $app-subtle;
        padding: 0;
    }

    #health-close {
        width: 1fr;
        border: none;
        content-align: left middle;
        height: 1;
        padding: 0 1 0 1;
        margin: 0;
        background: $surface;
        color: $foreground;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", show=False),
        Binding("enter", "cancel", show=False),
        Binding("return", "cancel", show=False),
    ]

    def __init__(self, report: dict[str, Any]) -> None:
        super().__init__()
        self._report = report

    def compose(self) -> ComposeResult:
        with Vertical(id="health-modal"):
            yield Static(Text.from_markup(health_report_markup(self._report)), id="health-content")
            yield Static("Use /doctor for the verbose diagnostic report.", id="health-hint")
            yield Button("Close [Enter]", id="health-close", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#health-close", Button).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#health-close")
    def _close_button(self) -> None:
        self.action_cancel()


class SessionManagerModal(ModalScreen[dict[str, str] | None]):
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

    #session-search-input {
        width: 1fr;
        margin: 0;
        background: $panel;
        color: $foreground;
        border: round $app-border;
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

    def __init__(
        self,
        sessions: list[SessionSummary],
        active_session_id: str,
        *,
        search_sessions: Callable[[str], list[SessionSearchResult]] | None = None,
    ) -> None:
        super().__init__()
        self._sessions = sessions
        self._active_session_id = active_session_id
        self._search_sessions = search_sessions
        self._search_results: list[SessionSearchResult] = []
        self._pending_delete_session_id = ""

    def compose(self) -> ComposeResult:
        with Vertical(id="session-modal"):
            yield Static("SESSIONS", id="session-modal-kicker")
            yield Static("Open, create, or delete sessions from one place.", id="session-modal-subtitle")
            yield Static(
                Text("Shortcuts: [Enter] opens selected · [D] deletes selected · [N] creates new · [Esc] closes"),
                id="session-modal-hint",
            )
            yield Input(placeholder="Search saved sessions", id="session-search-input")
            yield Static("Saved Sessions", id="session-modal-list-label")
            yield OptionList(id="session-modal-list")
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
            marker = f"[bold {accent}]active[/bold {accent}]" if session.is_active else f"[{subtle}]saved[/{subtle}]"
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

    def _search_options(self) -> list[Option]:
        theme_color = getattr(self.app, "_theme_color", None)
        accent = str(theme_color("accent", DEFAULT_ACCENT_COLOR)) if callable(theme_color) else DEFAULT_ACCENT_COLOR
        subtle = str(theme_color("subtle", DEFAULT_SUBTLE_COLOR)) if callable(theme_color) else DEFAULT_SUBTLE_COLOR
        text_color = str(theme_color("text", DEFAULT_TEXT_COLOR)) if callable(theme_color) else DEFAULT_TEXT_COLOR
        options: list[Option] = []
        for index, result in enumerate(self._search_results, start=1):
            marker = f"[bold {accent}]active[/bold {accent}]" if result.is_active else f"[{subtle}]match[/{subtle}]"
            title = f"[{text_color}]{esc(result.title)}[/{text_color}]"
            prompt = (
                f"[bold {accent}]{index}.[/bold {accent}] {marker} {title} "
                f"[dim]{esc(result.kind)} · {result.turn_count} turns · {result.branch_count} branches[/dim]\n"
                f"   [{subtle}]{esc(result.preview)}[/{subtle}]"
            )
            options.append(Option(prompt, id=f"search:{result.id}"))
        return options

    def _query(self) -> str:
        try:
            return self.query_one("#session-search-input", Input).value.strip()
        except Exception:
            return ""

    def _in_search_mode(self) -> bool:
        return bool(self._query())

    def _refresh_session_options(self, query: str = "") -> None:
        if query and self._search_sessions is not None:
            self._search_results = self._search_sessions(query)
            options = self._search_options()
            label = f"Search Results ({len(options)})"
        else:
            self._search_results = []
            options = self._session_options()
            label = "Saved Sessions"
        option_list = self.query_one("#session-modal-list", OptionList)
        option_list.clear_options()
        if options:
            option_list.add_options(options)
            option_list.highlighted = 0
        self.query_one("#session-modal-list-label", Static).update(label)
        has_items = bool(options)
        self.query_one("#session-open", Button).disabled = not has_items
        self.query_one("#session-delete", Button).disabled = not self._sessions or bool(query)
        self._sync_delete_button()

    def on_mount(self) -> None:
        self._refresh_session_options("")
        if self._sessions:
            options = self.query_one("#session-modal-list", OptionList)
            highlighted = next(
                (idx for idx, session in enumerate(self._sessions) if session.id == self._active_session_id),
                0,
            )
            options.highlighted = highlighted
            self.query_one("#session-search-input", Input).focus()
        else:
            self.query_one("#session-search-input", Input).focus()
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

    def _selected_search_result(self) -> SessionSearchResult | None:
        if not self._search_results:
            return None
        try:
            options = self.query_one("#session-modal-list", OptionList)
        except Exception:
            return None
        selected = options.highlighted_option
        selected_id = str(selected.id) if selected is not None and selected.id is not None else ""
        if selected_id.startswith("search:"):
            result_id = selected_id.removeprefix("search:")
            return next((result for result in self._search_results if result.id == result_id), None)
        return None

    def _clear_delete_confirmation(self) -> None:
        if not self._pending_delete_session_id:
            return
        self._pending_delete_session_id = ""
        self._sync_delete_button()

    def _sync_delete_button(self) -> None:
        if not self._sessions:
            return
        button = self.query_one("#session-delete", Button)
        if self._in_search_mode():
            button.label = Text("Delete Selected Session [D]")
            button.variant = "default"
            button.disabled = True
            return
        confirming = bool(self._pending_delete_session_id and self._pending_delete_session_id == self._selected_session_id())
        button.label = Text("Confirm Delete [D]") if confirming else Text("Delete Selected Session [D]")
        button.variant = "error" if confirming else "default"
        button.disabled = False

    def action_open_selected(self) -> None:
        result = self._selected_search_result()
        if result is not None:
            self.dismiss({"action": "open", "session_id": result.session_id, "turn_id": result.turn_id})
            return
        if self._in_search_mode():
            return
        session_id = self._selected_session_id()
        if not session_id:
            return
        self.dismiss({"action": "open", "session_id": session_id})

    def action_delete_selected(self) -> None:
        if self._in_search_mode():
            return
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

    @on(Input.Changed, "#session-search-input")
    def _session_query_changed(self, event: Input.Changed) -> None:
        self._clear_delete_confirmation()
        self._refresh_session_options(event.value.strip())

    @on(Input.Submitted, "#session-search-input")
    def _session_query_submitted(self, _event: Input.Submitted) -> None:
        self.action_open_selected()


class SessionNameModal(ModalScreen[dict[str, str] | None]):
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


class SelectionPickerModal(ModalScreen[dict[str, str] | None]):
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


class CommandPaletteModal(ModalScreen[dict[str, str] | None]):
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

    def _score_item(self, item: CommandPaletteItem, query: str) -> tuple[int, int, int, str] | None:
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
