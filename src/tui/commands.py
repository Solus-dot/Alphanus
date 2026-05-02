from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandEntry:
    prompt: str
    insert_text: str
    description: str
    aliases: tuple[str, ...] = ()
    help_description: str | None = None


@dataclass(frozen=True)
class _CommandSection:
    title: str
    entries: tuple[CommandEntry, ...]


_COMMAND_SECTIONS: tuple[_CommandSection, ...] = (
    _CommandSection(
        "CONVERSATION",
        (
            CommandEntry("/help", "/help", "Show this help"),
            CommandEntry(
                "/shortcuts",
                "/shortcuts",
                "Show keyboard shortcuts",
                aliases=("/keymap", "/keys"),
            ),
            CommandEntry("/details", "/details", "Toggle tool execution details"),
            CommandEntry("/think", "/think", "Toggle thinking mode"),
            CommandEntry("/mode [plan|execute]", "/mode ", "Show or set collaboration mode"),
            CommandEntry("/clear", "/clear", "Clear tree and chat log"),
            CommandEntry("/sessions", "/sessions", "Open sessions"),
            CommandEntry("/rename <name>", "/rename ", "Rename the active session"),
            CommandEntry("/save [name]", "/save ", "Save the active session"),
            CommandEntry(
                "/file [path]",
                "/file ",
                "Attach a file or open the workspace picker",
                help_description="Attach a file to the next message or open the picker",
            ),
            CommandEntry("/detach [n|last|all]", "/detach ", "Remove pending attachments"),
            CommandEntry("/quit", "/quit", "Exit app", aliases=("/exit", "/q")),
        ),
    ),
    _CommandSection(
        "BRANCHING",
        (
            CommandEntry(
                "/branch [label]", "/branch ", "Arm the next user message as a branch", help_description="Arm next user message as a branch"
            ),
            CommandEntry("/unbranch", "/unbranch", "Return to the nearest branch fork"),
            CommandEntry("/branches", "/branches", "List branches from the current turn"),
            CommandEntry("/switch <n>", "/switch ", "Switch to a child branch"),
            CommandEntry("/tree", "/tree", "Render the full conversation tree"),
        ),
    ),
    _CommandSection(
        "SKILLS",
        (
            CommandEntry("/skills", "/skills", "List installed skills"),
            CommandEntry("/reload", "/reload", "Reload skills from disk"),
            CommandEntry("/doctor", "/doctor", "Run readiness diagnostics"),
            CommandEntry("/skill-on <id>", "/skill-on ", "Enable a skill"),
            CommandEntry("/skill-off <id>", "/skill-off ", "Disable a skill"),
            CommandEntry(
                "/skill-unload <id>",
                "/skill-unload ",
                "Unload a loaded skill from the current session",
                help_description="Unload a loaded skill",
            ),
            CommandEntry(
                "/skill-unload-all",
                "/skill-unload-all",
                "Unload all loaded skills from the current session",
                help_description="Unload all loaded skills",
            ),
            CommandEntry("/skill-reload", "/skill-reload", "Reload skills from disk"),
            CommandEntry("/skill-info <id>", "/skill-info ", "Show skill details"),
        ),
    ),
    _CommandSection(
        "UTILITIES",
        (
            CommandEntry("/memory-stats", "/memory-stats", "Show memory stats"),
            CommandEntry("/context", "/context", "Show inference engine context usage"),
            CommandEntry("/workspace-tree", "/workspace-tree", "Render workspace tree"),
            CommandEntry("/theme", "/theme", "Open theme picker"),
            CommandEntry("/config", "/config", "Edit the global config in a popup"),
            CommandEntry("/report [file]", "/report ", "Save a support bundle JSON"),
            CommandEntry("/code [n|last]", "/code ", "Open a copyable code block viewer"),
        ),
    ),
)


COMMAND_ENTRIES = [entry for section in _COMMAND_SECTIONS for entry in section.entries]


def _command_token_span(value: str, cursor_position: int | None = None) -> tuple[int, int] | None:
    if not value:
        return None
    cursor = len(value) if cursor_position is None else max(0, min(cursor_position, len(value)))
    start = cursor
    while start > 0 and not value[start - 1].isspace():
        start -= 1
    end = cursor
    while end < len(value) and not value[end].isspace():
        end += 1
    return start, end


def active_command_query(value: str, cursor_position: int | None = None) -> str:
    span = _command_token_span(value, cursor_position)
    if span is None:
        return ""
    start, end = span
    token = value[start:end].strip()
    if not token.startswith("/"):
        return ""
    if value[:start].strip():
        return ""
    return token


def active_command_span(value: str, cursor_position: int | None = None) -> tuple[int, int] | None:
    span = _command_token_span(value, cursor_position)
    if span is None:
        return None
    start, end = span
    if value[start:end].strip().startswith("/") and not value[:start].strip():
        return span
    return None


def popup_command_query(value: str, cursor_position: int | None = None) -> str:
    return active_command_query(value, cursor_position)


def command_entries_for_query(value: str) -> list[CommandEntry]:
    query = value.strip().lower()
    if not query.startswith("/"):
        return []
    needle = query[1:]
    if not needle:
        return COMMAND_ENTRIES

    def sort_key(entry: CommandEntry) -> tuple[int, int, str]:
        aliases = " ".join(entry.aliases)
        haystack = f"{entry.prompt} {aliases} {entry.description}".lower()
        starts = (
            0
            if (entry.prompt.lower().startswith(f"/{needle}") or any(alias.lower().startswith(f"/{needle}") for alias in entry.aliases))
            else 1
        )
        pos = haystack.find(needle)
        return (starts, pos if pos >= 0 else 9999, entry.prompt)

    matches = [
        entry
        for entry in COMMAND_ENTRIES
        if (
            needle in entry.prompt.lower() or any(needle in alias.lower() for alias in entry.aliases) or needle in entry.description.lower()
        )
    ]
    matches.sort(key=sort_key)
    return matches


def command_label(entry: CommandEntry) -> str:
    if not entry.aliases:
        return entry.prompt
    return f"{entry.prompt} {' '.join(entry.aliases)}"


HELP_SECTIONS = [
    (
        section.title,
        [(command_label(entry), entry.help_description or entry.description) for entry in section.entries],
    )
    for section in _COMMAND_SECTIONS
]


def exact_command_inputs() -> set[str]:
    exact = {entry.insert_text.strip().lower() for entry in COMMAND_ENTRIES}
    for entry in COMMAND_ENTRIES:
        exact.update(alias.strip().lower() for alias in entry.aliases)
    return exact
