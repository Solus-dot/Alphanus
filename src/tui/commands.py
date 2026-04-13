from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class CommandEntry:
    prompt: str
    insert_text: str
    description: str
    aliases: Tuple[str, ...] = ()


HELP_SECTIONS = [
    (
        "CONVERSATION",
        [
            ("/help", "Show this help"),
            ("/keyboard-shortcuts /shortcuts", "Show keyboard shortcuts"),
            ("/details", "Toggle tool execution details"),
            ("/think", "Toggle thinking mode"),
            ("/clear", "Clear tree and chat log"),
            ("/sessions", "Open sessions"),
            ("/rename <name>", "Rename the active session"),
            ("/save [name]", "Save the active session"),
            ("/file [path]", "Attach a file to the next message or open the picker"),
            ("/detach [n|last|all]", "Remove pending attachments"),
            ("/quit /exit /q", "Exit app"),
        ],
    ),
    (
        "BRANCHING",
        [
            ("/branch [label]", "Arm next user message as a branch"),
            ("/unbranch", "Return to nearest branch fork parent"),
            ("/branches", "List child turns of current turn"),
            ("/switch <n>", "Switch to child branch by index"),
            ("/tree", "Render full conversation tree"),
        ],
    ),
    (
        "SKILLS",
        [
            ("/skills", "List installed skills"),
            ("/reload", "Reload skills from disk"),
            ("/doctor", "Run readiness diagnostics"),
            ("/skill-on <id>", "Enable skill"),
            ("/skill-off <id>", "Disable skill"),
            ("/skill-unload <id>", "Unload a loaded skill"),
            ("/skill-unload-all", "Unload all loaded skills"),
            ("/skill-reload", "Reload skills from disk"),
            ("/skill-info <id>", "Show skill details"),
        ],
    ),
    (
        "UTILITIES",
        [
            ("/memory-stats", "Show memory stats"),
            ("/context", "Show inference engine context usage"),
            ("/workspace-tree", "Render workspace tree"),
            ("/config", "Edit global config in a popup"),
            ("/report [file]", "Save a support bundle JSON"),
            ("/code [n|last]", "Open a copyable code block viewer"),
        ],
    ),
]


COMMAND_ENTRIES = [
    CommandEntry("/help", "/help", "Show this help"),
    CommandEntry(
        "/keyboard-shortcuts",
        "/keyboard-shortcuts",
        "Show keyboard shortcuts",
        aliases=("/shortcuts", "/keymap", "/keys"),
    ),
    CommandEntry("/details", "/details", "Toggle tool execution details"),
    CommandEntry("/think", "/think", "Toggle thinking mode"),
    CommandEntry("/clear", "/clear", "Clear tree and chat log"),
    CommandEntry("/sessions", "/sessions", "Open sessions"),
    CommandEntry("/rename <name>", "/rename ", "Rename the active session"),
    CommandEntry("/save [name]", "/save ", "Save the active session"),
    CommandEntry("/file [path]", "/file ", "Attach a file or open the workspace picker"),
    CommandEntry("/detach [n|last|all]", "/detach ", "Remove pending attachments"),
    CommandEntry("/branch [label]", "/branch ", "Arm the next user message as a branch"),
    CommandEntry("/unbranch", "/unbranch", "Return to the nearest branch fork"),
    CommandEntry("/branches", "/branches", "List branches from the current turn"),
    CommandEntry("/switch <n>", "/switch ", "Switch to a child branch"),
    CommandEntry("/tree", "/tree", "Render the full conversation tree"),
    CommandEntry("/skills", "/skills", "List installed skills"),
    CommandEntry("/reload", "/reload", "Reload skills from disk"),
    CommandEntry("/doctor", "/doctor", "Run readiness diagnostics"),
    CommandEntry("/skill-on <id>", "/skill-on ", "Enable a skill"),
    CommandEntry("/skill-off <id>", "/skill-off ", "Disable a skill"),
    CommandEntry("/skill-unload <id>", "/skill-unload ", "Unload a loaded skill from the current session"),
    CommandEntry("/skill-unload-all", "/skill-unload-all", "Unload all loaded skills from the current session"),
    CommandEntry("/skill-reload", "/skill-reload", "Reload skills from disk"),
    CommandEntry("/skill-info <id>", "/skill-info ", "Show skill details"),
    CommandEntry("/memory-stats", "/memory-stats", "Show memory stats"),
    CommandEntry("/context", "/context", "Show inference engine context usage"),
    CommandEntry("/workspace-tree", "/workspace-tree", "Render workspace tree"),
    CommandEntry("/config", "/config", "Edit the global config in a popup"),
    CommandEntry("/report [file]", "/report ", "Save a support bundle JSON"),
    CommandEntry("/code [n|last]", "/code ", "Open a copyable code block viewer"),
    CommandEntry("/quit", "/quit", "Exit app", aliases=("/exit", "/q")),
]


def active_command_query(value: str, cursor_position: Optional[int] = None) -> str:
    if not value:
        return ""
    cursor = len(value) if cursor_position is None else max(0, min(cursor_position, len(value)))
    start = cursor
    while start > 0 and not value[start - 1].isspace():
        start -= 1
    end = cursor
    while end < len(value) and not value[end].isspace():
        end += 1
    token = value[start:end].strip()
    if not token.startswith("/"):
        return ""
    if value[:start].strip():
        return ""
    return token


def active_command_span(value: str, cursor_position: Optional[int] = None) -> Optional[Tuple[int, int]]:
    query = active_command_query(value, cursor_position)
    if not query:
        return None
    cursor = len(value) if cursor_position is None else max(0, min(cursor_position, len(value)))
    start = cursor
    while start > 0 and not value[start - 1].isspace():
        start -= 1
    end = cursor
    while end < len(value) and not value[end].isspace():
        end += 1
    return (start, end)


def popup_command_query(value: str, cursor_position: Optional[int] = None) -> str:
    return active_command_query(value, cursor_position)


def command_entries_for_query(value: str) -> List[CommandEntry]:
    query = value.strip().lower()
    if not query.startswith("/"):
        return []
    entries = COMMAND_ENTRIES
    needle = query[1:]
    if not needle:
        return entries

    def sort_key(entry: CommandEntry) -> Tuple[int, int, str]:
        aliases = " ".join(entry.aliases)
        haystack = f"{entry.prompt} {aliases} {entry.description}".lower()
        starts = 0 if (
            entry.prompt.lower().startswith(f"/{needle}")
            or any(alias.lower().startswith(f"/{needle}") for alias in entry.aliases)
        ) else 1
        pos = haystack.find(needle)
        return (starts, pos if pos >= 0 else 9999, entry.prompt)

    matches = [
        entry
        for entry in entries
        if (
            needle in entry.prompt.lower()
            or any(needle in alias.lower() for alias in entry.aliases)
            or needle in entry.description.lower()
        )
    ]
    matches.sort(key=sort_key)
    return matches


def command_label(entry: CommandEntry) -> str:
    if not entry.aliases:
        return entry.prompt
    return f"{entry.prompt} {' '.join(entry.aliases)}"


def exact_command_inputs() -> set[str]:
    entries = COMMAND_ENTRIES
    exact = {entry.insert_text.strip().lower() for entry in entries}
    for entry in entries:
        exact.update(alias.strip().lower() for alias in entry.aliases)
    return exact
