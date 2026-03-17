from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

DEFAULT_SAVE = "llamachat_tree.json"


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
            ("/details", "Toggle tool execution details"),
            ("/think", "Toggle thinking mode"),
            ("/clear", "Clear tree and chat log"),
            ("/file <path>", "Attach image/text file to next message"),
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
            ("/skill on <id>", "Enable skill"),
            ("/skill off <id>", "Disable skill"),
            ("/skill reload", "Reload skills from disk"),
            ("/skill info <id>", "Show skill details"),
        ],
    ),
    (
        "UTILITIES",
        [
            ("/memory stats", "Show memory stats"),
            ("/workspace tree", "Render workspace tree"),
            ("/config", "Edit global config in a popup"),
            ("/report [file]", "Save a support bundle JSON"),
            ("/code [n|last]", "Open a copyable code block viewer"),
            ("/save [file]", f"Save tree JSON (default {DEFAULT_SAVE})"),
            ("/load [file]", f"Load tree JSON (default {DEFAULT_SAVE})"),
        ],
    ),
]


COMMAND_ENTRIES = [
    CommandEntry("/help", "/help", "Show this help"),
    CommandEntry("/details", "/details", "Toggle tool execution details"),
    CommandEntry("/think", "/think", "Toggle thinking mode"),
    CommandEntry("/clear", "/clear", "Clear tree and chat log"),
    CommandEntry("/file <path>", "/file ", "Attach a file to the next message"),
    CommandEntry("/branch [label]", "/branch ", "Arm the next user message as a branch"),
    CommandEntry("/unbranch", "/unbranch", "Return to the nearest branch fork"),
    CommandEntry("/branches", "/branches", "List branches from the current turn"),
    CommandEntry("/switch <n>", "/switch ", "Switch to a child branch"),
    CommandEntry("/tree", "/tree", "Render the full conversation tree"),
    CommandEntry("/skills", "/skills", "List installed skills"),
    CommandEntry("/reload", "/reload", "Reload skills from disk"),
    CommandEntry("/doctor", "/doctor", "Run readiness diagnostics"),
    CommandEntry("/skill on <id>", "/skill on ", "Enable a skill"),
    CommandEntry("/skill off <id>", "/skill off ", "Disable a skill"),
    CommandEntry("/skill reload", "/skill reload", "Reload skills from disk"),
    CommandEntry("/skill info <id>", "/skill info ", "Show skill details"),
    CommandEntry("/memory stats", "/memory stats", "Show memory stats"),
    CommandEntry("/workspace tree", "/workspace tree", "Render the workspace tree"),
    CommandEntry("/config", "/config", "Edit the global config in a popup"),
    CommandEntry("/report [file]", "/report ", "Save a support bundle JSON"),
    CommandEntry("/code [n|last]", "/code ", "Open a copyable code block viewer"),
    CommandEntry("/save [file]", "/save ", f"Save tree JSON (default {DEFAULT_SAVE})"),
    CommandEntry("/load [file]", "/load ", f"Load tree JSON (default {DEFAULT_SAVE})"),
    CommandEntry("/quit", "/quit", "Exit app", aliases=("/exit", "/q")),
]


def command_entries_for_query(value: str) -> List[CommandEntry]:
    query = value.strip().lower()
    if not query.startswith("/"):
        return []
    needle = query[1:]
    if not needle:
        return COMMAND_ENTRIES

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
        for entry in COMMAND_ENTRIES
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
    exact = {entry.insert_text.strip().lower() for entry in COMMAND_ENTRIES}
    for entry in COMMAND_ENTRIES:
        exact.update(alias.strip().lower() for alias in entry.aliases)
    return exact
