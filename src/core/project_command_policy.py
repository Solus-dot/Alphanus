from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

READ_ONLY_SHELL_COMMANDS = {
    "ls",
    "pwd",
    "cat",
    "head",
    "tail",
    "grep",
    "rg",
    "find",
    "stat",
    "wc",
    "sort",
    "uniq",
    "cut",
}
READ_ONLY_GIT_SUBCOMMANDS = {"status", "diff", "show", "log", "rev-parse", "branch"}
MUTATING_SHELL_COMMANDS = {"touch", "mkdir", "mv", "cp", "rm", "chmod", "chown", "ln"}
MUTATING_GIT_SUBCOMMANDS = {"add", "rm", "mv", "restore", "checkout", "switch", "commit", "clean", "apply", "am", "stash"}


def shell_has_boundary(command: str) -> bool:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(command):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            elif quote == '"' and (char == "`" or (char == "$" and command[index + 1 : index + 2] == "(")):
                return True
            continue
        if char in "'\"":
            quote = char
        elif char == "`" or char == "\n" or char in ";&|<>" or (char == "$" and command[index + 1 : index + 2] == "("):
            return True
    return False


def unwrap_shell_command(argv: list[str], wrappers: Collection[str]) -> tuple[str, list[str]]:
    index = 0
    while index < len(argv):
        candidate = argv[index]
        name = Path(candidate).name
        if "=" in candidate and not candidate.startswith("-") and candidate.split("=", 1)[0]:
            index += 1
        elif name in wrappers:
            index += 1
            while index < len(argv) and argv[index].startswith("-"):
                index += 1
        else:
            return name, argv[index + 1 :]
    return "", []


class ProjectCommandPolicy:
    @staticmethod
    def git_subcommand(argv: list[str]) -> str:
        if len(argv) < 2:
            return ""
        if argv[1] == "-C" and len(argv) >= 4:
            return argv[3]
        return argv[1]

    @staticmethod
    def git_subcommand_index(argv: list[str]) -> int:
        if len(argv) < 2:
            return -1
        if argv[1] == "-C" and len(argv) >= 4:
            return 3
        return 1

    @staticmethod
    def classify_shell_command(argv: list[str]) -> str:
        if not argv:
            return "ambiguous"
        executable = argv[0]
        if executable in READ_ONLY_SHELL_COMMANDS:
            return "readonly"
        if executable in MUTATING_SHELL_COMMANDS:
            return "mutating"
        if executable == "sed":
            return "readonly" if "-i" not in argv else "mutating"
        if executable != "git":
            return "ambiguous"

        subcommand = ProjectCommandPolicy.git_subcommand(argv)
        if subcommand == "branch":
            if "--show-current" in argv:
                return "readonly"
            return "ambiguous"
        if subcommand in READ_ONLY_GIT_SUBCOMMANDS:
            return "readonly"
        if subcommand in MUTATING_GIT_SUBCOMMANDS:
            return "mutating"
        return "ambiguous"
