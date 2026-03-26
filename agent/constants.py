"""Centralized structural constants for the agent package.

Contains only constants that represent the system's API contract — tool names,
skill identifiers, and security policy sets. These are structural facts about
the system, not semantic heuristics.

Intent classification (time-sensitivity, workspace action detection, etc.)
is handled entirely by the model-based classifier.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Core skill IDs — the built-in skills that receive special treatment
# ---------------------------------------------------------------------------
CORE_SKILL_IDS: frozenset[str] = frozenset({
    "workspace-ops",
    "search-ops",
    "shell-ops",
    "memory-rag",
    "utilities",
})

# ---------------------------------------------------------------------------
# Tool name sets used for policy enforcement
# ---------------------------------------------------------------------------
MUTATING_TOOL_NAMES: frozenset[str] = frozenset({
    "create_directory",
    "create_file",
    "create_files",
    "edit_file",
    "delete_path",
    "move_path",
    "run_skill_command",
    "run_skill_entrypoint",
    "run_skill_script",
})

LOCAL_WORKSPACE_BLOCKED_TOOLS: frozenset[str] = frozenset({
    "shell_command",
    "web_search",
    "fetch_url",
    "open_url",
    "play_youtube",
})

MATERIALIZER_TOOL_NAMES: frozenset[str] = frozenset({
    "create_file",
    "create_files",
    "edit_file",
})

FILE_CREATION_TOOLS: frozenset[str] = frozenset({
    "create_file",
    "edit_file",
    "create_directory",
    "read_file",
})

BATCH_FILE_TOOLS: frozenset[str] = frozenset({
    "create_files",
    "read_files",
})
