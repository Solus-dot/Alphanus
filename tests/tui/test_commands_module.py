from __future__ import annotations

from tui.commands import (
    COMMAND_ENTRIES,
    HELP_SECTIONS,
    active_command_query,
    active_command_span,
    command_entries_for_query,
    exact_command_inputs,
)


def test_active_command_query_and_span_share_token_boundaries() -> None:
    value = "/shortcuts extra text"

    assert active_command_query(value, 2) == "/shortcuts"
    assert active_command_span(value, 2) == (0, 10)


def test_help_sections_are_derived_from_command_metadata() -> None:
    sections = {title: rows for title, rows in HELP_SECTIONS}

    assert ("/quit /exit /q", "Exit app") in sections["CONVERSATION"]
    assert ("/file [path]", "Attach a file to the next message or open the picker") in sections["CONVERSATION"]
    assert ("/mode [plan|execute]", "Show or set collaboration mode") in sections["CONVERSATION"]
    assert ("/theme", "Open theme picker") in sections["UTILITIES"]

    file_entry = next(entry for entry in COMMAND_ENTRIES if entry.prompt == "/file [path]")
    assert file_entry.description == "Attach a file or open the workspace picker"
    assert file_entry.help_description == "Attach a file to the next message or open the picker"


def test_command_matching_prefers_prefix_matches_and_aliases() -> None:
    matches = command_entries_for_query("/key")

    assert matches
    assert matches[0].prompt == "/shortcuts"
    assert "/exit" in exact_command_inputs()
    assert "/q" in exact_command_inputs()
