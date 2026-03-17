from __future__ import annotations

from tui.commands import command_entries_for_query
from tui.interface import AlphanusTUI


def test_command_entries_match_quit_aliases() -> None:
    quit_matches = [entry.prompt for entry in command_entries_for_query("/qu")]
    exit_matches = [entry.prompt for entry in command_entries_for_query("/exit")]

    assert "/quit" in quit_matches
    assert "/quit" in exit_matches


def test_config_editor_view_omits_secrets_and_unused_minilm() -> None:
    config = {
        "agent": {"auth_header": "Authorization: Bearer secret"},
        "search": {"provider": "tavily", "tavily_api_key": "tvly-secret"},
        "memory": {"embedding_backend": "hash", "model_name": "BAAI/bge-small-en-v1.5"},
    }

    cleaned = AlphanusTUI._config_for_editor(config)

    assert "auth_header" not in cleaned["agent"]
    assert "tavily_api_key" not in cleaned["search"]
    assert "model_name" not in cleaned["memory"]
