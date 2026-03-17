from __future__ import annotations

from tui.interface import AlphanusTUI


def test_command_entries_match_quit_aliases() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)

    quit_matches = [entry.prompt for entry in tui._command_entries_for_query("/qu")]
    exit_matches = [entry.prompt for entry in tui._command_entries_for_query("/exit")]

    assert "/quit" in quit_matches
    assert "/quit" in exit_matches


def test_config_editor_view_omits_secrets_and_unused_minilm() -> None:
    config = {
        "search": {"provider": "tavily", "tavily_api_key": "tvly-secret"},
        "memory": {"embedding_backend": "hash", "model_name": "all-MiniLM-L6-v2"},
    }

    cleaned = AlphanusTUI._config_for_editor(config)

    assert "tavily_api_key" not in cleaned["search"]
    assert "model_name" not in cleaned["memory"]
