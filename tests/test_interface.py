from __future__ import annotations

from types import SimpleNamespace

from tui.live_tool_preview import LiveToolPreviewManager
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


def test_tool_result_lines_hidden_when_details_off() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tui._show_tool_details = False
    tui._live_preview = SimpleNamespace(streamed_file_tools={"create_file", "edit_file"})

    assert tui._show_tool_result_line("create_file", True) is False
    assert tui._show_tool_result_line("web_search", True) is False


def test_historical_tool_details_default_off() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tui._show_tool_details = True
    tui._show_historical_tool_details = False
    tui._write = lambda *_args, **_kwargs: None
    tui._write_indented = lambda *_args, **_kwargs: None
    tui._write_code_block = lambda *_args, **_kwargs: None
    tui._live_preview = SimpleNamespace(streamed_file_tools={"create_file", "edit_file"}, compact_tool_args=lambda *_args, **_kwargs: "")

    turn = SimpleNamespace(
        skill_exchanges=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "create_file",
                            "arguments": "{\"filepath\":\"/tmp/x\",\"content\":\"hello\"}",
                        }
                    }
                ],
            }
        ]
    )

    # Should be a no-op even with live details enabled.
    assert tui._write_skill_exchanges(turn) is None


def test_live_tool_preview_shows_create_files_contents() -> None:
    preview = LiveToolPreviewManager()
    lines: list[str] = []
    code_blocks: list[tuple[list[str], str | None, int]] = []

    preview.write_static_preview(
        "create_files",
        {
            "files": [
                {"filepath": "site/index.html", "content": "<h1>Hello</h1>\n"},
                {"filepath": "site/script.js", "content": "console.log('hi')\n"},
            ]
        },
        lines.append,
        lambda markup, _indent=0: lines.append(markup),
        lambda code, language, indent=0: code_blocks.append((code, language, indent)),
    )

    assert any("site/index.html" in line for line in lines)
    assert any("site/script.js" in line for line in lines)
    assert any(language == "html" for _code, language, _indent in code_blocks)
    assert any(language == "javascript" for _code, language, _indent in code_blocks)


def test_live_tool_preview_shows_edit_file_diff() -> None:
    preview = LiveToolPreviewManager()
    lines: list[str] = []
    code_blocks: list[tuple[list[str], str | None, int]] = []

    preview.write_result_preview(
        "edit_file",
        {
            "ok": True,
            "data": {
                "filepath": "notes.txt",
                "diff": "--- notes.txt (before)\n+++ notes.txt (after)\n@@\n-beta\n+gamma",
            },
        },
        lines.append,
        lambda markup, _indent=0: lines.append(markup),
        lambda code, language, indent=0: code_blocks.append((code, language, indent)),
    )

    assert any("edit diff: notes.txt" in line for line in lines)
    assert code_blocks
    assert code_blocks[0][1] == "diff"
    assert "-beta" in "\n".join(code_blocks[0][0])
