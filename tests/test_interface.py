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


def test_live_tool_preview_streams_create_files_current_draft() -> None:
    preview = LiveToolPreviewManager()
    lines: list[str] = []
    partial_updates: list[tuple[list[str], str | None]] = []

    preview.update(
        "stream-1",
        "create_files",
        '{"files":[{"filepath":"site/index.html","content":"<h1>Hello</h1>\\n"},{"filepath":"site/script.js","content":"console.log(\\"hi\\")\\n"}]}',
        lines.append,
        lambda code, language: partial_updates.append((code, language)),
    )

    assert any("site/script.js" in line for line in lines)
    assert partial_updates
    assert partial_updates[-1][1] == "javascript"
    assert "console.log" in "\n".join(partial_updates[-1][0])


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


def test_take_pending_tool_detail_is_fifo_by_name() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tui._pending_tool_details = [
        ("web_search", "q=one"),
        ("workspace_tree", "max_depth=5"),
        ("web_search", "q=two"),
    ]

    assert tui._take_pending_tool_detail("web_search") == "q=one"
    assert tui._take_pending_tool_detail("workspace_tree") == "max_depth=5"
    assert tui._take_pending_tool_detail("web_search") == "q=two"
    assert tui._take_pending_tool_detail("web_search") == ""


def test_file_tool_success_lines_stay_hidden() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tui._show_tool_details = True
    tui._live_preview = SimpleNamespace(streamed_file_tools={"create_file", "edit_file", "create_files"})

    assert tui._show_tool_result_line("create_files", True) is False
    assert tui._show_tool_result_line("create_file", True) is False
    assert tui._show_tool_result_line("edit_file", True) is False
    assert tui._show_tool_result_line("workspace_tree", True) is True


def test_show_keymap_writes_expected_sections() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    lines: list[str] = []
    tui._write = lines.append
    tui._write_section_heading = lambda text: lines.append(f"SECTION:{text}")
    tui._write_command_row = lambda command, desc, *, col: lines.append(f"ROW:{command}:{desc}:{col}")

    tui.action_show_keymap()

    assert "SECTION:Keymap" in lines
    assert any("Tab / Shift+Tab" in line for line in lines)
    assert any("SECTION:Tree" == line for line in lines)
    assert any("Enter / o" in line for line in lines)
