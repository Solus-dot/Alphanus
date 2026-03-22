from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core.conv_tree import ConvTree
from core.sessions import ChatSession
from core.workspace import WorkspaceManager
from tui.live_tool_preview import LiveToolPreviewManager
from tui.commands import command_entries_for_query
from tui.interface import AlphanusTUI, ChatInput
from tui.popups import SessionPickerModal


def test_command_entries_match_quit_aliases() -> None:
    quit_matches = [entry.prompt for entry in command_entries_for_query("/qu")]
    exit_matches = [entry.prompt for entry in command_entries_for_query("/exit")]

    assert "/quit" in quit_matches
    assert "/quit" in exit_matches


def test_command_entries_match_keyboard_shortcuts_aliases() -> None:
    shortcut_matches = [entry.prompt for entry in command_entries_for_query("/short")]
    keymap_matches = [entry.prompt for entry in command_entries_for_query("/keym")]

    assert "/keyboard-shortcuts" in shortcut_matches
    assert "/keyboard-shortcuts" in keymap_matches


def test_command_entries_match_context_command() -> None:
    context_matches = [entry.prompt for entry in command_entries_for_query("/cont")]

    assert "/context" in context_matches


def test_chat_input_binds_new_shortcuts_locally() -> None:
    bindings = {binding.key: binding.action for binding in ChatInput.BINDINGS}

    assert bindings["ctrl+g"] == "focus_input"
    assert bindings["ctrl+p"] == "open_command_palette"
    assert bindings["f1"] == "show_keymap"
    assert bindings["f2"] == "toggle_details"
    assert bindings["f3"] == "toggle_thinking"


def test_config_editor_view_omits_secrets_and_unused_minilm() -> None:
    config = {
        "agent": {"auth_header": "Authorization: Bearer secret", "context_budget_max_tokens": 2048},
        "search": {"provider": "tavily", "tavily_api_key": "tvly-secret"},
        "memory": {"embedding_backend": "hash", "model_name": "BAAI/bge-small-en-v1.5"},
        "context": {"context_limit": 8192, "safety_margin": 500, "keep_last_n": 10},
    }

    cleaned = AlphanusTUI._config_for_editor(config)

    assert "auth_header" not in cleaned["agent"]
    assert "context_budget_max_tokens" not in cleaned["agent"]
    assert "tavily_api_key" not in cleaned["search"]
    assert "model_name" not in cleaned["memory"]
    assert "context_limit" not in cleaned["context"]
    assert "safety_margin" not in cleaned["context"]
    assert cleaned["context"]["keep_last_n"] == 10


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
    code_blocks: list[tuple[list[str], str | None, int]] = []

    preview.update(
        "stream-1",
        "create_files",
        '{"files":[{"filepath":"site/index.html","content":"<h1>Hello</h1>\\n"},{"filepath":"site/script.js","content":"console.log(\\"hi\\")\\n"}]}',
        lines.append,
        lambda code, language: partial_updates.append((code, language)),
        lambda _text, _indent=0: None,
        lambda code, language, indent=0: code_blocks.append((code, language, indent)),
        lambda: None,
    )

    assert any("site/index.html" in line for line in lines)
    assert any("site/script.js" in line for line in lines)
    assert any(language == "html" for _code, language, _indent in code_blocks)
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


def test_live_tool_preview_skips_static_edit_file_request_preview() -> None:
    preview = LiveToolPreviewManager()
    lines: list[str] = []
    code_blocks: list[tuple[list[str], str | None, int]] = []

    preview.write_static_preview(
        "edit_file",
        {
            "filepath": "notes.txt",
            "old_string": "beta",
            "new_string": "gamma",
        },
        lines.append,
        lambda markup, _indent=0: lines.append(markup),
        lambda code, language, indent=0: code_blocks.append((code, language, indent)),
    )

    assert lines == []
    assert code_blocks == []


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


def test_flush_reasoning_buffer_skips_whitespace_only_panel() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    writes: list[tuple[object, int]] = []
    partial_updates: list[object] = []
    tui._buf_r = "\n   \n"
    tui._partial = lambda: SimpleNamespace(update=partial_updates.append)
    tui._write_renderable = lambda renderable, indent=2: writes.append((renderable, indent))
    tui._reasoning_panel_renderable = lambda text: text
    tui._is_tool_trace_line = lambda _line: False

    tui._flush_reasoning_buffer()

    assert writes == []
    assert partial_updates == [""]
    assert tui._buf_r == ""


def test_file_tool_success_lines_use_standard_tool_blocks() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tui._show_tool_details = True
    tui._live_preview = SimpleNamespace(streamed_file_tools={"create_file", "edit_file", "create_files"})

    assert tui._show_tool_result_line("create_files", True) is True
    assert tui._show_tool_result_line("create_file", True) is True
    assert tui._show_tool_result_line("edit_file", True) is True
    assert tui._show_tool_result_line("workspace_tree", True) is True


def test_tool_call_create_files_writes_all_file_previews_without_deltas() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    writes: list[str] = []
    code_blocks: list[tuple[list[str], str | None, int]] = []
    partial_updates: list[object] = []

    tui._show_tool_details = True
    tui._pending_tool_details = []
    tui._live_preview = SimpleNamespace(
        compact_tool_args=lambda *_args, **_kwargs: "3 files",
        rendered_filepaths=lambda _stream_id: set(),
        mark_rendered_filepaths=lambda _stream_id, _paths: None,
        close=lambda *_args, **_kwargs: False,
        write_static_preview=lambda *_args, **_kwargs: None,
        _guess_language=LiveToolPreviewManager._guess_language,
    )
    tui._write = writes.append
    tui._write_indented = lambda markup, indent=0: writes.append(markup)
    tui._write_code_block = lambda code, language, indent=0: code_blocks.append((list(code), language, indent))
    tui._clear_partial_preview = lambda: None
    tui._close_reasoning_section = lambda: False
    tui._update_tool_call_partial = lambda name, detail="", indent=2: partial_updates.append((name, detail, indent))
    tui._maybe_scroll_end = lambda *args, **kwargs: None
    tui._partial = lambda: SimpleNamespace(update=lambda *_args, **_kwargs: None, display=False)
    tui._last_scroll = 0.0
    tui._scroll_interval = 999.0

    tui._on_agent_event(
        {
            "type": "tool_call",
            "stream_id": "s1",
            "name": "create_files",
            "arguments": {
                "files": [
                    {"filepath": "site/index.html", "content": "<h1>Hello</h1>\n"},
                    {"filepath": "site/styles.css", "content": "body {}\n"},
                    {"filepath": "site/script.js", "content": "console.log(1)\n"},
                ]
            },
        }
    )

    assert any("site/index.html" in line for line in writes)
    assert any("site/styles.css" in line for line in writes)
    assert any("site/script.js" in line for line in writes)
    assert any(language == "html" for _code, language, _indent in code_blocks)
    assert any(language == "css" for _code, language, _indent in code_blocks)
    assert any(language == "javascript" for _code, language, _indent in code_blocks)

def test_update_context_usage_ignores_total_tokens_without_prompt_tokens() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tui._last_model_context_tokens = 321
    updates: list[str] = []
    tui._update_topbar = lambda: updates.append("topbar")

    tui._update_context_usage_from_payload({"total_tokens": 999})

    assert tui._last_model_context_tokens == 321
    assert updates == ["topbar"]


def test_cmd_context_writes_percent_and_tokens() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    lines: list[str] = []
    tui._write = lines.append
    tui._write_section_heading = lambda text: lines.append(f"SECTION:{text}")
    tui._write_detail_line = lambda label, value, value_markup=False: lines.append(f"DETAIL:{label}:{value}:{value_markup}")
    tui._context_tokens = lambda: 1612
    tui._context_window_tokens = lambda: 40960

    assert tui._cmd_context("") is True
    assert "SECTION:Context" in lines
    assert "DETAIL:usage:4%:False" in lines
    assert "DETAIL:tokens:1612 / 40960:False" in lines


def test_handle_context_command_renders_context_summary() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tui._id = "app"
    tui._reactive_streaming = False
    rendered: list[str] = []
    tui._cmd_context = lambda arg: rendered.append(arg) or True

    assert tui._handle_command("/context") is True
    assert rendered == [""]


def test_show_keymap_writes_expected_sections() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    lines: list[str] = []
    command_cols: list[int] = []
    tui._write = lines.append
    tui._write_section_heading = lambda text: lines.append(f"SECTION:{text}")

    def capture_command_row(command: str, desc: str, *, col: int) -> None:
        command_cols.append(col)
        lines.append(f"ROW:{command}:{desc}:{col}")

    tui._write_command_row = capture_command_row

    tui.action_show_keymap()

    assert "SECTION:Keymap" in lines
    assert any("F1 / ?" in line for line in lines)
    assert any("Ctrl+P / /" in line for line in lines)
    assert any("Tab / Shift+Tab" in line for line in lines)
    assert any("SECTION:Tree" == line for line in lines)
    assert any("SECTION:Slash Palette" == line for line in lines)
    assert any("Enter / o" in line for line in lines)
    assert len(set(command_cols)) == 1


def test_handle_keyboard_shortcuts_command_renders_keymap() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tui._id = "app"
    tui._reactive_streaming = False
    rendered: list[str] = []
    tui._show_keyboard_shortcuts = lambda: rendered.append("keymap")

    assert tui._handle_command("/keyboard-shortcuts") is True
    assert rendered == ["keymap"]


def test_handle_save_renames_and_persists_active_session() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tree = ConvTree()
    saved_calls: list[tuple[str, str, str]] = []
    actions: list[str] = []
    errors: list[str] = []
    tui._id = "app"
    tui._reactive_streaming = False

    def save_tree(session_id: str, title: str, saved_tree: ConvTree, *, created_at: str, activate: bool = True) -> ChatSession:
        saved_calls.append((session_id, title, created_at))
        assert saved_tree is tree
        assert activate is True
        return ChatSession(
            id=session_id,
            title=title,
            created_at=created_at,
            updated_at="2026-03-20T10:30:00+00:00",
            tree=saved_tree,
        )

    tui._session_store = SimpleNamespace(save_tree=save_tree)
    tui._session_id = "sess-1"
    tui._session_title = "Session 1"
    tui._session_created_at = "2026-03-20T10:00:00+00:00"
    tui.conv_tree = tree
    tui._write_command_action = lambda text, **_kwargs: actions.append(text)
    tui._write_error = errors.append
    tui._update_topbar = lambda: None

    assert tui._handle_command("/save Backend Work") is True
    assert saved_calls == [("sess-1", "Backend Work", "2026-03-20T10:00:00+00:00")]
    assert tui._session_title == "Backend Work"
    assert errors == []
    assert actions == ["Saved session 'Backend Work'"]


def test_handle_load_switches_sessions_and_rebuilds_view() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tui._id = "app"
    tui._reactive_streaming = False
    opened: list[str] = []
    tui._open_load_session_picker = lambda: opened.append("load")

    assert tui._handle_command("/load") is True
    assert opened == ["load"]


def test_handle_file_without_path_opens_attachment_picker() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tui._id = "app"
    tui._reactive_streaming = False
    opened: list[str] = []
    tui._open_attachment_picker = lambda relative_dir=".": opened.append(relative_dir)

    assert tui._handle_command("/file") is True
    assert opened == ["."]


def test_handle_file_resolves_workspace_relative_path(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    target = workspace_root / "notes.txt"
    target.write_text("hello", encoding="utf-8")

    tui = AlphanusTUI.__new__(AlphanusTUI)
    tui._id = "app"
    tui._reactive_streaming = False
    tui.agent = SimpleNamespace(skill_runtime=SimpleNamespace(workspace=SimpleNamespace(workspace_root=str(workspace_root))))
    tui.pending = []
    tui._write_error = lambda *_args, **_kwargs: None
    tui._write_info = lambda *_args, **_kwargs: None
    tui._write_command_action = lambda *_args, **_kwargs: None
    tui._update_pending_attachments = lambda: None
    tui._update_status1 = lambda: None

    assert tui._handle_command("/file notes.txt") is True
    assert tui.pending == [(str(target.resolve()), "text")]


def test_attachment_picker_items_include_home_source_and_home_files(tmp_path: Path) -> None:
    home_root = tmp_path / "home"
    workspace_root = home_root / "workspace"
    downloads = home_root / "Downloads"
    workspace_root.mkdir(parents=True)
    downloads.mkdir(parents=True)
    (downloads / "notes.txt").write_text("hello", encoding="utf-8")

    tui = AlphanusTUI.__new__(AlphanusTUI)
    tui.agent = SimpleNamespace(skill_runtime=SimpleNamespace(workspace=WorkspaceManager(str(workspace_root), home_root=str(home_root))))

    workspace_items = tui._attachment_picker_items(".", root_id="workspace")
    home_items = tui._attachment_picker_items("Downloads", root_id="home")

    assert any(item.id == "root:home:." for item in workspace_items)
    assert any(item.id == "file:Downloads/notes.txt" for item in home_items)


def test_open_new_session_reuses_blank_current_session() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tui.conv_tree = ConvTree()
    reused = ChatSession(
        id="sess-1",
        title="Fresh Session",
        created_at="2026-03-20T10:00:00+00:00",
        updated_at="2026-03-20T10:01:00+00:00",
        tree=tui.conv_tree,
    )
    titles: list[str | None] = []

    tui._save_active_session = lambda rename_to=None: titles.append(rename_to) or reused
    tui._session_store = SimpleNamespace(create_session=lambda _title: (_ for _ in ()).throw(AssertionError("should not create")))

    session = tui._open_new_session("Fresh Session")

    assert session is reused
    assert titles == ["Fresh Session"]


def test_startup_session_picker_close_loads_selected_session() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    loaded = ChatSession(
        id="sess-2",
        title="Loaded Session",
        created_at="2026-03-20T10:00:00+00:00",
        updated_at="2026-03-20T10:01:00+00:00",
        tree=ConvTree(),
    )
    switched: list[ChatSession] = []

    tui._session_store = SimpleNamespace(load_session=lambda selector: loaded if selector == "sess-2" else None)
    tui._switch_to_session = lambda session, clear_pending=True: switched.append(session)

    tui._on_startup_session_picker_close({"action": "load", "selector": "sess-2"})

    assert switched == [loaded]


def test_startup_session_picker_close_reports_load_failure() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    errors: list[str] = []
    switched: list[ChatSession] = []

    def load_session(_selector: str) -> ChatSession:
        raise ValueError("broken session file")

    tui._session_store = SimpleNamespace(load_session=load_session)
    tui._switch_to_session = lambda session, clear_pending=True: switched.append(session)
    tui._write_error = errors.append

    tui._on_startup_session_picker_close({"action": "load", "selector": "sess-2"})

    assert switched == []
    assert errors == ["Load failed: broken session file"]


def test_load_session_picker_close_switches_sessions() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    current_tree = ConvTree()
    loaded_tree = ConvTree()
    loaded_turn = loaded_tree.add_turn("loaded")
    loaded_tree.complete_turn(loaded_turn.id, "done")
    events: list[str] = []
    errors: list[str] = []

    def save_tree(session_id: str, title: str, tree: ConvTree, *, created_at: str, activate: bool = True) -> ChatSession:
        assert tree is current_tree
        return ChatSession(
            id=session_id,
            title=title,
            created_at=created_at,
            updated_at="2026-03-20T10:05:00+00:00",
            tree=tree,
        )

    def load_session(selector: str) -> ChatSession:
        assert selector == "sess-2"
        return ChatSession(
            id="sess-2",
            title="Loaded Session",
            created_at="2026-03-20T10:06:00+00:00",
            updated_at="2026-03-20T10:07:00+00:00",
            tree=loaded_tree,
        )

    tui._session_store = SimpleNamespace(save_tree=save_tree, load_session=load_session)
    tui._session_id = "sess-1"
    tui._session_title = "Session 1"
    tui._session_created_at = "2026-03-20T10:00:00+00:00"
    tui.conv_tree = current_tree
    tui.pending = [("/tmp/example.txt", "text")]
    tui._switch_to_session = lambda session, clear_pending=True: events.append(session.title)
    tui._write_command_action = lambda text, **_kwargs: events.append(text)
    tui._write_error = errors.append

    tui._on_load_session_picker_close({"id": "sess-2"})

    assert errors == []
    assert events == ["Loaded Session", "Loaded session 'Loaded Session'"]


def test_import_picker_close_imports_selected_export() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    current_tree = ConvTree()
    imported_tree = ConvTree()
    imported_turn = imported_tree.add_turn("imported")
    imported_tree.complete_turn(imported_turn.id, "done")
    events: list[str] = []
    errors: list[str] = []

    def save_tree(session_id: str, title: str, tree: ConvTree, *, created_at: str, activate: bool = True) -> ChatSession:
        assert tree is current_tree
        return ChatSession(
            id=session_id,
            title=title,
            created_at=created_at,
            updated_at="2026-03-20T10:05:00+00:00",
            tree=tree,
        )

    def import_tree(path, title: str = "", activate: bool = True) -> ChatSession:
        assert str(path).endswith("export.json")
        return ChatSession(
            id="sess-3",
            title="Imported Session",
            created_at="2026-03-20T10:06:00+00:00",
            updated_at="2026-03-20T10:07:00+00:00",
            tree=imported_tree,
        )

    tui._session_store = SimpleNamespace(
        save_tree=save_tree,
        resolve_export_path=lambda selector: selector,
        import_tree=import_tree,
    )
    tui._session_id = "sess-1"
    tui._session_title = "Session 1"
    tui._session_created_at = "2026-03-20T10:00:00+00:00"
    tui.conv_tree = current_tree
    tui._switch_to_session = lambda session, clear_pending=True: events.append(session.title)
    tui._write_command_action = lambda text, **_kwargs: events.append(text)
    tui._write_error = errors.append

    tui._on_import_picker_close({"id": "export.json"})

    assert errors == []
    assert events == ["Imported Session", "Imported session 'Imported Session'"]


def test_session_row_label_highlights_active_session() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    label = tui._session_row_label(
        SimpleNamespace(
            is_active=True,
            title="Test123",
            id="25e4a7bf",
            turn_count=0,
            branch_count=0,
            updated_at="2026-03-19T21:14:00+00:00",
        )
    )

    assert "active" in label
    assert "#6366f1" in label
    assert "[25e4a7bf]" in label


def test_activate_session_state_moves_non_empty_root_session_to_latest_leaf() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tree = ConvTree()
    first = tree.add_turn("first")
    tree.complete_turn(first.id, "one")
    second = tree.add_turn("second")
    tree.complete_turn(second.id, "two")
    tree.current_id = "root"

    session = ChatSession(
        id="sess-2",
        title="Loaded Session",
        created_at="2026-03-20T10:00:00+00:00",
        updated_at="2026-03-20T10:05:00+00:00",
        tree=tree,
    )
    tui._apply_tree_compaction_policy = lambda current_tree: current_tree

    tui._activate_session_state(session)

    assert tui.conv_tree.current_id == second.id
    assert tui._tree_cursor_id == second.id


def test_activate_session_state_preserves_root_when_pending_branch_is_armed() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tree = ConvTree()
    first = tree.add_turn("first")
    tree.complete_turn(first.id, "one")
    tree.current_id = "root"
    tree.arm_branch("new-root-branch")

    session = ChatSession(
        id="sess-2",
        title="Loaded Session",
        created_at="2026-03-20T10:00:00+00:00",
        updated_at="2026-03-20T10:05:00+00:00",
        tree=tree,
    )
    tui._apply_tree_compaction_policy = lambda current_tree: current_tree

    tui._activate_session_state(session)

    assert tui.conv_tree.current_id == "root"
    assert tui.conv_tree._pending_branch is True
    assert tui._tree_cursor_id == "root"


def test_activate_session_state_uses_newest_leaf_not_rightmost_descendant() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tree = ConvTree()
    root_turn = tree.add_turn("root turn")
    tree.complete_turn(root_turn.id, "root done")

    tree.current_id = root_turn.id
    tree.arm_branch("left")
    left = tree.add_turn("left")
    tree.complete_turn(left.id, "left done")

    tree.current_id = root_turn.id
    tree.arm_branch("right")
    right = tree.add_turn("right")
    tree.complete_turn(right.id, "right done")

    tree.current_id = left.id
    newest = tree.add_turn("latest under left")
    tree.complete_turn(newest.id, "latest done")
    tree.current_id = "root"

    session = ChatSession(
        id="sess-3",
        title="Branched Session",
        created_at="2026-03-20T10:00:00+00:00",
        updated_at="2026-03-20T10:05:00+00:00",
        tree=tree,
    )
    tui._apply_tree_compaction_policy = lambda current_tree: current_tree

    tui._activate_session_state(session)

    assert tui.conv_tree.current_id == newest.id
    assert tui._tree_cursor_id == newest.id


def test_switch_to_session_resets_context_usage_and_refreshes_topbar() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    session = ChatSession(
        id="sess-4",
        title="Next Session",
        created_at="2026-03-20T10:00:00+00:00",
        updated_at="2026-03-20T10:05:00+00:00",
        tree=ConvTree(),
    )
    events: list[str] = []
    tui.pending = [("/tmp/example.txt", "text")]
    tui._last_model_context_tokens = 512
    tui._activate_session_state = lambda current: events.append(f"activate:{current.title}")
    tui._rebuild_viewport = lambda: events.append("rebuild")
    tui._update_sidebar = lambda: events.append("sidebar")
    tui._update_pending_attachments = lambda: events.append("attachments")
    tui._update_status1 = lambda: events.append("status1")
    tui._update_status2 = lambda: events.append("status2")
    tui._update_input_placeholder = lambda: events.append("placeholder")
    tui._update_topbar = lambda: events.append("topbar")

    tui._switch_to_session(session)

    assert tui._last_model_context_tokens is None
    assert tui.pending == []
    assert events == ["activate:Next Session", "rebuild", "sidebar", "attachments", "status1", "status2", "placeholder", "topbar"]


def test_handle_clear_resets_context_usage_and_refreshes_topbar() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tui._id = "app"
    tui._reactive_streaming = False
    tui.conv_tree = ConvTree()
    tui.pending = [("/tmp/example.txt", "text")]
    tui._last_model_context_tokens = 768
    events: list[str] = []
    fresh_tree = ConvTree()
    tui._new_conv_tree = lambda: fresh_tree
    tui._log = lambda: SimpleNamespace(clear=lambda: events.append("log"))
    tui._partial = lambda: SimpleNamespace(update=lambda _value: events.append("partial"))
    tui._save_active_session = lambda rename_to=None: events.append("save")
    tui._update_pending_attachments = lambda: events.append("attachments")
    tui._update_status1 = lambda: events.append("status1")
    tui._update_status2 = lambda: events.append("status2")
    tui._update_sidebar = lambda: events.append("sidebar")
    tui._update_input_placeholder = lambda: events.append("placeholder")
    tui._update_topbar = lambda: events.append("topbar")

    assert tui._handle_command("/clear") is True
    assert tui.conv_tree is fresh_tree
    assert tui.pending == []
    assert tui._last_model_context_tokens is None
    assert events == ["log", "partial", "save", "attachments", "status1", "status2", "sidebar", "placeholder", "topbar"]


def test_session_picker_name_submit_creates_new_session() -> None:
    modal = SessionPickerModal([], "sess-1", "Session 1")
    dismissed: list[dict[str, str]] = []
    modal.dismiss = lambda payload=None: dismissed.append(payload)

    modal._new_name_submitted(SimpleNamespace(value="Backend Work"))

    assert dismissed == [{"action": "new", "title": "Backend Work"}]
