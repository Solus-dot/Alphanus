from __future__ import annotations

import queue
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from rich.text import Text

from core.conv_tree import ConvTree
from core.sessions import ChatSession
from core.workspace import WorkspaceManager
from agent.policies import OutputSanitizer
from tui.live_tool_preview import LiveToolPreviewManager
from tui.commands import active_command_query, active_command_span, command_entries_for_query, popup_command_query
from tui.interface import AlphanusTUI, ChatInput
from tui.popups import SessionPickerModal


def _render_lines(renderable, *, width: int = 40) -> list[str]:
    console = Console(width=width, record=True)
    console.print(renderable, end="")
    return [line for line in console.export_text().splitlines() if line]


def _assert_barred(renderable, *, width: int = 40) -> None:
    lines = _render_lines(renderable, width=width)
    assert lines
    assert all(line.startswith("┃ ") for line in lines)


def _tui_agent_stub(tmp_path: Path) -> SimpleNamespace:
    workspace_root = tmp_path / "ws"
    workspace_root.mkdir()
    return SimpleNamespace(
        config={"tui": {}},
        skill_runtime=SimpleNamespace(workspace=SimpleNamespace(workspace_root=workspace_root)),
        connect_timeout_s=1.0,
        model_endpoint="http://127.0.0.1:8080/v1/chat/completions",
        fetch_model_metadata=lambda timeout_s=None: (None, None),
        reload_skills=lambda: 0,
        run_turn=lambda *args, **kwargs: None,
        doctor_report=lambda: {},
    )


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


def test_active_command_query_tracks_command_token_at_cursor() -> None:
    assert active_command_query("/cont", 5) == "/cont"
    assert active_command_query("/cont notes", 3) == "/cont"
    assert active_command_query("/cont notes", 5) == "/cont"
    assert active_command_query("/cont notes", 7) == ""


def test_active_command_query_ignores_non_command_prefix_text() -> None:
    assert active_command_query("note /cont", 6) == ""


def test_popup_command_query_keeps_exact_commands() -> None:
    assert popup_command_query("/exit", 5) == "/exit"
    assert popup_command_query("/quit", 5) == "/quit"


def test_popup_command_query_keeps_partial_command_matches() -> None:
    assert popup_command_query("/ex", 3) == "/ex"
    assert popup_command_query("/cont", 5) == "/cont"


@pytest.mark.anyio
async def test_command_popup_tracks_chat_input_geometry(tmp_path: Path) -> None:
    tui = AlphanusTUI(_tui_agent_stub(tmp_path))
    tui._open_startup_session_picker = lambda: None
    tui._maybe_refresh_model_name = lambda force=False: None

    async with tui.run_test() as pilot:
        await pilot.press("/")
        await pilot.pause()
        popup = tui.query_one("#command-popup")
        chat_input = tui.query_one(ChatInput)
        separator = tui.query_one("#footer-sep")

        assert popup.display is True
        assert popup.region.width <= 72
        assert popup.region.y < separator.region.y
        assert popup.region.bottom <= separator.region.y
        assert popup.region.x >= chat_input.region.x
        assert tui.query_one("#command-options").region.bottom <= popup.region.bottom


@pytest.mark.anyio
async def test_command_popup_hides_for_unknown_query_and_reappears_when_query_matches(tmp_path: Path) -> None:
    tui = AlphanusTUI(_tui_agent_stub(tmp_path))
    tui._open_startup_session_picker = lambda: None
    tui._maybe_refresh_model_name = lambda force=False: None

    async with tui.run_test() as pilot:
        await pilot.press("/", "z")
        await pilot.pause()
        popup = tui.query_one("#command-popup")
        assert popup.display is False

        await pilot.press("backspace", "h")
        await pilot.pause()
        assert popup.display is True


@pytest.mark.anyio
async def test_command_popup_narrows_from_he_to_hel_and_keeps_help_visible(tmp_path: Path) -> None:
    tui = AlphanusTUI(_tui_agent_stub(tmp_path))
    tui._open_startup_session_picker = lambda: None
    tui._maybe_refresh_model_name = lambda force=False: None

    async with tui.run_test() as pilot:
        await pilot.press("/", "h", "e")
        await pilot.pause()
        assert tui.query_one("#command-popup").display is True
        assert "/help" in [entry.prompt for entry in tui._command_matches]

        await pilot.press("l")
        await pilot.pause()
        assert tui.query_one("#command-popup").display is True
        assert [entry.prompt for entry in tui._command_matches] == ["/help"]
        assert tui.query_one("#command-options").region.bottom <= tui.query_one("#command-popup").region.bottom


def test_active_command_span_covers_command_token() -> None:
    assert active_command_span("/cont notes", 3) == (0, 5)
    assert active_command_span("  /cont notes", 2) == (2, 7)


def test_chat_input_binds_new_shortcuts_locally() -> None:
    bindings = {binding.key: binding.action for binding in ChatInput.BINDINGS}

    assert bindings["ctrl+g"] == "focus_input"
    assert bindings["ctrl+p"] == "open_command_palette"
    assert bindings["f1"] == "show_keymap"
    assert bindings["f2"] == "toggle_details"
    assert bindings["f3"] == "toggle_thinking"


def test_config_editor_view_omits_secrets_and_internal_fields() -> None:
    config = {
        "agent": {"auth_header": "Authorization: Bearer secret", "context_budget_max_tokens": 2048},
        "search": {"provider": "tavily", "tavily_api_key": "tvly-secret"},
        "memory": {"model_name": "BAAI/bge-small-en-v1.5"},
        "context": {"context_limit": 8192, "safety_margin": 500, "keep_last_n": 10},
    }

    cleaned = AlphanusTUI._config_for_editor(config)

    assert "auth_header" not in cleaned["agent"]
    assert "context_budget_max_tokens" not in cleaned["agent"]
    assert "tavily_api_key" not in cleaned["search"]
    assert "context_limit" not in cleaned["context"]
    assert "safety_margin" not in cleaned["context"]
    assert cleaned["context"]["keep_last_n"] == 10


def test_tool_result_lines_hidden_when_details_off() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tui._show_tool_details = False
    tui._live_preview = SimpleNamespace(streamed_file_tools={"create_file", "edit_file"})

    assert tui._show_tool_result_line("create_file", True) is False
    assert tui._show_tool_result_line("web_search", True) is False


def test_write_skill_exchanges_skips_historical_previews_when_details_off() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tui._show_tool_details = False
    tui._live_preview = SimpleNamespace(
        compact_tool_args=lambda *_args, **_kwargs: "",
        write_static_preview=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not render")),
        write_result_preview=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not render")),
    )

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
            },
            {
                "role": "tool",
                "name": "create_file",
                "content": "{\"ok\": true, \"data\": {}}",
            },
        ]
    )

    assert tui._write_skill_exchanges(turn) is None


def test_write_skill_exchanges_keeps_failed_results_visible_when_details_off() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    writes: list[tuple[str, bool, str]] = []
    tui._show_tool_details = False
    tui._live_preview = SimpleNamespace(
        compact_tool_args=lambda *_args, **_kwargs: "",
        write_static_preview=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not render preview")),
        write_result_preview=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not render preview")),
    )
    tui._write_tool_lifecycle_block = lambda name, ok, detail="": writes.append((name, ok, detail))

    turn = SimpleNamespace(
        skill_exchanges=[
            {
                "role": "tool",
                "name": "create_file",
                "content": "{\"ok\": false, \"error\": {\"message\": \"blocked\"}}",
            }
        ]
    )

    assert tui._write_skill_exchanges(turn) is None
    assert writes == [("create_file", False, "blocked")]


def test_write_turn_user_renders_green_edge_bar_without_role_label() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    writes: list[str] = []
    renderables: list[tuple[object, int]] = []
    tui._write = writes.append
    tui._write_renderable = lambda renderable, indent=0: renderables.append((renderable, indent))

    turn = SimpleNamespace(
        branch_root=False,
        label="",
        attachment_summary=lambda: "notes.txt",
        user_text=lambda: "Hello\nA wrapped user line for the rail",
    )

    tui._write_turn_user(turn)

    assert writes == [""]
    assert all("You" not in line for line in writes)
    assert [indent for _renderable, indent in renderables] == [0, 0, 0]
    _assert_barred(renderables[0][0], width=30)
    _assert_barred(renderables[1][0], width=30)
    _assert_barred(renderables[2][0], width=20)


def test_write_completed_turn_asst_omits_role_label() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    writes: list[str] = []
    renderables: list[tuple[object, int]] = []
    tui._write = writes.append
    tui._write_renderable = lambda renderable, indent=0: renderables.append((renderable, indent))
    tui._write_skill_exchanges = lambda _turn: None

    turn = SimpleNamespace(assistant_content="A wrapped assistant reply for the rail", assistant_state="done")

    tui._write_completed_turn_asst(turn)

    assert writes == ["", ""]
    assert all("Assistant" not in line for line in writes)
    assert [indent for _renderable, indent in renderables] == [0]
    _assert_barred(renderables[0][0], width=20)


def test_update_partial_content_renders_assistant_bar_during_stream() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)

    class PartialStub:
        def __init__(self) -> None:
            self.value = None

        def update(self, value) -> None:
            self.value = value

    partial = PartialStub()
    tui._partial = lambda: partial
    tui._in_fence = False
    tui._buf_c = "Streaming reply that wraps"

    tui._update_partial_content()

    _assert_barred(partial.value, width=20)


def test_edge_bar_preserves_content_indent_on_wrapped_lines() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    renderable = tui._bar_renderable(
        Text.from_markup("* nested item that wraps across multiple visual lines"),
        "#6366f1",
        content_indent=4,
        continuation_indent=6,
    )

    lines = _render_lines(renderable, width=20)
    assert lines[0].startswith("┃     * ")
    assert all(line.startswith("┃       ") for line in lines[1:])


def test_tool_call_partial_renders_panel_with_assistant_bar() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)

    class PartialStub:
        def __init__(self) -> None:
            self.value = None
            self.display = False

        def update(self, value) -> None:
            self.value = value

    partial = PartialStub()
    tui._partial = lambda: partial

    tui._update_tool_call_partial("recall_memory", "query=user name")

    assert partial.display is True
    _assert_barred(partial.value, width=40)


def test_live_preview_partial_renders_panel_with_assistant_bar() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)

    class PartialStub:
        def __init__(self) -> None:
            self.value = None
            self.display = False

        def update(self, value) -> None:
            self.value = value

    partial = PartialStub()
    tui._partial = lambda: partial

    tui._update_live_preview_partial(["console.log('hi')"], "javascript")

    assert partial.display is True
    _assert_barred(partial.value, width=40)


def test_handle_content_token_uses_barred_spacer_after_reasoning() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    writes: list[str] = []
    tui._content_open = False
    tui._buf_c = ""
    tui._append_reply_token = lambda _token: None
    tui._close_reasoning_section = lambda: True
    tui._write_assistant_bar_line = lambda markup="", content_indent=0: writes.append((markup, content_indent))
    tui._flush_content_buffer = lambda include_partial=False, update_partial=True: None

    tui._handle_content_token("hello")

    assert writes == [("", 0)]


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


def test_reply_accumulator_preserves_content_and_caps_length() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tui._reply_acc = ""

    tui._append_reply_token("hello")
    tui._append_reply_token(" world")

    assert tui._reply_acc == "hello world"

    tui._reply_acc = ""
    tui._append_reply_token("x" * 30000)
    assert len(tui._reply_acc) == 24000


def test_output_sanitizer_removes_tool_markup() -> None:
    text = "<think>secret</think>\n<tool_call>ignore</tool_call>\nVisible\n<function=run>\n</function>"

    cleaned = OutputSanitizer(max_reasoning_chars=100).sanitize_final_content(text)

    assert cleaned == "Visible"
    assert OutputSanitizer(max_reasoning_chars=100).contains_tool_markup("<tool_call>run</tool_call>") is True


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


def test_visible_reasoning_text_strips_think_markers() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tui._is_tool_trace_line = lambda _line: False

    visible = tui._visible_reasoning_text("<think>\ninternal reasoning\n</think>")

    assert "<think>" not in visible
    assert "</think>" not in visible
    assert "internal reasoning" in visible


def test_drain_stream_event_queue_renders_reasoning_once_per_tick() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    partial_updates: list[object] = []
    scrolls: list[str] = []

    tui._stream_event_queue = queue.SimpleQueue()
    tui._stream_drain_active = False
    tui._stream_partial_dirty = False
    tui._deferred_live_preview = None
    tui._reasoning_open = False
    tui._content_open = False
    tui._buf_r = ""
    tui._buf_c = ""
    tui._last_scroll = 0.0
    tui._scroll_interval = 0.0
    tui._partial = lambda: SimpleNamespace(update=partial_updates.append, display=False)
    tui._reasoning_panel_renderable = lambda text: text
    tui._visible_reasoning_text = lambda text: text
    tui._update_partial_content = lambda: partial_updates.append("content")
    tui._maybe_scroll_end = lambda *args, **kwargs: scrolls.append("scroll")

    tui._stream_event_queue.put({"type": "reasoning_token", "text": "hel"})
    tui._stream_event_queue.put({"type": "reasoning_token", "text": "lo"})

    tui._drain_stream_event_queue()

    assert tui._buf_r == "hello"
    assert len(partial_updates) == 1
    assert scrolls == ["scroll"]


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
    tui._write_assistant_bar_line = lambda markup="", content_indent=0: writes.append(markup)
    tui._write_code_block = lambda code, language, indent=0: code_blocks.append((list(code), language, indent))
    tui._clear_partial_preview = lambda: None
    tui._close_reasoning_section = lambda: False
    tui._update_tool_call_partial = lambda name, detail="": partial_updates.append((name, detail))
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


def test_tool_call_delta_shows_fallback_partial_when_preview_not_ready() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    partial_updates: list[tuple[str, str]] = []

    tui._show_tool_details = True
    tui._live_preview = SimpleNamespace(update=lambda *_args, **_kwargs: False)
    tui._update_tool_call_partial = lambda name, detail="": partial_updates.append((name, detail))
    tui._partial = lambda: SimpleNamespace(update=lambda *_args, **_kwargs: None, display=False)
    tui._last_scroll = 0.0
    tui._scroll_interval = 999.0
    tui._maybe_scroll_end = lambda *args, **kwargs: None

    tui._on_agent_event(
        {
            "type": "tool_call_delta",
            "stream_id": "s1",
            "name": "create_files",
            "raw_arguments": '{"files":[',
        }
    )

    assert partial_updates == [("create_files", "streaming…")]


def test_live_tool_preview_update_returns_false_for_empty_content() -> None:
    preview = LiveToolPreviewManager()
    lines: list[str] = []
    partial_updates: list[tuple[list[str], str | None]] = []

    rendered = preview.update(
        "stream-1",
        "create_file",
        '{"filepath":"a.txt","content":""}',
        lines.append,
        lambda code, language: partial_updates.append((code, language)),
    )

    assert rendered is False
    assert lines == []
    assert partial_updates == []

def test_update_context_usage_ignores_total_tokens_without_prompt_tokens() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tui._last_model_context_tokens = 321
    updates: list[str] = []
    tui._update_topbar = lambda: updates.append("topbar")

    tui._update_context_usage_from_payload({"total_tokens": 999})

    assert tui._last_model_context_tokens == 321
    assert updates == ["topbar"]


def test_update_context_usage_accepts_llamacpp_prompt_eval_count() -> None:
    tui = AlphanusTUI.__new__(AlphanusTUI)
    tui._last_model_context_tokens = None
    updates: list[str] = []
    tui._update_topbar = lambda: updates.append("topbar")

    tui._update_context_usage_from_payload({"prompt_eval_count": 512, "eval_count": 32})

    assert tui._last_model_context_tokens == 512
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


def test_cmd_skills_shows_trust_validation_and_shadowing() -> None:
    skill = SimpleNamespace(
        id="home-helper",
        version="1.0.0",
        description="Home helper",
        metadata={"_pack_id": "pack-1"},
        user_invocable=True,
        disable_model_invocation=False,
        trust_level="untrusted",
        execution_allowed=False,
        adapter="claude",
        blocked_features=["untrusted_root", "scripts"],
        validation_errors=["untrusted skill roots are metadata-only; executable surfaces are blocked"],
        shadowed_by="",
        available=False,
        availability_code="untrusted",
        availability_reason="untrusted skill roots are metadata-only",
    )
    runtime = SimpleNamespace(
        list_skills=lambda: [skill],
        skill_status_label=lambda _skill: ("blocked", "yellow"),
        skill_source_label=lambda _skill: "home/.claude/skills/home-helper",
        skill_provenance_label=lambda _skill: "user/local",
        _reported_skill_tools=lambda _skill: [],
        _reported_skill_scripts=lambda _skill: [],
        _reported_skill_entrypoints=lambda _skill: [],
        _reported_skill_agents=lambda _skill: [],
    )

    tui = AlphanusTUI.__new__(AlphanusTUI)
    lines: list[str] = []
    tui.agent = SimpleNamespace(skill_runtime=runtime)
    tui._write = lines.append
    tui._write_section_heading = lambda text: lines.append(f"SECTION:{text}")

    tui._cmd_skills()

    joined = "\n".join(lines)
    assert "SECTION:Skills" in joined
    assert "trust=untrusted" in joined
    assert "execution=no" in joined
    assert "adapter=claude" in joined
    assert "blocked_features: untrusted_root, scripts" in joined
    assert "validation:" in joined


def test_cmd_doctor_shows_skill_policy_details() -> None:
    report = {
        "agent": {"ready": True, "endpoint_policy_error": ""},
        "workspace": {"path": "/tmp/ws", "writable": True},
        "memory": {
            "mode": "semantic",
            "backend": "transformer",
            "allow_model_download": False,
            "encoder_status": "ready",
            "encoder_source": "transformer-local",
            "encoder_detail": "",
            "model_name": "BAAI/bge-small-en-v1.5",
            "recommended_model_name": "BAAI/bge-small-en-v1.5",
        },
        "search": {"provider": "tavily", "ready": False, "reason": "missing env: TAVILY_API_KEY"},
        "skills": [
            {
                "id": "dup-skill",
                "source_tier": "bundled",
                "pack_id": "standalone",
                "availability_code": "shadowed",
                "availability_reason": "shadowed by dup-skill (workspace/.claude/skills/dup-skill)",
                "status": "shadowed",
                "trust_level": "trusted",
                "execution_allowed": False,
                "adapter": "agentskills",
                "tools": [],
                "scripts": [],
                "entrypoints": [],
                "agents": [],
                "user_invocable": True,
                "model_invocable": True,
                "blocked_features": ["command_tools"],
                "validation_errors": ["command_tools are disabled_pending_safe_runner"],
                "shadowed_by": "dup-skill",
            }
        ],
    }

    tui = AlphanusTUI.__new__(AlphanusTUI)
    lines: list[str] = []
    tui.agent = SimpleNamespace(
        doctor_report=lambda: report,
        skill_runtime=SimpleNamespace(list_agents=lambda: []),
    )
    tui._write = lines.append
    tui._write_section_heading = lambda text: lines.append(f"SECTION:{text}")
    tui._write_detail_line = lambda label, value, value_markup=False: lines.append(f"DETAIL:{label}:{value}:{value_markup}")

    tui._cmd_doctor()

    joined = "\n".join(lines)
    assert "SECTION:Doctor" in joined
    assert "SECTION:Skills" in joined
    assert "trust=trusted" in joined
    assert "execution=no" in joined
    assert "adapter=agentskills" in joined
    assert "blocked_features: command_tools" in joined
    assert "shadowed_by: dup-skill" in joined
