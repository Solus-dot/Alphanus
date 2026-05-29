from __future__ import annotations

from tui.live_tool_preview import LiveToolPreviewManager


def test_live_preview_streams_completed_lines_before_close():
    manager = LiveToolPreviewManager()
    writes = []
    indented = []
    previews = []
    code_blocks = []

    manager.update(
        "s1",
        "create_file",
        '{"filepath":"demo.py","content":"print(1)\\nprint(2)',
        writes.append,
        lambda lines, language: previews.append((list(lines), language)),
    )

    assert writes == ["[dim]  · file draft: demo.py[/dim]"]
    assert previews == [(["print(1)", "print(2)"], "python")]

    streamed = manager.close(
        "s1",
        lambda text, indent: indented.append((text, indent)),
        lambda lines, language, indent: code_blocks.append((list(lines), language, indent)),
        lambda: previews.append((["<cleared>"], None)),
    )

    assert streamed is True
    assert code_blocks[-1] == (["print(1)", "print(2)"], "python", 2)
    assert previews[-1] == (["<cleared>"], None)


def test_live_preview_clips_long_single_line_stream():
    manager = LiveToolPreviewManager(max_static_preview_chars=8)
    writes = []
    indented = []
    previews = []
    code_blocks = []
    long_line = "a" * 20000
    clipped_line = "a" * 8

    manager.update(
        "s3",
        "create_file",
        '{"filepath":"demo.js","content":"' + long_line + '"}',
        writes.append,
        lambda lines, language: previews.append((list(lines), language)),
    )

    streamed = manager.close(
        "s3",
        lambda text, indent: indented.append((text, indent)),
        lambda lines, language, indent: code_blocks.append((list(lines), language, indent)),
        lambda: previews.append((["<cleared>"], None)),
    )

    assert streamed is True
    assert indented == [("[dim]... (preview clipped; file write still uses full content) ...[/dim]", 2)]
    assert previews[0] == ([clipped_line], "javascript")
    assert code_blocks[-1] == ([clipped_line], "javascript", 2)


def test_live_preview_language_detection_is_data_driven():
    assert LiveToolPreviewManager._guess_language("module.cxx") == "cpp"  # noqa: SLF001
    assert LiveToolPreviewManager._guess_language("component.tsx") == "typescript"  # noqa: SLF001
    assert LiveToolPreviewManager._guess_language("unknown.nope") is None  # noqa: SLF001


def test_static_file_preview_reports_display_clipping_not_write_truncation():
    manager = LiveToolPreviewManager(max_static_preview_chars=5)
    writes = []
    indented = []
    code_blocks = []

    manager.write_static_preview(
        "create_file",
        {"filepath": "demo.js", "content": "const value = 1;\n"},
        writes.append,
        lambda text, indent: indented.append((text, indent)),
        lambda lines, language, indent: code_blocks.append((list(lines), language, indent)),
    )

    assert writes == ["[dim]  · file draft: demo.js[/dim]"]
    assert code_blocks == [(["const"], "javascript", 2)]
    assert indented == [("[dim]... (preview clipped; file write still uses full content) ...[/dim]", 2)]


def test_static_file_preview_skips_compacted_history_content_when_file_is_unavailable():
    manager = LiveToolPreviewManager()
    writes = []
    indented = []
    code_blocks = []

    manager.write_static_preview(
        "create_file",
        {
            "filepath": "UNIV/style.css",
            "content": "body {}\n...[history excerpt; 4029 chars omitted]",
        },
        writes.append,
        lambda text, indent: indented.append((text, indent)),
        lambda lines, language, indent: code_blocks.append((list(lines), language, indent)),
    )

    assert writes == ["[dim]  · file draft: UNIV/style.css[/dim]"]
    assert indented == [("[dim]preview unavailable; compacted history no longer contains file contents[/dim]", 2)]
    assert code_blocks == []


def test_static_file_preview_restores_compacted_history_from_workspace_file(tmp_path):
    workspace = tmp_path / "workspace"
    target = workspace / "UNIV" / "style.css"
    target.parent.mkdir(parents=True)
    target.write_text("body {\n  color: black;\n}\n", encoding="utf-8")
    manager = LiveToolPreviewManager()
    writes = []
    indented = []
    code_blocks = []

    manager.write_static_preview(
        "create_file",
        {
            "filepath": "UNIV/style.css",
            "content": "body {}\n...[history excerpt; 4029 chars omitted]",
        },
        writes.append,
        lambda text, indent: indented.append((text, indent)),
        lambda lines, language, indent: code_blocks.append((list(lines), language, indent)),
        workspace_root=workspace,
    )

    assert writes == ["[dim]  · file draft: UNIV/style.css[/dim]"]
    assert code_blocks == [(["body {", "  color: black;", "}"], "css", 2)]
    assert indented == [("[dim]preview restored from current workspace file[/dim]", 2)]


def test_live_preview_resets_when_stream_rewinds():
    manager = LiveToolPreviewManager()
    writes = []
    previews = []

    manager.update(
        "s4",
        "create_file",
        '{"filepath":"demo.py","content":"print(1)\\nprint(2)',
        writes.append,
        lambda lines, language: previews.append((list(lines), language)),
    )
    manager.update(
        "s4",
        "create_file",
        '{"filepath":"demo.py","content":"reset"}',
        writes.append,
        lambda lines, language: previews.append((list(lines), language)),
    )

    assert writes == ["[dim]  · file draft: demo.py[/dim]", "[dim]  · file draft: demo.py[/dim]"]
    assert previews[-1] == (["reset"], "python")


def test_live_preview_can_leave_partial_visible_after_close():
    manager = LiveToolPreviewManager()
    previews = []
    cleared = []

    manager.update(
        "s5",
        "create_file",
        '{"filepath":"demo.py","content":"print(1)"}',
        lambda _text: None,
        lambda lines, language: previews.append((list(lines), language)),
    )

    streamed = manager.close(
        "s5",
        lambda _text, _indent: None,
        lambda _lines, _language, _indent: None,
        lambda: cleared.append("cleared"),
        retain_partial=True,
    )

    assert streamed is True
    assert previews == [(["print(1)"], "python")]
    assert cleared == []


def test_live_preview_accepts_namespaced_legacy_write_file_alias():
    manager = LiveToolPreviewManager()
    writes = []
    previews = []

    rendered = manager.update(
        "s6",
        "workspace-ops:write_file",
        '{"filepath":"login.html","content":"<main>Login</main>"}',
        writes.append,
        lambda lines, language: previews.append((list(lines), language)),
    )

    assert rendered is True
    assert writes == ["[dim]  · file draft: login.html[/dim]"]
    assert previews == [(["<main>Login</main>"], "html")]


def test_live_preview_streams_edit_file_content_mode():
    manager = LiveToolPreviewManager()
    writes = []
    previews = []

    rendered = manager.update(
        "s7",
        "edit_file",
        '{"filepath":"app.py","content":"print(2)"}',
        writes.append,
        lambda lines, language: previews.append((list(lines), language)),
    )

    assert rendered is True
    assert writes == ["[dim]  · file draft: app.py[/dim]"]
    assert previews == [(["print(2)"], "python")]


def test_live_preview_applies_final_filepath_before_flushing_pending_draft():
    manager = LiveToolPreviewManager()
    writes = []
    previews = []
    code_blocks = []

    manager.update(
        "s8",
        "create_file",
        '{"content":"<main>RPS</main>',
        writes.append,
        lambda lines, language: previews.append((list(lines), language)),
    )
    manager.apply_final_arguments(
        "s8",
        "create_file",
        {"filepath": "RPS.html", "content": "<main>RPS</main>"},
    )

    streamed = manager.close(
        "s8",
        lambda _text, _indent: None,
        lambda lines, language, indent: code_blocks.append((list(lines), language, indent)),
        lambda: previews.append((["<cleared>"], None)),
    )

    assert streamed is True
    assert writes == ["[dim]  · file draft: (pending filepath)[/dim]"]
    assert code_blocks == [(["<main>RPS</main>"], "html", 2)]
