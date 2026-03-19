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


def test_live_preview_create_files_flushes_previous_file_when_switching():
    manager = LiveToolPreviewManager()
    writes = []
    indented = []
    previews = []
    code_blocks = []

    manager.update(
        "s2",
        "create_files",
        '{"files":[{"filepath":"site/index.html","content":"<h1>Hello</h1>\\n"}',
        writes.append,
        lambda lines, language: previews.append((list(lines), language)),
        lambda text, indent: indented.append((text, indent)),
        lambda lines, language, indent: code_blocks.append((list(lines), language, indent)),
        lambda: previews.append((["<cleared>"], None)),
    )

    manager.update(
        "s2",
        "create_files",
        '{"files":[{"filepath":"site/index.html","content":"<h1>Hello</h1>\\n"},{"filepath":"site/style.css","content":"body { color: red;"}',
        writes.append,
        lambda lines, language: previews.append((list(lines), language)),
        lambda text, indent: indented.append((text, indent)),
        lambda lines, language, indent: code_blocks.append((list(lines), language, indent)),
        lambda: previews.append((["<cleared>"], None)),
    )

    assert writes[0] == "[dim]  · file draft: site/index.html[/dim]"
    assert writes[-1] == "[dim]  · file draft: site/style.css[/dim]"
    assert any("site/index.html" in line for line in writes)
    assert any("site/style.css" in line for line in writes)
    assert code_blocks[0] == (["<h1>Hello</h1>"], "html", 2)
    assert previews[-1] == (["body { color: red;"], "css")

    streamed = manager.close(
        "s2",
        lambda text, indent: indented.append((text, indent)),
        lambda lines, language, indent: code_blocks.append((list(lines), language, indent)),
        lambda: previews.append((["<cleared>"], None)),
    )

    assert streamed is True
    assert code_blocks[-1] == (["body { color: red;"], "css", 2)

