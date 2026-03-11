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
