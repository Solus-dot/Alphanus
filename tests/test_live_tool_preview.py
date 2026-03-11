from __future__ import annotations

from tui.live_tool_preview import LiveToolPreviewManager


def test_live_preview_streams_completed_lines_before_close():
    manager = LiveToolPreviewManager()
    writes = []
    indented = []
    code_blocks = []

    manager.update(
        "s1",
        "create_file",
        '{"filepath":"demo.py","content":"print(1)\\nprint(2)',
        writes.append,
        lambda text, indent: indented.append((text, indent)),
        lambda lines, language, indent: code_blocks.append((list(lines), language, indent)),
    )

    assert writes == ["[dim]  · file draft: demo.py[/dim]"]
    assert code_blocks == [(["print(1)"], "python", 2)]

    streamed = manager.close(
        "s1",
        lambda text, indent: indented.append((text, indent)),
        lambda lines, language, indent: code_blocks.append((list(lines), language, indent)),
    )

    assert streamed is True
    assert code_blocks[-1] == (["print(2)"], "python", 2)
