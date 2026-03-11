from __future__ import annotations

from tui.markdown_utils import fence_language


def test_fence_language_extracts_info_string():
    assert fence_language("```python") == "python"
    assert fence_language("~~~bash") == "bash"


def test_fence_language_returns_none_when_missing():
    assert fence_language("```") is None
    assert fence_language("plain text") is None
