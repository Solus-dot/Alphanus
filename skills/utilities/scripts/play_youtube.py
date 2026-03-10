#!/usr/bin/env python3
from __future__ import annotations

import sys
import urllib.parse
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.tool_script import emit, read_args  # noqa: E402


def main() -> int:
    args = read_args()
    topic = str(args["topic"]).strip()
    url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(topic)
    try:
        opened = webbrowser.open(url, new=2)
    except Exception as exc:
        raise RuntimeError(f"Browser launch failed: {exc}") from exc
    if not opened:
        raise RuntimeError("Unable to open browser in this environment")
    emit({"url": url, "topic": topic})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
