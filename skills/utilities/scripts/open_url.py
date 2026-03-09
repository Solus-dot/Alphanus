#!/usr/bin/env python3
from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.tool_script import emit, read_args  # noqa: E402


def main() -> int:
    args = read_args()
    url = str(args["url"])
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("URL must start with http:// or https://")
    opened = webbrowser.open(url, new=2)
    if not opened:
        raise RuntimeError("Unable to open browser in this environment")
    emit({"url": url})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
