#!/usr/bin/env python3
from __future__ import annotations

import webbrowser

from core.tool_script import emit, read_args


def main() -> int:
    args = read_args()
    url = str(args["url"])
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("URL must start with http:// or https://")
    try:
        opened = webbrowser.open(url, new=2)
    except Exception as exc:
        raise RuntimeError(f"Browser launch failed: {exc}") from exc
    if not opened:
        raise RuntimeError("Unable to open browser in this environment")
    emit({"url": url})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
