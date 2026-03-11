#!/usr/bin/env python3
from __future__ import annotations

import re
import urllib.parse
import urllib.request
import webbrowser

from core.tool_script import emit, read_args

_VIDEO_ID_RE = re.compile(r'"videoId":"([A-Za-z0-9_-]{11})"')


def _resolve_first_video_url(search_url: str) -> tuple[str, str, bool]:
    req = urllib.request.Request(
        search_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            )
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return search_url, "", False

    match = _VIDEO_ID_RE.search(html)
    if not match:
        return search_url, "", False

    video_id = match.group(1)
    return f"https://www.youtube.com/watch?v={video_id}&autoplay=1", video_id, True


def main() -> int:
    args = read_args()
    topic = str(args["topic"]).strip()
    search_url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(topic)
    url, video_id, resolved = _resolve_first_video_url(search_url)
    try:
        opened = webbrowser.open(url, new=2)
    except Exception as exc:
        raise RuntimeError(f"Browser launch failed: {exc}") from exc
    if not opened:
        raise RuntimeError("Unable to open browser in this environment")
    emit(
        {
            "url": url,
            "topic": topic,
            "search_url": search_url,
            "video_id": video_id,
            "resolved_first_result": resolved,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
