#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.tool_script import emit, read_args  # noqa: E402


def main() -> int:
    args = read_args()
    city = str(args["city"]).strip()
    query = urllib.parse.quote(city)
    url = f"https://wttr.in/{query}?format=j1"
    with urllib.request.urlopen(url, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    current = payload.get("current_condition", [{}])[0]
    emit(
        {
            "city": city,
            "temp_c": current.get("temp_C"),
            "feels_like_c": current.get("FeelsLikeC"),
            "desc": (current.get("weatherDesc") or [{"value": ""}])[0].get("value", ""),
            "humidity": current.get("humidity"),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
