#!/usr/bin/env python3
from __future__ import annotations

import json
import urllib.parse
import urllib.request
import urllib.error

from core.tool_script import emit, read_args


def main() -> int:
    args = read_args()
    city = str(args["city"]).strip()
    query = urllib.parse.quote(city)
    url = f"https://wttr.in/{query}?format=j1"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Weather service returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Weather service unreachable: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Weather service returned invalid JSON") from exc
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
