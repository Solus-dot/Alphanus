from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict

from core.skills import ToolExecutionEnv

TOOL_SPECS = {
    "get_weather": {
        "capability": "utility_weather",
        "description": "Fetch weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    "search_home_files": {
        "capability": "utility_file_search",
        "description": "Search filenames under home directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "directory": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    "open_url": {
        "capability": "utility_open_url",
        "description": "Open URL in default browser.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    "play_youtube": {
        "capability": "utility_play_youtube",
        "description": "Open the first YouTube video result for a topic and autoplay when resolvable.",
        "parameters": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    },
}

_VIDEO_ID_RE = re.compile(r'"videoId":"([A-Za-z0-9_-]{11})"')


def _is_under(path: Path, root: Path) -> bool:
    return path.is_relative_to(root)


def _ok(data: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, "data": data, "error": None, "meta": {}}


def _err(code: str, message: str, data: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {"ok": False, "data": data, "error": {"code": code, "message": message}, "meta": {}}


def _get_weather(args: Dict[str, Any]) -> Dict[str, Any]:
    city = str(args["city"]).strip()
    query = urllib.parse.quote(city)
    url = f"https://wttr.in/{query}?format=j1"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return _err("E_IO", f"Weather service returned HTTP {exc.code}", {"city": city})
    except urllib.error.URLError as exc:
        return _err("E_IO", f"Weather service unreachable: {exc.reason}", {"city": city})
    except json.JSONDecodeError as exc:
        return _err("E_IO", "Weather service returned invalid JSON", {"city": city})

    current = payload.get("current_condition", [{}])[0]
    return _ok(
        {
            "city": city,
            "temp_c": current.get("temp_C"),
            "feels_like_c": current.get("FeelsLikeC"),
            "desc": (current.get("weatherDesc") or [{"value": ""}])[0].get("value", ""),
            "humidity": current.get("humidity"),
        }
    )


def _search_home_files(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    query = str(args["query"]).lower()
    home_root = Path(os.path.expanduser(str(env.workspace.home_root))).resolve()
    directory = str(args.get("directory") or str(home_root))
    root = Path(os.path.expanduser(directory)).resolve()
    if not _is_under(root, home_root):
        raise PermissionError("Search directory outside home root")

    matches = []
    ignore_dirs = {".git", "node_modules", ".venv", "venv", "__pycache__", ".next", "dist", "build"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in ignore_dirs]
        for name in filenames:
            if query in name.lower():
                matches.append(str(Path(dirpath) / name))
                if len(matches) >= 200:
                    break
        if len(matches) >= 200:
            break
    return _ok({"matches": matches})


def _open_url(args: Dict[str, Any]) -> Dict[str, Any]:
    url = str(args["url"]).strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https", "file"}:
        raise ValueError("URL must start with http://, https://, or file://")
    if parsed.scheme == "file" and not parsed.path:
        raise ValueError("file:// URL must include a path")
    try:
        opened = webbrowser.open(url, new=2)
    except Exception as exc:
        return _err("E_IO", f"Browser launch failed: {exc}", {"url": url})
    if not opened:
        return _err("E_IO", "Unable to open browser in this environment", {"url": url})
    return _ok({"url": url})


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


def _play_youtube(args: Dict[str, Any]) -> Dict[str, Any]:
    topic = str(args["topic"]).strip()
    search_url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(topic)
    url, video_id, resolved = _resolve_first_video_url(search_url)
    try:
        opened = webbrowser.open(url, new=2)
    except Exception as exc:
        return _err("E_IO", f"Browser launch failed: {exc}", {"topic": topic, "url": url, "search_url": search_url})
    if not opened:
        return _err("E_IO", "Unable to open browser in this environment", {"topic": topic, "url": url, "search_url": search_url})
    return _ok(
        {
            "url": url,
            "topic": topic,
            "search_url": search_url,
            "video_id": video_id,
            "resolved_first_result": resolved,
        }
    )


def execute(tool_name: str, args: Dict[str, Any], env: ToolExecutionEnv):
    if tool_name == "get_weather":
        return _get_weather(args)
    if tool_name == "search_home_files":
        return _search_home_files(args, env)
    if tool_name == "open_url":
        return _open_url(args)
    if tool_name == "play_youtube":
        return _play_youtube(args)
    raise ValueError(f"Unsupported tool: {tool_name}")
