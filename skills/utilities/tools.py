from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
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
}


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _get_weather(args: Dict[str, Any]) -> Dict[str, Any]:
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
    return {
        "city": city,
        "temp_c": current.get("temp_C"),
        "feels_like_c": current.get("FeelsLikeC"),
        "desc": (current.get("weatherDesc") or [{"value": ""}])[0].get("value", ""),
        "humidity": current.get("humidity"),
    }


def _search_home_files(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    query = str(args["query"]).lower()
    home_root = Path(os.path.expanduser(str(env.workspace.home_root))).resolve()
    directory = str(args.get("directory") or str(home_root))
    root = Path(os.path.expanduser(directory)).resolve()
    if not _is_under(root, home_root):
        raise PermissionError("Search directory outside home root")

    matches = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if query in name.lower():
                matches.append(str(Path(dirpath) / name))
                if len(matches) >= 200:
                    break
        if len(matches) >= 200:
            break
    return {"matches": matches}


def execute(tool_name: str, args: Dict[str, Any], env: ToolExecutionEnv):
    if tool_name == "get_weather":
        return _get_weather(args)
    if tool_name == "search_home_files":
        return _search_home_files(args, env)
    raise ValueError(f"Unsupported tool: {tool_name}")
