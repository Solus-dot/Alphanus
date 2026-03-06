from __future__ import annotations

import imaplib
import json
import os
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict


TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "get_weather": {
        "capability": "utility_weather",
        "description": "Fetch weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    "read_email": {
        "capability": "utility_email_read",
        "description": "Read latest email metadata via IMAP.",
        "parameters": {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": [],
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
        "description": "Open first YouTube search URL for a topic.",
        "parameters": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    },
}


def execute(tool_name: str, args: Dict[str, Any], env) -> Dict[str, Any]:
    if tool_name == "get_weather":
        city = args["city"].strip()
        query = urllib.parse.quote(city)
        url = f"https://wttr.in/{query}?format=j1"
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        current = payload.get("current_condition", [{}])[0]
        weather = {
            "city": city,
            "temp_c": current.get("temp_C"),
            "feels_like_c": current.get("FeelsLikeC"),
            "desc": (current.get("weatherDesc") or [{"value": ""}])[0].get("value", ""),
            "humidity": current.get("humidity"),
        }
        return {"ok": True, "data": weather, "error": None, "meta": {}}

    if tool_name == "read_email":
        if not env.config.get("capabilities", {}).get("email_enabled", False):
            return {
                "ok": False,
                "data": None,
                "error": {"code": "E_POLICY", "message": "Email capability is disabled"},
                "meta": {},
            }
        user = os.getenv("EMAIL_USER")
        pw = os.getenv("EMAIL_PASSWORD")
        host = env.config.get("capabilities", {}).get("email_imap_server", "imap.gmail.com")
        if not user or not pw:
            return {
                "ok": False,
                "data": None,
                "error": {"code": "E_VALIDATION", "message": "EMAIL_USER or EMAIL_PASSWORD missing"},
                "meta": {},
            }
        count = int(args.get("count", 5))
        with imaplib.IMAP4_SSL(host) as mail:
            mail.login(user, pw)
            mail.select("INBOX")
            _, ids = mail.search(None, "ALL")
            msg_ids = ids[0].split()[-count:]
            out = [m.decode("utf-8", errors="replace") for m in msg_ids]
        return {"ok": True, "data": {"message_ids": out}, "error": None, "meta": {}}

    if tool_name == "search_home_files":
        query = args["query"].lower()
        directory = args.get("directory") or str(env.workspace.home_root)
        root = Path(os.path.expanduser(directory)).resolve()
        if not env.workspace._is_relative_to(root, env.workspace.home_root):  # pylint: disable=protected-access
            return {
                "ok": False,
                "data": None,
                "error": {"code": "E_POLICY", "message": "Search directory outside home root"},
                "meta": {},
            }
        matches = []
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if query in name.lower():
                    matches.append(str(Path(dirpath) / name))
                    if len(matches) >= 200:
                        break
            if len(matches) >= 200:
                break
        return {"ok": True, "data": {"matches": matches}, "error": None, "meta": {}}

    if tool_name == "open_url":
        url = args["url"]
        if not (url.startswith("http://") or url.startswith("https://")):
            return {
                "ok": False,
                "data": None,
                "error": {"code": "E_VALIDATION", "message": "URL must start with http:// or https://"},
                "meta": {},
            }
        try:
            opened = webbrowser.open(url, new=2)
            if not opened:
                raise RuntimeError("No browser handler")
        except Exception:
            return {
                "ok": False,
                "data": None,
                "error": {"code": "E_UNSUPPORTED", "message": "Unable to open browser in this environment"},
                "meta": {},
            }
        return {"ok": True, "data": {"url": url}, "error": None, "meta": {}}

    if tool_name == "play_youtube":
        topic = args["topic"].strip()
        url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(topic)
        try:
            opened = webbrowser.open(url, new=2)
            if not opened:
                raise RuntimeError("No browser handler")
        except Exception:
            return {
                "ok": False,
                "data": None,
                "error": {"code": "E_UNSUPPORTED", "message": "Unable to open browser in this environment"},
                "meta": {},
            }
        return {"ok": True, "data": {"url": url, "topic": topic}, "error": None, "meta": {}}

    return {
        "ok": False,
        "data": None,
        "error": {"code": "E_UNSUPPORTED", "message": f"Unsupported tool: {tool_name}"},
        "meta": {},
    }
