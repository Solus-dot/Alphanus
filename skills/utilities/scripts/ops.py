#!/usr/bin/env python3
from __future__ import annotations

import imaplib
import json
import os
import sys
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict


def _ok(data: Any) -> Dict[str, Any]:
    return {"ok": True, "data": data, "error": None, "meta": {}}


def _err(code: str, message: str) -> Dict[str, Any]:
    return {"ok": False, "data": None, "error": {"code": code, "message": message}, "meta": {}}


def _read_args() -> Dict[str, Any]:
    raw = os.getenv("ALPHANUS_TOOL_ARGS_JSON", "").strip()
    if not raw:
        raw = sys.stdin.read().strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Tool args must be a JSON object")
    return parsed


def _read_config() -> Dict[str, Any]:
    raw = os.getenv("ALPHANUS_CONFIG_JSON", "").strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def _is_under_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _run(tool_name: str, args: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "get_weather":
        city = str(args["city"]).strip()
        query = urllib.parse.quote(city)
        url = f"https://wttr.in/{query}?format=j1"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return _err("E_IO", f"Weather lookup failed: {exc}")
        current = payload.get("current_condition", [{}])[0]
        weather = {
            "city": city,
            "temp_c": current.get("temp_C"),
            "feels_like_c": current.get("FeelsLikeC"),
            "desc": (current.get("weatherDesc") or [{"value": ""}])[0].get("value", ""),
            "humidity": current.get("humidity"),
        }
        return _ok(weather)

    if tool_name == "read_email":
        if not config.get("capabilities", {}).get("email_enabled", False):
            return _err("E_POLICY", "Email capability is disabled")
        user = os.getenv("EMAIL_USER")
        pw = os.getenv("EMAIL_PASSWORD")
        host = config.get("capabilities", {}).get("email_imap_server", "imap.gmail.com")
        if not user or not pw:
            return _err("E_VALIDATION", "EMAIL_USER or EMAIL_PASSWORD missing")
        count = int(args.get("count", 5))
        with imaplib.IMAP4_SSL(host) as mail:
            mail.login(user, pw)
            mail.select("INBOX")
            _, ids = mail.search(None, "ALL")
            msg_ids = ids[0].split()[-count:]
            out = [m.decode("utf-8", errors="replace") for m in msg_ids]
        return _ok({"message_ids": out})

    if tool_name == "search_home_files":
        query = str(args["query"]).lower()
        home_root = Path(os.path.expanduser(os.getenv("ALPHANUS_HOME_ROOT", str(Path.home())))).resolve()
        directory = str(args.get("directory") or str(home_root))
        root = Path(os.path.expanduser(directory)).resolve()
        if not _is_under_root(root, home_root):
            return _err("E_POLICY", "Search directory outside home root")
        matches = []
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if query in name.lower():
                    matches.append(str(Path(dirpath) / name))
                    if len(matches) >= 200:
                        break
            if len(matches) >= 200:
                break
        return _ok({"matches": matches})

    if tool_name == "open_url":
        url = str(args["url"])
        if not (url.startswith("http://") or url.startswith("https://")):
            return _err("E_VALIDATION", "URL must start with http:// or https://")
        try:
            opened = webbrowser.open(url, new=2)
            if not opened:
                raise RuntimeError("No browser handler")
        except Exception:
            return _err("E_UNSUPPORTED", "Unable to open browser in this environment")
        return _ok({"url": url})

    if tool_name == "play_youtube":
        topic = str(args["topic"]).strip()
        url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(topic)
        try:
            opened = webbrowser.open(url, new=2)
            if not opened:
                raise RuntimeError("No browser handler")
        except Exception:
            return _err("E_UNSUPPORTED", "Unable to open browser in this environment")
        return _ok({"url": url, "topic": topic})

    return _err("E_UNSUPPORTED", f"Unsupported tool: {tool_name}")


def main() -> int:
    tool_name = sys.argv[1] if len(sys.argv) > 1 else os.getenv("ALPHANUS_TOOL_NAME", "").strip()
    if not tool_name:
        print(json.dumps(_err("E_VALIDATION", "Missing tool name"), ensure_ascii=False))
        return 2

    try:
        args = _read_args()
        config = _read_config()
        out = _run(tool_name, args, config)
    except ValueError as exc:
        out = _err("E_VALIDATION", str(exc))
    except FileNotFoundError as exc:
        out = _err("E_NOT_FOUND", str(exc))
    except PermissionError as exc:
        out = _err("E_POLICY", str(exc))
    except TimeoutError as exc:
        out = _err("E_TIMEOUT", str(exc))
    except Exception as exc:  # pragma: no cover - defensive safeguard
        out = _err("E_IO", str(exc))

    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
