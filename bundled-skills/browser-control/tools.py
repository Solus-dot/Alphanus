from __future__ import annotations

import platform
import subprocess
import urllib.parse
import webbrowser
from typing import Any

from skills.runtime import ToolExecutionEnv

_MAC_BROWSER_APPS = {
    "safari": "Safari",
    "google chrome": "Google Chrome",
    "chrome": "Google Chrome",
    "chromium": "Chromium",
    "brave browser": "Brave Browser",
    "brave": "Brave Browser",
    "microsoft edge": "Microsoft Edge",
    "edge": "Microsoft Edge",
}
_DEFAULT_MAC_BROWSER_APPS = ["Safari", "Google Chrome", "Chromium", "Brave Browser", "Microsoft Edge"]

TOOL_SPECS = {
    "open_browser_url": {
        "capability": "browser_open",
        "mutates": True,
        "actions": ["open"],
        "description": "Open a URL in the default browser. Requires confirm_open=true.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}, "confirm_open": {"type": "boolean"}},
            "required": ["url"],
        },
    },
    "browser_search": {
        "capability": "browser_open",
        "mutates": True,
        "actions": ["open", "read"],
        "description": "Open a browser search. Requires confirm_open=true.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "engine": {"type": "string"}, "confirm_open": {"type": "boolean"}},
            "required": ["query"],
        },
    },
    "get_current_browser_page": {
        "capability": "browser_read",
        "mutates": False,
        "actions": ["read", "check"],
        "description": "Return best-effort current browser URL/title/text where platform support exists.",
        "parameters": {
            "type": "object",
            "properties": {"browser": {"type": "string"}},
            "required": [],
        },
    },
}


def _ok(data: dict[str, object]) -> dict[str, object]:
    return {"ok": True, "data": data, "error": None, "meta": {}}


def _err(code: str, message: str, data: dict[str, object] | None = None) -> dict[str, object]:
    return {"ok": False, "data": data or {}, "error": {"code": code, "message": message}, "meta": {}}


def _confirm(args: dict[str, object], data: dict[str, object]) -> dict[str, object] | None:
    if bool(args.get("confirm_open")):
        return None
    return _err("E_CONFIRMATION_REQUIRED", "Opening a browser page requires confirm_open=true", data)


def _mac_browser_candidates(browser: str) -> list[str]:
    if not browser:
        return list(_DEFAULT_MAC_BROWSER_APPS)
    app_name = _MAC_BROWSER_APPS.get(browser.strip().lower())
    if app_name is None:
        raise ValueError("browser must be Safari, Chrome, Chromium, Brave, or Edge")
    return [app_name]


def _open_url(args: dict[str, object]) -> dict[str, object]:
    url = str(args.get("url") or "").strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https", "file"}:
        raise ValueError("URL must start with http://, https://, or file://")
    if blocked := _confirm(args, {"url": url}):
        return blocked
    try:
        opened = webbrowser.open(url, new=2)
    except Exception as exc:
        return _err("E_IO", f"Browser launch failed: {exc}", {"url": url})
    if not opened:
        return _err("E_IO", "Unable to open browser in this environment", {"url": url})
    return _ok({"url": url})


def _search(args: dict[str, object]) -> dict[str, object]:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    engine = str(args.get("engine") or "google").strip().lower()
    base = {
        "google": "https://www.google.com/search?q=",
        "duckduckgo": "https://duckduckgo.com/?q=",
        "bing": "https://www.bing.com/search?q=",
    }.get(engine)
    if base is None:
        raise ValueError("engine must be google, duckduckgo, or bing")
    url = base + urllib.parse.quote_plus(query)
    opened = _open_url({"url": url, "confirm_open": args.get("confirm_open")})
    if opened.get("ok"):
        opened["data"] = {"url": url, "query": query, "engine": engine}
    return opened


def _current_page(args: dict[str, object]) -> dict[str, object]:
    system = platform.system().lower()
    browser = str(args.get("browser") or "").strip()
    if system == "darwin":
        # Text extraction is intentionally omitted; URL/title is the useful common denominator.
        candidates = _mac_browser_candidates(browser)

        for app_name in candidates:
            if app_name == "Safari":
                script = (
                    'tell application "Safari"\n'
                    "if it is running then\n"
                    "set pageUrl to URL of current tab of front window\n"
                    "set pageTitle to name of front document\n"
                    "return pageTitle & linefeed & pageUrl\n"
                    "end if\n"
                    "end tell"
                )
            else:
                script = (
                    f'tell application "{app_name}"\n'
                    "if it is running then\n"
                    "set pageUrl to URL of active tab of front window\n"
                    "set pageTitle to title of active tab of front window\n"
                    "return pageTitle & linefeed & pageUrl\n"
                    "end if\n"
                    "end tell"
                )
            proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=5)
            if proc.returncode == 0 and proc.stdout.strip():
                title, _, url = proc.stdout.partition("\n")
                return _ok({"platform": system, "browser": app_name, "title": title.strip(), "url": url.strip(), "text": ""})
    return _err("E_UNSUPPORTED", "Current browser page inspection is not supported on this platform/browser", {"platform": system, "browser": browser})


def execute(tool_name: str, args: dict[str, object], _env: ToolExecutionEnv) -> dict[str, Any]:
    if tool_name == "open_browser_url":
        return _open_url(args)
    if tool_name == "browser_search":
        return _search(args)
    if tool_name == "get_current_browser_page":
        return _current_page(args)
    raise ValueError(f"Unsupported tool: {tool_name}")
