from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from typing import Any

from skills.runtime import ToolExecutionEnv

TOOL_SPECS = {
    "list_apps": {
        "capability": "desktop_read",
        "mutates": False,
        "actions": ["list", "read"],
        "description": "List running desktop applications where the platform supports it.",
        "parameters": {
            "type": "object",
            "properties": {"include_windows": {"type": "boolean"}},
            "required": [],
        },
    },
    "open_app": {
        "capability": "desktop_control",
        "mutates": True,
        "actions": ["open"],
        "description": "Open a desktop application. Requires confirm_open=true.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "confirm_open": {"type": "boolean"}},
            "required": ["name"],
        },
    },
    "focus_app": {
        "capability": "desktop_control",
        "mutates": True,
        "actions": ["open"],
        "description": "Focus a running desktop application. Requires confirm_focus=true.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "window_title": {"type": "string"}, "confirm_focus": {"type": "boolean"}},
            "required": ["name"],
        },
    },
    "quit_app": {
        "capability": "desktop_control",
        "mutates": True,
        "actions": ["delete"],
        "description": "Quit a desktop application. Requires confirm_quit=true.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "force": {"type": "boolean"}, "confirm_quit": {"type": "boolean"}},
            "required": ["name"],
        },
    },
}


def _ok(data: dict[str, object]) -> dict[str, object]:
    return {"ok": True, "data": data, "error": None, "meta": {}}


def _err(code: str, message: str, data: dict[str, object] | None = None) -> dict[str, object]:
    return {"ok": False, "data": data or {}, "error": {"code": code, "message": message}, "meta": {}}


def _require(args: dict[str, object], flag: str, action: str, data: dict[str, object]) -> dict[str, object] | None:
    if bool(args.get(flag)):
        return None
    return _err("E_CONFIRMATION_REQUIRED", f"{action} requires {flag}=true", data)


def _run(cmd: list[str], *, timeout: int = 10, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)


def _system() -> str:
    return platform.system().lower()


_LSAPPINFO_RECORD_RE = re.compile(r'^\s*\d+\) "([^"]+)".*?(?=^\s*\d+\) |\Z)', re.MULTILINE | re.DOTALL)


def _macos_foreground_apps(output: str) -> list[str]:
    return sorted(dict.fromkeys(match.group(1) for match in _LSAPPINFO_RECORD_RE.finditer(output) if 'type="Foreground"' in match.group(0)))


def _applescript_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\r", " ").replace("\n", " ")
    return f'"{escaped}"'


def _list_apps(args: dict[str, object]) -> dict[str, object]:
    include_windows = bool(args.get("include_windows"))
    system = _system()
    if system == "darwin":
        # lsappinfo reads LaunchServices state without requiring the macOS
        # Automation permission that makes System Events hang in headless and
        # first-run environments.
        proc = _run(["lsappinfo", "list"], timeout=5)
        if proc.returncode != 0:
            return _err("E_UNSUPPORTED", proc.stderr.strip() or "Unable to list macOS applications", {"platform": system})
        return _ok({"platform": system, "apps": _macos_foreground_apps(proc.stdout), "windows": []})
    if system == "linux":
        if shutil.which("wmctrl") is None:
            return _err("E_DEPENDENCY", "wmctrl is required to list graphical applications on Linux", {"platform": system})
        proc = _run(["wmctrl", "-lx"])
        if proc.returncode != 0:
            return _err("E_IO", proc.stderr.strip() or "wmctrl failed", {"platform": system})
        windows = []
        apps = []
        for line in proc.stdout.splitlines():
            parts = line.split(None, 4)
            if len(parts) >= 4:
                app = parts[2]
                title = parts[4] if len(parts) > 4 else ""
                apps.append(app)
                if include_windows:
                    windows.append({"app": app, "title": title, "window_id": parts[0]})
        return _ok({"platform": system, "apps": sorted(dict.fromkeys(apps)), "windows": windows})
    if system == "windows":
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-Process | Where-Object {$_.MainWindowTitle} | Select-Object ProcessName,MainWindowTitle | ConvertTo-Json",
        ]
        proc = _run(cmd)
        if proc.returncode != 0:
            return _err("E_IO", proc.stderr.strip() or "PowerShell process listing failed", {"platform": system})
        return _ok({"platform": system, "apps_text": proc.stdout.strip(), "windows": []})
    return _err("E_UNSUPPORTED", f"Unsupported platform: {system}", {"platform": system})


def _open_app(args: dict[str, object]) -> dict[str, object]:
    name = str(args.get("name") or "").strip()
    if not name:
        raise ValueError("name is required")
    if blocked := _require(args, "confirm_open", "Opening an application", {"name": name}):
        return blocked
    system = _system()
    if system == "darwin":
        proc = _run(["open", "-a", name])
    elif system == "linux":
        executable = shutil.which(name)
        if executable is None:
            return _err("E_NOT_FOUND", f"Application not found on PATH: {name}", {"name": name})
        try:
            subprocess.Popen([executable], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            return _err("E_IO", f"Failed to open {name}: {exc}", {"platform": system, "name": name})
        return _ok({"platform": system, "name": name})
    elif system == "windows":
        process_env = os.environ.copy()
        process_env["ALPHANUS_APP_CONTROL_NAME"] = name
        proc = _run(["powershell", "-NoProfile", "-Command", "Start-Process -FilePath $env:ALPHANUS_APP_CONTROL_NAME"], env=process_env)
    else:
        return _err("E_UNSUPPORTED", f"Unsupported platform: {system}", {"platform": system, "name": name})
    if proc.returncode != 0:
        return _err("E_IO", proc.stderr.strip() or f"Failed to open {name}", {"platform": system, "name": name})
    return _ok({"platform": system, "name": name})


def _focus_app(args: dict[str, object]) -> dict[str, object]:
    name = str(args.get("name") or "").strip()
    if not name:
        raise ValueError("name is required")
    if blocked := _require(args, "confirm_focus", "Focusing an application", {"name": name}):
        return blocked
    system = _system()
    if system == "darwin":
        # `open -a` activates an existing app and launches it only if needed;
        # unlike AppleScript it does not depend on Automation permission.
        proc = _run(["open", "-a", name])
    elif system == "linux":
        if shutil.which("wmctrl") is None:
            return _err("E_DEPENDENCY", "wmctrl is required to focus applications on Linux", {"platform": system})
        target = str(args.get("window_title") or name)
        proc = _run(["wmctrl", "-a", target])
    elif system == "windows":
        return _err("E_UNSUPPORTED", "Focusing windows on Windows is not supported by app-control", {"platform": system, "name": name})
    else:
        return _err("E_UNSUPPORTED", f"Unsupported platform: {system}", {"platform": system, "name": name})
    if proc.returncode != 0:
        return _err("E_IO", proc.stderr.strip() or f"Failed to focus {name}", {"platform": system, "name": name})
    return _ok({"platform": system, "name": name})


def _quit_app(args: dict[str, object]) -> dict[str, object]:
    name = str(args.get("name") or "").strip()
    if not name:
        raise ValueError("name is required")
    if blocked := _require(args, "confirm_quit", "Quitting an application", {"name": name, "force": bool(args.get("force"))}):
        return blocked
    system = _system()
    force = bool(args.get("force"))
    if system == "darwin":
        verb = "quit" if not force else "quit saving no"
        proc = _run(["osascript", "-e", f"tell application {_applescript_string(name)} to {verb}"])
    elif system == "linux":
        sig = "-9" if force else "-TERM"
        proc = _run(["pkill", sig, "-x", name])
    elif system == "windows":
        proc = _run(["taskkill", "/IM", f"{name}.exe", *(["/F"] if force else [])])
    else:
        return _err("E_UNSUPPORTED", f"Unsupported platform: {system}", {"platform": system, "name": name})
    if proc.returncode != 0:
        return _err("E_IO", proc.stderr.strip() or f"Failed to quit {name}", {"platform": system, "name": name})
    return _ok({"platform": system, "name": name, "force": force})


def execute(tool_name: str, args: dict[str, object], _env: ToolExecutionEnv) -> dict[str, Any]:
    if tool_name == "list_apps":
        return _list_apps(args)
    if tool_name == "open_app":
        return _open_app(args)
    if tool_name == "focus_app":
        return _focus_app(args)
    if tool_name == "quit_app":
        return _quit_app(args)
    raise ValueError(f"Unsupported tool: {tool_name}")
