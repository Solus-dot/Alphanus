from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rich.markup import escape as esc

from tui.themes import fallback_color

_DEFAULT_COLORS = {
    "accent": fallback_color("accent"),
    "text": fallback_color("text"),
    "muted": fallback_color("muted"),
    "subtle": fallback_color("subtle"),
    "success": fallback_color("success"),
    "error": fallback_color("error"),
}


@dataclass(slots=True)
class ActivityRow:
    name: str
    detail: str = ""
    status: str = "running"
    message: str = ""
    started_at: float = field(default_factory=time.monotonic)
    duration_ms: int | None = None


@dataclass(slots=True)
class ActivityState:
    phase: str = "idle"
    started_at: float = 0.0
    rows: list[ActivityRow] = field(default_factory=list)
    last_error: str = ""
    interrupted: bool = False

    def reset(self) -> None:
        self.phase = "starting"
        self.started_at = time.monotonic()
        self.rows = []
        self.last_error = ""
        self.interrupted = False

    def start_tool(self, name: str, detail: str = "") -> None:
        self.phase = "tools"
        normalized = str(name or "tool").strip() or "tool"
        for row in reversed(self.rows):
            if row.name == normalized and row.status == "running":
                row.detail = detail or row.detail
                return
        self.rows.append(ActivityRow(name=normalized, detail=str(detail or "")))

    def finish_tool(self, name: str, *, ok: bool, message: str = "", duration_ms: int | None = None) -> None:
        normalized = str(name or "tool").strip() or "tool"
        row = next((item for item in reversed(self.rows) if item.name == normalized and item.status == "running"), None)
        if row is None:
            row = ActivityRow(name=normalized)
            self.rows.append(row)
        row.status = "done" if ok else "failed"
        row.message = str(message or "")
        if duration_ms is None:
            row.duration_ms = max(0, int((time.monotonic() - row.started_at) * 1000))
        else:
            row.duration_ms = max(0, int(duration_ms))
        if not ok:
            self.phase = "failed"
            self.last_error = row.message

    def note_error(self, message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        self.phase = "failed"
        self.last_error = text

    def mark_interrupted(self) -> None:
        self.phase = "interrupted"
        self.interrupted = True


def _theme_colors(colors: dict[str, str] | None = None) -> dict[str, str]:
    merged = dict(_DEFAULT_COLORS)
    if isinstance(colors, dict):
        merged.update({key: str(value) for key, value in colors.items() if value})
    return merged


def _truncate(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len == 1:
        return "…"
    return text[: max_len - 1] + "…"


def compact_tool_result_message(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    error = result.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error.get("message") or "")
    data = result.get("data")
    if isinstance(data, dict):
        if data.get("returncode") not in (None, 0):
            return f"exit {data.get('returncode')}"
        if data.get("passed") is True:
            return "passed"
        if data.get("passed") is False:
            return "failed"
    return ""


def tool_result_duration_ms(result: Any) -> int | None:
    if not isinstance(result, dict):
        return None
    meta = result.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("duration_ms"), int):
        return int(meta["duration_ms"])
    data = result.get("data")
    if isinstance(data, dict) and isinstance(data.get("duration_ms"), int):
        return int(data["duration_ms"])
    return None


def render_activity_markup(state: ActivityState, *, width: int = 30, colors: dict[str, str] | None = None) -> str:
    theme = _theme_colors(colors)
    width = max(1, int(width))
    elapsed = int(time.monotonic() - state.started_at) if state.started_at > 0 else 0
    phase = "interrupted" if state.interrupted else state.phase
    state_label = _truncate(f"state: {phase}  {elapsed}s", width)
    lines = [
        f"[bold {theme['accent']}]Turn activity[/bold {theme['accent']}]",
        f"[{theme['muted']}]{esc(state_label)}[/{theme['muted']}]",
    ]
    if not state.rows:
        lines.append(f"[{theme['muted']}]no tool activity yet[/{theme['muted']}]")
    for row in state.rows[-8:]:
        status_color = theme["success"] if row.status == "done" else theme["error"] if row.status == "failed" else theme["accent"]
        label = {"running": "running", "done": "done", "failed": "failed"}.get(row.status, row.status)
        duration = "" if row.duration_ms is None else f" {row.duration_ms}ms"
        name_width = max(0, width - len(label) - len(duration) - 1)
        name = _truncate(row.name, name_width)
        detail = _truncate(row.detail, width)
        lines.append(
            f"[{status_color}]{label}[/] [{theme['text']}]{esc(name)}[/{theme['text']}]"
            f"[{theme['subtle']}]{duration}[/{theme['subtle']}]"
        )
        if detail:
            lines.append(f"[{theme['subtle']}]{esc(detail)}[/{theme['subtle']}]")
        if row.status == "failed" and row.message:
            msg = _truncate(row.message, width)
            lines.append(f"[{theme['error']}]{esc(msg)}[/{theme['error']}]")
    if state.last_error and not any(row.status == "failed" for row in state.rows[-3:]):
        lines.append(f"[{theme['error']}]{esc(_truncate(state.last_error, width))}[/{theme['error']}]")
    return "\n".join(lines)
