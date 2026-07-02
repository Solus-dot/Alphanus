from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from rich.markup import escape as esc

from agent.file_audit import build_file_audit_from_skill_exchanges
from core.conv_tree import Turn
from core.message_types import ChatMessage, JSONValue


def audit_rows_for_turn(app: Any, turn: Turn) -> list[dict[str, JSONValue]]:
    project_root: str | Path | None = None
    project_root_fn = getattr(app, "_project_root", None)
    if callable(project_root_fn):
        try:
            value = project_root_fn()
            if isinstance(value, str | Path):
                project_root = value
        except Exception:
            project_root = None
    skill_exchanges = getattr(turn, "skill_exchanges", [])
    return build_file_audit_from_skill_exchanges(
        cast(list[ChatMessage], skill_exchanges if isinstance(skill_exchanges, list) else []),
        project_root=project_root,
    )


def _format_bytes(value: object) -> str:
    if not isinstance(value, int) or isinstance(value, bool):
        return ""
    if value < 1024:
        return f"{value} B"
    if value < 1024 * 1024:
        return f"{value / 1024:.1f} KB"
    return f"{value / (1024 * 1024):.1f} MB"


def _row_summary(row: dict[str, JSONValue]) -> str:
    action = str(row.get("action") or "")
    status = str(row.get("status") or "success")
    failed = status == "failed"
    prefix = "failed " if failed else ""
    if action == "created":
        path = str(row.get("path") or "")
        kind = str(row.get("kind") or "path")
        details = []
        if isinstance(row.get("lines"), int):
            details.append(f"{row['lines']} lines")
        if size := _format_bytes(row.get("bytes")):
            details.append(size)
        suffix = f"  {', '.join(details)}" if details else ""
        return f"{prefix}created  {path}  {kind}{suffix}".strip()
    if action == "edited":
        path = str(row.get("path") or "")
        changed = row.get("changed_lines")
        detail = f"  {changed} changed lines" if isinstance(changed, int) else ""
        return f"{prefix}edited   {path}{detail}".strip()
    if action == "moved":
        source = str(row.get("from") or "")
        target = str(row.get("to") or "")
        return f"{prefix}moved    {source} -> {target}".strip()
    if action == "deleted":
        path = str(row.get("path") or "")
        kind = str(row.get("kind") or "path")
        details = [kind]
        if isinstance(row.get("file_count"), int):
            details.append(f"{row['file_count']} files")
        elif size := _format_bytes(row.get("bytes")):
            details.append(size)
        return f"{prefix}deleted  {path}  {', '.join(details)}".strip()
    if action == "project_changed":
        command = str(row.get("command") or "shell command")
        return f"{prefix}shell    {command}  project changed, paths unknown".strip()
    return f"{prefix}{action or 'changed'}"


def write_file_audit(app: Any, rows: list[dict[str, JSONValue]], *, empty_message: str = "") -> None:
    if not rows:
        if empty_message:
            app._write_info(empty_message)
        return
    app._write_section_heading("File Changes")
    for row in rows:
        failed = str(row.get("status") or "success") == "failed"
        color = app._theme_color("error", "#ef4444") if failed else app._theme_color("success", "#22c55e")
        icon = "x" if failed else "+"
        app._write_assistant_bar_line(f"[{color}]{icon}[/{color}] {esc(_row_summary(row))}", content_indent=2)
    app._write("")
