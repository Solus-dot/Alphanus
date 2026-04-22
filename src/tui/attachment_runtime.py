from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.markup import escape as esc

from core.attachments import classify_attachment
from tui.popups import PickerItem, SelectionPickerModal
from tui.themes import fallback_color

DEFAULT_ACCENT_COLOR = fallback_color("accent")
DEFAULT_TEXT_COLOR = fallback_color("text")
DEFAULT_MUTED_COLOR = fallback_color("muted")


def workspace_root(app: Any) -> Path:
    return Path(str(app.agent.skill_runtime.workspace.workspace_root)).resolve()


def home_root(app: Any) -> Path:
    candidate = getattr(app.agent.skill_runtime.workspace, "home_root", None)
    if candidate:
        return Path(str(candidate)).resolve()
    return Path.home().resolve()


def attachment_root_path(app: Any, root_id: str) -> Path:
    if root_id == "home":
        return home_root(app)
    return workspace_root(app)


def attachment_root_label(root_id: str) -> str:
    return "Home" if root_id == "home" else "Workspace"


def root_relative_label(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root)
        text = relative.as_posix()
        return text if text else "."
    except ValueError:
        return path.as_posix()


def resolve_attachment_path(app: Any, raw_path: str) -> Path:
    candidate = Path(os.path.expanduser(raw_path.strip()))
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
        raise FileNotFoundError(str(resolved))

    workspace_candidate = (workspace_root(app) / candidate).resolve()
    cwd_candidate = (Path.cwd() / candidate).resolve()
    for resolved in (workspace_candidate, cwd_candidate):
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(str(workspace_candidate))


def attach_file_path(app: Any, path: str | Path) -> bool:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        app._write_error(f"File not found: {resolved}")
        return False
    kind = classify_attachment(str(resolved))
    if kind == "unknown":
        app._write_error("Unsupported file type")
        return False
    normalized = str(resolved)
    if any(existing_path == normalized for existing_path, _existing_kind in app.pending):
        app._write_info(f"Already attached: {resolved.name}")
        return False
    app.pending.append((normalized, kind))
    app._update_pending_attachments()
    app._update_status1()
    return True


def attachment_picker_items(
    app: Any,
    relative_dir: str = ".",
    *,
    root_id: str = "workspace",
    accent_color: str = DEFAULT_ACCENT_COLOR,
) -> List[PickerItem]:
    text_color = app._theme_color("text", DEFAULT_TEXT_COLOR) if hasattr(app, "_theme_color") else DEFAULT_TEXT_COLOR
    muted = app._theme_color("muted", DEFAULT_MUTED_COLOR) if hasattr(app, "_theme_color") else DEFAULT_MUTED_COLOR
    items: List[PickerItem] = []
    current = Path(relative_dir)
    for candidate_root in ("workspace", "home"):
        if candidate_root == root_id:
            continue
        items.append(
            PickerItem(
                id=f"root:{candidate_root}:.",
                prompt=f"[bold {accent_color}]switch -> {attachment_root_label(candidate_root).lower()}[/bold {accent_color}]",
            )
        )
    if relative_dir not in {".", ""}:
        parent = current.parent.as_posix()
        if parent == "":
            parent = "."
        items.append(PickerItem(id=f"nav:{parent}", prompt=f"[dim]{esc('../')}[/dim] [{muted}]parent[/{muted}]"))

    root_path = attachment_root_path(app, root_id)
    list_target = str(root_path if relative_dir in {".", ""} else (root_path / relative_dir))
    entries = app.agent.skill_runtime.workspace.list_files(list_target)
    for entry in entries:
        target = (current / entry.rstrip("/")).as_posix()
        if entry.endswith("/"):
            items.append(
                PickerItem(
                    id=f"nav:{target}",
                    prompt=f"[bold {accent_color}]{esc(entry)}[/bold {accent_color}] [dim]open[/dim]",
                )
            )
            continue
        target_path = root_path / target
        kind = classify_attachment(str(target_path))
        if kind == "unknown":
            continue
        items.append(
            PickerItem(
                id=f"file:{target}",
                prompt=f"[{text_color}]{esc(entry)}[/{text_color}] [dim]{kind}[/dim]",
            )
        )
    return items


def open_attachment_picker(
    app: Any,
    relative_dir: str = ".",
    root_id: str = "workspace",
    accent_color: str = DEFAULT_ACCENT_COLOR,
) -> None:
    clean_dir = relative_dir or "."
    root_path = attachment_root_path(app, root_id)
    title = f"Attach File · {attachment_root_label(root_id)}"
    current_dir = root_relative_label(root_path / clean_dir, root_path)
    if root_id == "home":
        if current_dir in {".", ""}:
            current_dir = "~/"
        else:
            current_dir = f"~/{current_dir}"
    else:
        if current_dir in {".", ""}:
            current_dir = "workspace root"
        else:
            current_dir = f"./{current_dir}"
    subtitle = f"Current directory: {current_dir}"
    items = attachment_picker_items(app, clean_dir, root_id=root_id, accent_color=accent_color)
    app.push_screen(
        SelectionPickerModal(
            title=title,
            subtitle=subtitle,
            confirm_label="Open / Attach",
            empty_text="No attachable files in this folder.",
            items=items,
        ),
        lambda result: on_attachment_picker_close(app, root_id, clean_dir, result, accent_color=accent_color),
    )


def on_attachment_picker_close(
    app: Any,
    root_id: str,
    current_dir: str,
    result: Optional[Dict[str, str]],
    *,
    accent_color: str = DEFAULT_ACCENT_COLOR,
) -> None:
    _ = current_dir
    selection = str((result or {}).get("id") or "").strip()
    if not selection:
        app.query_one("#chat-input").focus()
        return
    if selection.startswith("root:"):
        _, next_root, next_dir = selection.split(":", 2)
        open_attachment_picker(app, next_dir or ".", root_id=next_root or "workspace", accent_color=accent_color)
        return
    if selection.startswith("nav:"):
        open_attachment_picker(app, selection[4:] or ".", root_id=root_id, accent_color=accent_color)
        return
    if selection.startswith("file:"):
        target = selection[5:]
        attach_file_path(app, attachment_root_path(app, root_id) / target)
    app.query_one("#chat-input").focus()
