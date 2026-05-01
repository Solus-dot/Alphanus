from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rich.markup import escape as esc

from core.attachments import classify_attachment
from core.configuration import load_global_config, normalize_config, validate_endpoint_policy
from core.runtime_config import UiRuntimeConfig
from tui.commands import COMMAND_ENTRIES
from tui.themes import available_theme_ids, fallback_color, theme_spec

DEFAULT_ACCENT_COLOR = fallback_color("accent")
DEFAULT_TEXT_COLOR = fallback_color("text")
DEFAULT_SUBTLE_COLOR = fallback_color("subtle")
DEFAULT_SUCCESS_COLOR = fallback_color("success")
DEFAULT_WARNING_COLOR = fallback_color("warning")
DEFAULT_MUTED_COLOR = fallback_color("muted")


def workspace_file_candidates(
    app: Any,
    *,
    max_items: int = 60,
    classify_attachment_fn: Any = None,
) -> list[Path]:
    classifier = classify_attachment_fn or classify_attachment
    root = app._workspace_root()
    if not root.exists() or not root.is_dir():
        return []
    files: list[Path] = []
    skip_dirs = {
        ".git",
        ".hg",
        ".svn",
        ".alphanus",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
    for current, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in skip_dirs and not d.startswith("."))
        for name in sorted(names):
            if name.startswith("."):
                continue
            candidate = (Path(current) / name).resolve()
            if not candidate.is_file():
                continue
            if classifier(str(candidate)) == "unknown":
                continue
            files.append(candidate)
            if len(files) >= max_items:
                return files
    return files


def build_global_palette_catalog(app: Any, *, command_palette_item_cls: Any) -> tuple[list[Any], dict[str, dict[str, str]]]:
    items: list[Any] = []
    actions: dict[str, dict[str, str]] = {}
    accent = app._theme_color("accent", DEFAULT_ACCENT_COLOR)
    text_color = app._theme_color("text", DEFAULT_TEXT_COLOR)
    subtle = app._theme_color("subtle", DEFAULT_SUBTLE_COLOR)
    success = app._theme_color("success", DEFAULT_SUCCESS_COLOR)
    warning = app._theme_color("warning", DEFAULT_WARNING_COLOR)
    muted = app._theme_color("muted", DEFAULT_MUTED_COLOR)

    def add_item(
        *,
        kind: str,
        value: str,
        prompt: str,
        search_text: str,
        rank: int,
    ) -> None:
        item_id = f"{kind}:{len(items)}"
        items.append(
            command_palette_item_cls(
                id=item_id,
                prompt=prompt,
                search_text=search_text,
                rank=rank,
            )
        )
        actions[item_id] = {"kind": kind, "value": value}

    for entry in COMMAND_ENTRIES:
        aliases = " ".join(entry.aliases)
        add_item(
            kind="command_insert",
            value=entry.insert_text,
            prompt=(
                f"[bold {accent}]cmd[/bold {accent}] [{text_color}]{esc(entry.prompt)}[/{text_color}] [dim]{esc(entry.description)}[/dim]"
            ),
            search_text=f"command {entry.prompt} {aliases} {entry.description}".strip(),
            rank=0,
        )

    for summary in app._session_store.list_sessions()[:20]:
        state = f"[bold {accent}]active[/bold {accent}]" if summary.is_active else f"[{subtle}]saved[/{subtle}]"
        add_item(
            kind="session_open",
            value=summary.id,
            prompt=(
                f"[bold {success}]session[/bold {success}] "
                f"{state} [{text_color}]{esc(summary.title)}[/{text_color}] "
                f"[dim]{summary.turn_count} turns[/dim]"
            ),
            search_text=f"session {summary.id} {summary.title} {summary.turn_count} turns",
            rank=1 if summary.is_active else 2,
        )

    workspace_root = app._workspace_root()
    for path in workspace_file_candidates(app, max_items=60):
        rel = app._root_relative_label(path, workspace_root)
        add_item(
            kind="file_attach",
            value=str(path),
            prompt=(f"[bold {warning}]file[/bold {warning}] [{text_color}]{esc(rel)}[/{text_color}]"),
            search_text=f"file {rel} {path.name}",
            rank=3,
        )

    loaded_ids = set(getattr(app, "_loaded_skill_ids", []))
    for skill in app.agent.skill_runtime.list_skills():
        loaded = skill.id in loaded_ids
        if not loaded and (not bool(getattr(skill, "enabled", False)) or not bool(getattr(skill, "available", False))):
            continue
        action = "unload" if loaded else "load"
        state_markup = f"[{success}]loaded[/{success}]" if loaded else f"[{muted}]available[/{muted}]"
        add_item(
            kind="skill_toggle",
            value=skill.id,
            prompt=(
                f"[bold {success}]skill[/bold {success}] [{text_color}]{esc(skill.id)}[/{text_color}] {state_markup} [dim]{action}[/dim]"
            ),
            search_text=f"skill {skill.id} {skill.name} {skill.description} {action}",
            rank=4 if loaded else 5,
        )

    return items, actions


def on_global_palette_close(app: Any, result: dict[str, str] | None, *, chat_input_cls: Any) -> None:
    app.query_one(chat_input_cls).focus()
    selected_id = str((result or {}).get("id") or "").strip()
    if not selected_id:
        return
    action = app._global_palette_actions.get(selected_id)
    if not action:
        return
    kind = str(action.get("kind") or "").strip()
    value = str(action.get("value") or "").strip()
    if not kind or not value:
        return
    if kind == "command_insert":
        chat_input = app.query_one(chat_input_cls)
        chat_input.value = value
        chat_input.cursor_position = len(chat_input.value)
        app._refresh_command_popup(chat_input.value)
        return
    if kind == "session_open":
        try:
            app._load_session_from_manager(value)
        except (ValueError, OSError, KeyError) as exc:
            app._write_error(f"Load failed: {exc}")
        return
    if kind == "file_attach":
        if app._attach_file_path(value):
            app._write_info(f"Attached file: {Path(value).name}")
        return
    if kind == "skill_toggle":
        if value in set(getattr(app, "_loaded_skill_ids", [])):
            if app._unload_skill_from_session(value):
                app._write_info(f"Unloaded skill {value}")
        else:
            if app._load_skill_into_session(value):
                app._write_info(f"Loaded skill {value}")


def open_theme_picker(app: Any, *, picker_item_cls: Any, selection_picker_modal_cls: Any) -> None:
    active = app._theme_id()
    accent = app._theme_color("accent", DEFAULT_ACCENT_COLOR)
    text_color = app._theme_color("text", DEFAULT_TEXT_COLOR)
    muted = app._theme_color("muted", DEFAULT_MUTED_COLOR)
    items = []
    for theme_id in available_theme_ids():
        spec = theme_spec(theme_id)
        marker = f"[bold {accent}]active[/bold {accent}]" if theme_id == active else f"[{muted}]available[/{muted}]"
        items.append(
            picker_item_cls(
                id=f"theme:{theme_id}",
                prompt=(
                    f"[bold {spec.colors.get('accent', accent)}]{esc(spec.title)}[/bold {spec.colors.get('accent', accent)}] "
                    f"{marker} [{text_color}]{esc(spec.description)}[/{text_color}]"
                ),
            )
        )
    app.push_screen(
        selection_picker_modal_cls(
            kicker="THEME",
            title="Choose Theme",
            subtitle=f"Current theme: {active}",
            list_label="Available Themes",
            confirm_label="Apply / Save",
            empty_text="No themes available.",
            items=items,
        ),
        app._on_theme_picker_close,
    )


def persist_theme_preference(app: Any, theme_id: str, *, config_path: Path) -> list[str]:
    warnings: list[str] = []
    try:
        current = load_global_config(config_path, warnings=warnings)
    except (OSError, ValueError) as exc:
        app._write_error(f"Theme persisted only for this run: {exc}")
        return warnings

    updates = app._merge_live_config(current, {"tui": {"theme": theme_id}})
    try:
        normalized, normalize_warnings = normalize_config(updates)
        warnings.extend(normalize_warnings)
        validate_endpoint_policy(normalized)
    except ValueError as exc:
        app._write_error(f"Theme persisted only for this run: {exc}")
        return warnings

    cleaned = app._config_for_editor(normalized)
    try:
        config_path.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        app._write_error(f"Theme persisted only for this run: {exc}")
        return warnings

    merged = app._merge_live_config(app.agent.config, normalized)
    app.agent.reload_config(merged)
    app._ui_config = UiRuntimeConfig.from_config(merged)
    app._ui_timing = app._ui_config.timing
    app._apply_tui_config()
    return warnings


def on_theme_picker_close(app: Any, result: dict[str, str] | None, *, chat_input_cls: Any, config_path: Path) -> None:
    selected_id = str((result or {}).get("id") or "").strip()
    if not selected_id.startswith("theme:"):
        app.query_one(chat_input_cls).focus()
        return
    selected_theme = selected_id.split(":", 1)[1]
    resolved = app._apply_theme(selected_theme)
    persist = getattr(app, "_persist_theme_preference", None)
    if callable(persist):
        persist_warnings = persist(resolved)
    else:
        persist_warnings = persist_theme_preference(app, resolved, config_path=config_path)
    app._write_command_action(f"Theme set to '{resolved}'", icon="◉")
    warning_rows = persist_warnings if isinstance(persist_warnings, list) else []
    for warning in warning_rows:
        app._write_info(f"Config warning: {warning}")
    app.query_one(chat_input_cls).focus()
