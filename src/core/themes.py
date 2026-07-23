from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from alphanus.paths import APP_ROOT_ENV_VAR, DEFAULT_APP_DIRNAME
from core.theme_catalog import BUILTIN_THEME_IDS, DEFAULT_THEME_ID, normalize_theme_id

_THEME_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_BORDER_SETS = {"plain", "rounded", "double", "thick"}
_STYLE_KEYS = {"base", "text", "muted", "subtle", "accent", "success", "warning", "error", "border", "selection", "user", "assistant"}
_STYLE_MODIFIERS = {"bold", "dim", "italic", "underlined", "reversed", "crossed_out"}


class ThemeLoadError(ValueError):
    pass


def _user_theme_dirs() -> list[Path]:
    configured = [Path(item).expanduser().resolve() for item in os.environ.get("ALPHANUS_THEME_PATHS", "").split(os.pathsep) if item]
    override = os.environ.get(APP_ROOT_ENV_VAR, "").strip()
    root = Path(override).expanduser().resolve() if override else (Path.home() / DEFAULT_APP_DIRNAME).resolve()
    default = root / "themes"
    return configured if default in configured else [*configured, default]


def _validate(payload: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ThemeLoadError(f"{source}: theme must be an object")
    theme_id = str(payload.get("id") or "").strip().lower()
    if not _THEME_ID_RE.fullmatch(theme_id):
        raise ThemeLoadError(f"{source}: invalid theme id")
    if not str(payload.get("title") or "").strip():
        raise ThemeLoadError(f"{source}: title must not be empty")
    colors = payload.get("colors")
    theme = payload.get("theme")
    if not isinstance(colors, dict) or not isinstance(theme, dict):
        raise ThemeLoadError(f"{source}: colors and theme must be objects")
    ratatui = payload.get("ratatui")
    warnings: list[str] = []
    if ratatui is not None and not isinstance(ratatui, dict):
        warnings.append("ratatui override ignored because it is not an object")
        payload = dict(payload)
        payload.pop("ratatui", None)
    elif isinstance(ratatui, dict):
        normalized: dict[str, Any] = {}
        border = str(ratatui.get("border_set") or "").strip().lower()
        if border:
            if border in _BORDER_SETS:
                normalized["border_set"] = border
            else:
                warnings.append(f"invalid ratatui.border_set {border!r}; using rounded")
        syntax = str(ratatui.get("syntax_theme") or "").strip()
        if syntax:
            normalized["syntax_theme"] = syntax
        styles = ratatui.get("styles")
        if styles is not None:
            if isinstance(styles, dict):
                normalized_styles: dict[str, Any] = {}
                for raw_key, raw_style in styles.items():
                    key = str(raw_key).strip().lower()
                    if key not in _STYLE_KEYS or not isinstance(raw_style, dict):
                        warnings.append(f"unknown or invalid ratatui style {key!r}; ignoring")
                        continue
                    style: dict[str, Any] = {}
                    for color_key in ("foreground", "background"):
                        raw_color = str(raw_style.get(color_key) or "").strip()
                        if raw_color:
                            if re.fullmatch(r"#[0-9a-fA-F]{6}", raw_color):
                                style[color_key] = raw_color
                            else:
                                warnings.append(f"invalid ratatui.styles.{key}.{color_key}; ignoring")
                    modifiers = raw_style.get("modifiers", [])
                    if isinstance(modifiers, list):
                        accepted = [str(item).strip().lower() for item in modifiers if str(item).strip().lower() in _STYLE_MODIFIERS]
                        if accepted:
                            style["modifiers"] = accepted
                    if style:
                        normalized_styles[key] = style
                normalized["styles"] = normalized_styles
            else:
                warnings.append("ratatui.styles ignored because it is not an object")
        payload = dict(payload)
        payload["ratatui"] = normalized
    payload = dict(payload)
    payload["id"] = theme_id
    if warnings:
        payload["_warnings"] = warnings
    return payload


def _read(text: str, *, source: str) -> dict[str, Any]:
    try:
        return _validate(json.loads(text), source=source)
    except json.JSONDecodeError as exc:
        raise ThemeLoadError(f"{source}: invalid JSON: {exc}") from exc


@lru_cache(maxsize=1)
def theme_payloads() -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    root = resources.files("core") / "theme_specs"
    for theme_id in BUILTIN_THEME_IDS:
        payload = _read((root / f"{theme_id}.json").read_text(encoding="utf-8"), source=f"builtin:{theme_id}")
        payloads[theme_id] = payload
    for directory in _user_theme_dirs():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                payload = _read(path.read_text(encoding="utf-8"), source=str(path))
            except (OSError, ThemeLoadError):
                continue
            payloads[payload["id"]] = payload
    return payloads


def reload_themes() -> None:
    theme_payloads.cache_clear()


def available_theme_ids() -> list[str]:
    payloads = theme_payloads()
    builtins = [theme_id for theme_id in BUILTIN_THEME_IDS if theme_id in payloads]
    return [*builtins, *sorted(theme_id for theme_id in payloads if theme_id not in BUILTIN_THEME_IDS)]


def theme_payload(theme_id: str) -> dict[str, Any]:
    payloads = theme_payloads()
    resolved, _ = normalize_theme_id(theme_id, default=DEFAULT_THEME_ID, available=payloads)
    return dict(payloads[resolved])
