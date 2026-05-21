from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Final

from textual.theme import Theme

from alphanus_paths import APP_ROOT_ENV_VAR, DEFAULT_APP_DIRNAME
from core.coercion import coerce_bool
from core.theme_catalog import BUILTIN_THEME_IDS, DEFAULT_THEME_ID, normalize_theme_id

_THEME_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_BUILTIN_PACKAGE = "tui.theme_specs"
_USER_THEME_PATHS_ENV = "ALPHANUS_THEME_PATHS"


@dataclass(frozen=True, slots=True)
class ThemeSpec:
    id: str
    title: str
    description: str
    theme: Theme
    syntax_theme: str
    text_area_theme: str
    colors: dict[str, str]


class ThemeLoadError(ValueError):
    pass


def _coerce_mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ThemeLoadError(f"{path} must be an object")
    return value


def _coerce_str(value: Any, *, path: str, default: str = "") -> str:
    text = str(default if value is None else value).strip()
    if not text:
        raise ThemeLoadError(f"{path} must not be empty")
    return text


def _coerce_optional_str(value: Any, default: str = "") -> str:
    return str(default if value is None else value).strip()


def _coerce_bool(value: Any, default: bool = True) -> bool:
    return coerce_bool(value, default)


def _coerce_string_dict(value: Any, *, path: str, required: bool = False) -> dict[str, str]:
    if value is None:
        if required:
            raise ThemeLoadError(f"{path} must be an object")
        return {}
    raw = _coerce_mapping(value, path=path)
    return {str(key).strip(): str(item).strip() for key, item in raw.items() if str(key).strip() and str(item).strip()}


def _theme_from_payload(payload: dict[str, Any], *, source: str) -> ThemeSpec:
    theme_id = _coerce_str(payload.get("id"), path=f"{source}: id").lower()
    if not _THEME_ID_RE.match(theme_id):
        raise ThemeLoadError(f"{source}: invalid theme id {theme_id!r}")
    title = _coerce_str(payload.get("title"), path=f"{source}: title")
    description = _coerce_str(payload.get("description"), path=f"{source}: description")
    theme_cfg = _coerce_mapping(payload.get("theme"), path=f"{source}: theme")
    variables = _coerce_string_dict(theme_cfg.get("variables"), path=f"{source}: theme.variables")
    colors = _coerce_string_dict(payload.get("colors"), path=f"{source}: colors", required=True)
    return ThemeSpec(
        id=theme_id,
        title=title,
        description=description,
        theme=Theme(
            name=theme_id,
            primary=_coerce_str(theme_cfg.get("primary"), path=f"{source}: theme.primary"),
            secondary=_coerce_str(theme_cfg.get("secondary"), path=f"{source}: theme.secondary"),
            accent=_coerce_str(theme_cfg.get("accent"), path=f"{source}: theme.accent"),
            foreground=_coerce_str(theme_cfg.get("foreground"), path=f"{source}: theme.foreground"),
            background=_coerce_str(theme_cfg.get("background"), path=f"{source}: theme.background"),
            surface=_coerce_str(theme_cfg.get("surface"), path=f"{source}: theme.surface"),
            panel=_coerce_str(theme_cfg.get("panel"), path=f"{source}: theme.panel"),
            success=_coerce_str(theme_cfg.get("success"), path=f"{source}: theme.success"),
            warning=_coerce_str(theme_cfg.get("warning"), path=f"{source}: theme.warning"),
            error=_coerce_str(theme_cfg.get("error"), path=f"{source}: theme.error"),
            dark=_coerce_bool(theme_cfg.get("dark"), True),
            variables=variables,
        ),
        syntax_theme=_coerce_optional_str(payload.get("syntax_theme"), "github-dark") or "github-dark",
        text_area_theme=_coerce_optional_str(payload.get("text_area_theme"), "dracula") or "dracula",
        colors=colors,
    )


def _load_theme_json(text: str, *, source: str) -> ThemeSpec:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ThemeLoadError(f"{source}: invalid JSON: {exc}") from exc
    return _theme_from_payload(_coerce_mapping(payload, path=source), source=source)


def _default_user_theme_dir() -> Path:
    override = os.environ.get(APP_ROOT_ENV_VAR, "").strip()
    root = Path(os.path.expanduser(override)).resolve() if override else (Path.home() / DEFAULT_APP_DIRNAME).resolve()
    return root / "themes"


def user_theme_dirs() -> list[Path]:
    raw_paths = [item for item in os.environ.get(_USER_THEME_PATHS_ENV, "").split(os.pathsep) if item.strip()]
    paths = [Path(os.path.expanduser(item)).resolve() for item in raw_paths]
    default_dir = _default_user_theme_dir()
    if default_dir not in paths:
        paths.append(default_dir)
    return paths


def _load_builtin_theme_specs() -> dict[str, ThemeSpec]:
    specs: dict[str, ThemeSpec] = {}
    root = resources.files(_BUILTIN_PACKAGE)
    for theme_id in BUILTIN_THEME_IDS:
        path = root / f"{theme_id}.json"
        spec = _load_theme_json(path.read_text(encoding="utf-8"), source=f"builtin:{theme_id}")
        specs[spec.id] = spec
    return specs


def _load_user_theme_specs() -> dict[str, ThemeSpec]:
    specs: dict[str, ThemeSpec] = {}
    for directory in user_theme_dirs():
        if not directory.exists() or not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                spec = _load_theme_json(path.read_text(encoding="utf-8"), source=str(path))
            except Exception:
                continue
            specs[spec.id] = spec
    return specs


@lru_cache(maxsize=1)
def theme_specs() -> dict[str, ThemeSpec]:
    specs = _load_builtin_theme_specs()
    specs.update(_load_user_theme_specs())
    return specs


def reload_theme_specs() -> None:
    theme_specs.cache_clear()


_FALLBACK_THEME_ID: Final[str] = "classic"


def available_theme_ids() -> list[str]:
    specs = theme_specs()
    builtins = [theme_id for theme_id in BUILTIN_THEME_IDS if theme_id in specs]
    custom = sorted(theme_id for theme_id in specs if theme_id not in BUILTIN_THEME_IDS)
    return [*builtins, *custom]


def theme_spec(theme_id: str) -> ThemeSpec:
    specs = theme_specs()
    resolved, _ = normalize_theme_id(theme_id, default=DEFAULT_THEME_ID, available=specs.keys())
    return specs.get(resolved) or specs[DEFAULT_THEME_ID]


def fallback_theme_spec() -> ThemeSpec:
    specs = theme_specs()
    return specs.get(_FALLBACK_THEME_ID) or next(iter(specs.values()))


def fallback_color(key: str, default: str = "") -> str:
    spec = fallback_theme_spec()
    return str(spec.colors.get(key, default))


def fallback_theme_variables() -> dict[str, str]:
    variables = fallback_theme_spec().theme.variables or {}
    return {key: str(value) for key, value in variables.items()}


def default_theme_variables() -> dict[str, str]:
    variables = theme_spec(DEFAULT_THEME_ID).theme.variables or {}
    merged = fallback_theme_variables()
    merged.update({key: str(value) for key, value in variables.items()})
    return merged
