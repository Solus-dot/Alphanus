from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


DEFAULT_APP_DIRNAME = ".alphanus"
DEFAULT_WORKSPACE_DIRNAME = "Alphanus-Workspace"
APP_ROOT_ENV_VAR = "ALPHANUS_APP_ROOT"


@dataclass(frozen=True)
class AppPaths:
    app_root: Path
    bundled_skills_dir: Path
    repo_root: Path | None

    @property
    def config_path(self) -> Path:
        return self.app_root / "config" / "global_config.json"

    @property
    def dotenv_path(self) -> Path:
        return self.app_root / ".env"


def _is_repo_checkout(root: Path) -> bool:
    return (
        (root / "pyproject.toml").exists()
        and (root / "src").is_dir()
        and (root / "skills").is_dir()
        and (root / "config").is_dir()
    )


def _detect_repo_root() -> Path | None:
    candidates: list[Path] = []
    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    candidates.extend(path.resolve() for path in Path(__file__).resolve().parents)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if _is_repo_checkout(candidate):
            return candidate
    return None


def _bundled_skills_dir() -> Path:
    return Path(str(resources.files("alphanus_bundled"))).resolve()


def default_workspace_root(home_root: Path | None = None) -> Path:
    home = (home_root or Path.home()).resolve()
    return (home / "Desktop" / DEFAULT_WORKSPACE_DIRNAME).resolve()


def _installed_app_root() -> Path:
    override = os.environ.get(APP_ROOT_ENV_VAR, "").strip()
    if override:
        return Path(os.path.expanduser(override)).resolve()
    return (default_workspace_root() / DEFAULT_APP_DIRNAME).resolve()


def get_app_paths() -> AppPaths:
    repo_root = _detect_repo_root()
    app_root = repo_root if repo_root is not None else _installed_app_root()
    return AppPaths(
        app_root=app_root,
        bundled_skills_dir=_bundled_skills_dir(),
        repo_root=repo_root,
    )
