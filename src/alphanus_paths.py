from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources, util
from pathlib import Path

DEFAULT_APP_DIRNAME = ".alphanus"
APP_ROOT_ENV_VAR = "ALPHANUS_APP_ROOT"


@dataclass(frozen=True)
class AppPaths:
    app_root: Path
    state_root: Path
    bundled_skills_dir: Path
    user_skills_dir: Path
    repo_root: Path | None

    @property
    def config_path(self) -> Path:
        return self.state_root / "config" / "global_config.json"

    @property
    def dotenv_path(self) -> Path:
        return self.state_root / ".env"


def _is_repo_checkout(root: Path) -> bool:
    return (root / "pyproject.toml").exists() and (root / "src").is_dir() and (root / "bundled-skills").is_dir() and (root / "config").is_dir()


def _detect_repo_root() -> Path | None:
    return next((candidate for candidate in Path(__file__).resolve().parents if _is_repo_checkout(candidate)), None)


def default_state_root(home_root: Path | None = None) -> Path:
    return ((home_root or Path.home()).resolve() / DEFAULT_APP_DIRNAME).resolve()


def _installed_state_root() -> Path:
    override = os.environ.get(APP_ROOT_ENV_VAR, "").strip()
    return Path(os.path.expanduser(override)).resolve() if override else default_state_root()


def _bundled_skills_dir() -> Path:
    repo_root = _detect_repo_root()
    if repo_root is not None:
        return (repo_root / "bundled-skills").resolve()
    try:
        return Path(str(resources.files("alphanus_bundled"))).resolve()
    except Exception:
        spec = util.find_spec("alphanus_bundled")
        for raw_path in spec.submodule_search_locations or [] if spec else []:
            path = Path(str(raw_path))
            if path.is_dir():
                return path.resolve()
        raise


def get_app_paths() -> AppPaths:
    repo_root = _detect_repo_root()
    state_root = _installed_state_root()
    user_skills_dir = (repo_root / "skills").resolve() if repo_root is not None else (state_root / "skills").resolve()
    return AppPaths(
        app_root=repo_root if repo_root is not None else state_root,
        state_root=state_root,
        bundled_skills_dir=_bundled_skills_dir(),
        user_skills_dir=user_skills_dir,
        repo_root=repo_root,
    )
