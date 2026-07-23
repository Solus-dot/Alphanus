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
        return self.state_root / "config" / "config.toml"


def default_state_root(base_dir: Path | None = None) -> Path:
    return ((base_dir or Path.home()).resolve() / DEFAULT_APP_DIRNAME).resolve()


def get_app_paths() -> AppPaths:
    repo_root = next(
        (
            candidate
            for candidate in Path(__file__).resolve().parents
            if (candidate / "pyproject.toml").exists()
            and (candidate / "src").is_dir()
            and (candidate / "bundled-skills").is_dir()
            and (candidate / "config").is_dir()
        ),
        None,
    )
    override = os.environ.get(APP_ROOT_ENV_VAR, "").strip()
    state_root = Path(os.path.expanduser(override)).resolve() if override else default_state_root()
    user_skills_dir = (repo_root / "skills").resolve() if repo_root is not None else (state_root / "skills").resolve()
    if repo_root is not None:
        bundled_skills_dir = (repo_root / "bundled-skills").resolve()
    else:
        try:
            bundled_skills_dir = Path(str(resources.files("alphanus_bundled"))).resolve()
        except Exception:
            spec = util.find_spec("alphanus_bundled")
            for raw_path in spec.submodule_search_locations or [] if spec else []:
                path = Path(str(raw_path))
                if path.is_dir():
                    bundled_skills_dir = path.resolve()
                    break
            else:
                raise
    return AppPaths(
        app_root=repo_root if repo_root is not None else state_root,
        state_root=state_root,
        bundled_skills_dir=bundled_skills_dir,
        user_skills_dir=user_skills_dir,
        repo_root=repo_root,
    )
