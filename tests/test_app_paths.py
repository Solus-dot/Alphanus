from __future__ import annotations

from pathlib import Path

from alphanus_paths import DEFAULT_APP_DIRNAME, DEFAULT_WORKSPACE_DIRNAME, get_app_paths


def test_get_app_paths_uses_repo_root_inside_checkout() -> None:
    paths = get_app_paths()

    assert paths.repo_root == Path(__file__).resolve().parents[1]
    assert paths.app_root == paths.repo_root
    assert paths.config_path == paths.repo_root / "config" / "global_config.json"
    assert (paths.bundled_skills_dir / "utilities" / "SKILL.md").exists()


def test_get_app_paths_falls_back_to_user_dir_outside_checkout(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    outside = tmp_path / "outside"
    home.mkdir()
    outside.mkdir()
    (outside / "src").mkdir()
    (outside / "skills").mkdir()
    (outside / "config").mkdir()
    (outside / "pyproject.toml").write_text("[project]\nname='not-alphanus'\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(outside)

    import alphanus_paths

    monkeypatch.setattr(alphanus_paths, "__file__", str(tmp_path / "site-packages" / "alphanus_paths.py"))

    paths = alphanus_paths.get_app_paths()

    assert paths.repo_root is None
    assert paths.app_root == home / "Desktop" / DEFAULT_WORKSPACE_DIRNAME / DEFAULT_APP_DIRNAME
    assert paths.config_path == home / "Desktop" / DEFAULT_WORKSPACE_DIRNAME / DEFAULT_APP_DIRNAME / "config" / "global_config.json"
    assert (paths.bundled_skills_dir / "utilities" / "SKILL.md").exists()
