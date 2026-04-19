from __future__ import annotations

from types import SimpleNamespace

import alphanus_cli


def test_main_does_not_block_on_model_readiness_before_launching_tui(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.json"
    dotenv_path = tmp_path / ".env"
    bundled_skills_dir = tmp_path / "skills"
    bundled_skills_dir.mkdir()

    monkeypatch.setattr(
        alphanus_cli,
        "get_app_paths",
        lambda: SimpleNamespace(
            app_root=tmp_path,
            config_path=config_path,
            dotenv_path=dotenv_path,
            bundled_skills_dir=bundled_skills_dir,
        ),
    )
    monkeypatch.setattr(alphanus_cli, "load_dotenv", lambda _path: None)
    monkeypatch.setattr(
        alphanus_cli,
        "load_or_create_global_config",
        lambda _path, warnings=None: {
            "workspace": {"path": str(tmp_path / "ws")},
            "memory": {"backup_revisions": 2},
            "agent": {},
        },
    )
    monkeypatch.setattr(alphanus_cli, "normalize_config", lambda config: (config, []))
    monkeypatch.setattr(alphanus_cli, "validate_endpoint_policy", lambda config: None)
    monkeypatch.setattr(alphanus_cli, "resolve_path", lambda path, _root: path)
    monkeypatch.setattr(alphanus_cli, "WorkspaceManager", lambda workspace_root: SimpleNamespace(workspace_root=workspace_root))
    memory_calls: list[dict[str, object]] = []

    def _memory_stub(**kwargs):
        memory_calls.append(kwargs)
        return SimpleNamespace(
            stats=lambda **_kw: {"mode_label": "lexical", "min_score_default": 0.3},
        )

    monkeypatch.setattr(alphanus_cli, "VectorMemory", _memory_stub)
    runtime_calls: list[dict[str, object]] = []

    def _skill_runtime_stub(**kwargs):
        runtime_calls.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(alphanus_cli, "SkillRuntime", _skill_runtime_stub)

    class LoggerStub:
        def info(self, *_args, **_kwargs):
            return None

        def warning(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(alphanus_cli, "configure_logging", lambda _config: LoggerStub())

    agent_calls: list[str] = []

    class AgentStub:
        models_endpoint = "http://127.0.0.1:8080/v1/models"

        def __init__(self, **_kwargs):
            agent_calls.append("init")

        def ensure_ready(self, *_args, **_kwargs):
            agent_calls.append("ensure_ready")
            raise AssertionError("startup should not block on model readiness")

    monkeypatch.setattr(alphanus_cli, "Agent", AgentStub)

    app_calls: list[str] = []

    class AppStub:
        def __init__(self, *, agent, debug):
            assert isinstance(agent, AgentStub)
            assert debug is False
            app_calls.append("init")

        def run(self):
            app_calls.append("run")

    monkeypatch.setattr(alphanus_cli, "AlphanusTUI", AppStub)
    monkeypatch.setattr(alphanus_cli.argparse.ArgumentParser, "parse_args", lambda self: SimpleNamespace(debug=False, dangerously_skip_permissions=False))

    exit_code = alphanus_cli.main()

    assert exit_code == 0
    assert agent_calls == ["init"]
    assert app_calls == ["init", "run"]
    assert runtime_calls
    assert runtime_calls[0]["skills_dir"] == str(tmp_path / "skills")
    assert memory_calls
    assert memory_calls[0]["storage_path"] == str((tmp_path / "ws" / ".alphanus" / "memory" / "events.jsonl").resolve())
