from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import alphanus_cli


def test_main_does_not_block_on_model_readiness_before_launching_tui(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.json"
    dotenv_path = tmp_path / ".env"
    bundled_skills_dir = tmp_path / "skills"
    bundled_skills_dir.mkdir()
    config_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        alphanus_cli,
        "get_app_paths",
        lambda: SimpleNamespace(
            app_root=tmp_path,
            state_root=tmp_path,
            config_path=config_path,
            dotenv_path=dotenv_path,
            bundled_skills_dir=bundled_skills_dir,
            repo_root=tmp_path,
        ),
    )
    monkeypatch.setattr(alphanus_cli, "load_dotenv", lambda _path: None)
    monkeypatch.setattr(
        alphanus_cli,
        "load_global_config",
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
    assert memory_calls[0]["storage_path"] == str((tmp_path / "memory" / "events.jsonl").resolve())


def test_main_fails_fast_when_global_config_missing(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config" / "global_config.json"
    dotenv_path = tmp_path / ".env"

    monkeypatch.setattr(
        alphanus_cli,
        "get_app_paths",
        lambda: SimpleNamespace(
            app_root=tmp_path,
            state_root=tmp_path,
            config_path=config_path,
            dotenv_path=dotenv_path,
            bundled_skills_dir=tmp_path / "skills",
            repo_root=tmp_path,
        ),
    )
    monkeypatch.setattr(alphanus_cli.argparse.ArgumentParser, "parse_args", lambda self: SimpleNamespace(command="", debug=False, dangerously_skip_permissions=False))

    exit_code = alphanus_cli.main()

    assert exit_code == 2


def test_init_non_interactive_writes_global_config_and_env_template(monkeypatch, tmp_path) -> None:
    state_root = tmp_path / ".alphanus"
    config_path = state_root / "config" / "global_config.json"
    dotenv_path = state_root / ".env"

    monkeypatch.setattr(
        alphanus_cli,
        "get_app_paths",
        lambda: SimpleNamespace(
            app_root=tmp_path,
            state_root=state_root,
            config_path=config_path,
            dotenv_path=dotenv_path,
            bundled_skills_dir=tmp_path / "skills",
            repo_root=tmp_path,
        ),
    )
    monkeypatch.setattr(
        alphanus_cli.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            command="init",
            non_interactive=True,
            workspace_path=str(tmp_path / "ws"),
            model_endpoint="http://127.0.0.1:8080/v1/chat/completions",
            models_endpoint="http://127.0.0.1:8080/v1/models",
            search_provider="tavily",
            debug=False,
            dangerously_skip_permissions=False,
        ),
    )

    exit_code = alphanus_cli.main()

    assert exit_code == 0
    assert config_path.exists()
    assert dotenv_path.exists()
    stored = json.loads(config_path.read_text(encoding="utf-8"))
    assert stored["workspace"]["path"] == str(tmp_path / "ws")


def test_parser_accepts_global_flags_after_subcommand() -> None:
    parser = alphanus_cli._build_parser()

    args = parser.parse_args(["doctor", "--json", "--debug"])

    assert args.command == "doctor"
    assert args.json is True
    assert args.debug is True


def test_doctor_json_output_is_machine_readable(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(
        alphanus_cli,
        "get_app_paths",
        lambda: SimpleNamespace(
            config_path=tmp_path / "config.json",
            dotenv_path=tmp_path / ".env",
            state_root=tmp_path,
        ),
    )
    monkeypatch.setattr(alphanus_cli, "_load_runtime_config", lambda _app_paths, _args: ({}, ["legacy warning"]))

    report = {
        "agent": {"ready": True, "endpoint_policy_error": ""},
        "workspace": {"exists": True, "writable": True},
        "search": {"ready": True},
    }
    monkeypatch.setattr(
        alphanus_cli,
        "_build_agent_runtime",
        lambda *_args, **_kwargs: (
            SimpleNamespace(workspace_root=Path("/tmp/ws")),
            SimpleNamespace(storage_path=Path("/tmp/memory/events.jsonl")),
            SimpleNamespace(),
            SimpleNamespace(doctor_report=lambda: report),
        ),
    )

    exit_code = alphanus_cli._run_doctor(
        SimpleNamespace(
            json=True,
            debug=False,
            dangerously_skip_permissions=False,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["config_warnings"] == ["legacy warning"]
    assert payload["agent"]["ready"] is True


def test_init_partial_reset_preserves_unselected_sections(monkeypatch, tmp_path) -> None:
    state_root = tmp_path / ".alphanus"
    config_path = state_root / "config" / "global_config.json"
    dotenv_path = state_root / ".env"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}", encoding="utf-8")

    existing_config = {
        "workspace": {"path": str(tmp_path / "custom-workspace")},
        "agent": {
            "model_endpoint": "http://custom-host/v1/chat/completions",
            "models_endpoint": "http://custom-host/v1/models",
            "tls_verify": False,
        },
        "search": {"provider": "brave"},
    }

    monkeypatch.setattr(
        alphanus_cli,
        "get_app_paths",
        lambda: SimpleNamespace(
            app_root=tmp_path,
            state_root=state_root,
            config_path=config_path,
            dotenv_path=dotenv_path,
            bundled_skills_dir=tmp_path / "skills",
            repo_root=tmp_path,
        ),
    )
    monkeypatch.setattr(alphanus_cli, "load_global_config", lambda _path, warnings=None: existing_config)

    exit_code = alphanus_cli._run_init(
        SimpleNamespace(
            section="model",
            non_interactive=True,
            reset=True,
            workspace_path="",
            model_endpoint="",
            models_endpoint="",
            search_provider="",
        )
    )

    assert exit_code == 0
    stored = json.loads(config_path.read_text(encoding="utf-8"))
    assert stored["workspace"]["path"] == str(tmp_path / "custom-workspace")
    assert stored["search"]["provider"] == "brave"
    assert stored["agent"]["tls_verify"] is False
    assert stored["agent"]["model_endpoint"] == alphanus_cli.DEFAULT_CONFIG["agent"]["model_endpoint"]
    assert stored["agent"]["models_endpoint"] == alphanus_cli.DEFAULT_CONFIG["agent"]["models_endpoint"]
