from __future__ import annotations

import argparse
import io
import json
import urllib.error
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphanus import cli as alphanus_cli
from alphanus.commands import doctor as doctor_command
from alphanus.commands import exec as exec_command
from alphanus.commands import init as init_command
from alphanus.runtime_factory import _load_runtime_config, resolve_project_root
from core.configuration import load_global_config
from core.headless_protocol import EXIT_INVALID_INPUT, EXIT_MODEL_FAILURE, EXIT_POLICY_DENIED, EXIT_SUCCESS
from core.types import AgentTurnResult


def _paths(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / ".alphanus"
    return SimpleNamespace(
        app_root=tmp_path,
        state_root=root,
        config_path=root / "config" / "config.toml",
        bundled_skills_dir=tmp_path / "bundled-skills",
        user_skills_dir=root / "skills",
        repo_root=tmp_path,
    )


def _init_args(**updates: object) -> SimpleNamespace:
    values = {
        "section": "all",
        "non_interactive": True,
        "reset": False,
        "project_root": "",
        "debug": False,
        "base_url": "http://127.0.0.1:8080",
        "responses_endpoint": "",
        "model_endpoint": "",
        "models_endpoint": "",
        "endpoint_mode": "chat",
        "backend_profile": "auto",
        "api_key": "",
        "api_key_env": "ALPHANUS_API_KEY",
        "backend_api_key_env": "",
        "search_provider": "searxng",
        "search_fallback_provider": "none",
        "searxng_base_url": "",
        "tavily_api_key": "",
        "tavily_api_key_env": "TAVILY_API_KEY",
        "theme": "classic",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_init_writes_owner_only_versioned_toml_and_no_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(init_command, "get_app_paths", lambda: paths)
    assert alphanus_cli._run_init(_init_args()) == 0
    assert paths.config_path.stat().st_mode & 0o777 == 0o600
    assert not (paths.state_root / ".env").exists()
    assert load_global_config(paths.config_path)["config_version"] == 1


def test_init_rejects_secret_command_line_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(init_command, "get_app_paths", lambda: _paths(tmp_path))
    assert alphanus_cli._run_init(_init_args(api_key="sk-secret")) == 2


def test_resolve_project_root_uses_override(tmp_path: Path) -> None:
    target = tmp_path / "workspace"
    target.mkdir()
    assert resolve_project_root({}, override=str(target)) == target.resolve()


@pytest.mark.parametrize(
    "argv",
    [
        ["--project-root", "/tmp/workspace", "run"],
        ["run", "--project-root", "/tmp/workspace"],
    ],
)
def test_parser_preserves_project_root_on_either_side_of_subcommand(argv: list[str]) -> None:
    args = alphanus_cli._build_parser().parse_args(argv)
    assert args.project_root == "/tmp/workspace"


def test_run_endpoint_flags_and_validation() -> None:
    args = alphanus_cli._build_parser().parse_args(["run", "--endpoint", "127.0.0.1:8080", "--api-key", "secret"])
    assert args.api_key == "secret"
    assert alphanus_cli._run_endpoint(args.endpoint, args.api_key) == "http://127.0.0.1:8080"
    with pytest.raises(ValueError, match="port"):
        alphanus_cli._run_endpoint("127.0.0.1:70000", "")
    with pytest.raises(ValueError, match="HOST:PORT"):
        alphanus_cli._run_endpoint("127.0.0.1", "")


def test_run_endpoint_probe_distinguishes_offline_and_restricted(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_restricted(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.HTTPError("http://localhost:8080/v1/models", 401, "Unauthorized", Message(), None)

    monkeypatch.setattr(alphanus_cli.urllib.request, "urlopen", fail_restricted)
    with pytest.raises(ValueError, match="access restricted: no API key"):
        alphanus_cli._probe_endpoint("http://localhost:8080", "")

    def fail_offline(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(alphanus_cli.urllib.request, "urlopen", fail_offline)
    with pytest.raises(ValueError, match="endpoint is offline"):
        alphanus_cli._probe_endpoint("http://localhost:8080", "")


def test_run_endpoint_overrides_runtime_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    paths.config_path.parent.mkdir(parents=True)
    paths.config_path.write_text("config_version = 1\n", encoding="utf-8")
    monkeypatch.setenv("ALPHANUS_RUN_ENDPOINT", "http://10.0.0.2:9000")
    monkeypatch.setenv("ALPHANUS_RUN_API_KEY", "secret")
    config, _warnings = _load_runtime_config(paths, argparse.Namespace(debug=False, project_root=""))
    assert config.agent.model_endpoint == "http://10.0.0.2:9000/v1/chat/completions"
    assert config.agent.models_endpoint == "http://10.0.0.2:9000/v1/models"
    assert config.agent.api_key == "env:ALPHANUS_RUN_API_KEY"


def test_doctor_json_ok_matches_nonzero_exit_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = io.StringIO()
    report = {
        "agent": {"ready": True, "endpoint_policy_error": ""},
        "project": {"exists": True, "writable": True},
        "search": {"ready": False},
        "retrieval": {"ready": True},
    }
    agent = SimpleNamespace(doctor_report=lambda: report)
    args = SimpleNamespace(json=True, debug=False, project_root="")
    monkeypatch.setattr(doctor_command, "get_app_paths", lambda: _paths(tmp_path))
    monkeypatch.setattr(doctor_command, "_load_runtime_config", lambda _paths, _args: ({"logging": {}}, []))
    monkeypatch.setattr(doctor_command, "_build_agent_runtime", lambda *_args, **_kwargs: (None, None, None, agent))
    monkeypatch.setattr(alphanus_cli.sys, "stdout", output)

    assert alphanus_cli._run_doctor(args) == 1
    payload = json.loads(output.getvalue())
    assert payload["ok"] is False
    assert payload["failures"] == ["search"]


class _FakeMemory:
    def flush(self) -> None:
        return


class _FakeAgent:
    def __init__(self, result: AgentTurnResult, root: Path) -> None:
        self.result = result
        self.skill_runtime = SimpleNamespace(project=SimpleNamespace(project_root=root))

    def run_turn(self, **kwargs):
        callback = kwargs["on_event"]
        callback({"type": "content_token", "text": "hello"})
        if self.result.error == "approval denied":
            kwargs["request_approval"]({"kind": "shell_command"})
        return self.result


def _exec_args(prompt: str = "hello") -> SimpleNamespace:
    return SimpleNamespace(prompt=prompt, input="text", approval_policy="deny", no_thinking=False, project_root="", debug=False)


def test_exec_emits_versioned_jsonl_and_success_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = io.StringIO()
    paths = _paths(tmp_path)
    result = AgentTurnResult(status="done", content="final", reasoning="", skill_exchanges=[])
    monkeypatch.setattr(exec_command, "get_app_paths", lambda: paths)
    monkeypatch.setattr(exec_command, "_load_runtime_config", lambda _paths, _args: ({"logging": {}}, []))
    monkeypatch.setattr(
        exec_command, "_build_agent_runtime", lambda *_args, **_kwargs: (None, _FakeMemory(), None, _FakeAgent(result, tmp_path))
    )
    monkeypatch.setattr(alphanus_cli.sys, "stdout", output)
    assert alphanus_cli._run_exec(_exec_args()) == EXIT_SUCCESS
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert all(record["schema_version"] == 1 for record in records)
    assert records[-1]["type"] == "run.completed"
    assert records[-1]["data"]["status"] == "success"


def test_exec_policy_denial_has_stable_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = io.StringIO()
    result = AgentTurnResult(
        status="error",
        content="",
        reasoning="",
        skill_exchanges=[],
        error="approval denied",
        error_code="E_POLICY",
    )
    monkeypatch.setattr(exec_command, "get_app_paths", lambda: _paths(tmp_path))
    monkeypatch.setattr(exec_command, "_load_runtime_config", lambda _paths, _args: ({"logging": {}}, []))
    monkeypatch.setattr(
        exec_command, "_build_agent_runtime", lambda *_args, **_kwargs: (None, _FakeMemory(), None, _FakeAgent(result, tmp_path))
    )
    monkeypatch.setattr(alphanus_cli.sys, "stdout", output)
    assert alphanus_cli._run_exec(_exec_args()) == EXIT_POLICY_DENIED


def test_exec_provider_permission_error_is_not_a_policy_denial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = io.StringIO()
    result = AgentTurnResult(
        status="error",
        content="",
        reasoning="",
        skill_exchanges=[],
        error="Provider permission denied",
        error_code="E_PROVIDER",
    )
    monkeypatch.setattr(exec_command, "get_app_paths", lambda: _paths(tmp_path))
    monkeypatch.setattr(exec_command, "_load_runtime_config", lambda _paths, _args: ({"logging": {}}, []))
    monkeypatch.setattr(
        exec_command, "_build_agent_runtime", lambda *_args, **_kwargs: (None, _FakeMemory(), None, _FakeAgent(result, tmp_path))
    )
    monkeypatch.setattr(alphanus_cli.sys, "stdout", output)

    assert alphanus_cli._run_exec(_exec_args()) == EXIT_MODEL_FAILURE
    error = next(record for record in map(json.loads, output.getvalue().splitlines()) if record["type"] == "run.error")
    assert error["data"]["category"] == "model"


def test_exec_rejects_empty_input_with_final_event(monkeypatch: pytest.MonkeyPatch) -> None:
    output = io.StringIO()
    monkeypatch.setattr(alphanus_cli.sys, "stdout", output)
    monkeypatch.setattr(alphanus_cli.sys, "stdin", io.StringIO(""))
    assert alphanus_cli._run_exec(_exec_args("")) == EXIT_INVALID_INPUT
    assert json.loads(output.getvalue().splitlines()[-1])["type"] == "run.completed"
