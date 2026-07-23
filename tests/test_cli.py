from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphanus import cli as alphanus_cli
from core.configuration import load_global_config
from core.headless_protocol import EXIT_INVALID_INPUT, EXIT_POLICY_DENIED, EXIT_SUCCESS
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
    monkeypatch.setattr(alphanus_cli, "get_app_paths", lambda: paths)
    assert alphanus_cli._run_init(_init_args()) == 0
    assert paths.config_path.stat().st_mode & 0o777 == 0o600
    assert not (paths.state_root / ".env").exists()
    assert load_global_config(paths.config_path)["config_version"] == 1


def test_init_rejects_secret_command_line_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alphanus_cli, "get_app_paths", lambda: _paths(tmp_path))
    assert alphanus_cli._run_init(_init_args(api_key="sk-secret")) == 2


def test_resolve_project_root_uses_override(tmp_path: Path) -> None:
    target = tmp_path / "workspace"
    target.mkdir()
    assert alphanus_cli.resolve_project_root({}, override=str(target)) == target.resolve()


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
    monkeypatch.setattr(alphanus_cli, "get_app_paths", lambda: _paths(tmp_path))
    monkeypatch.setattr(alphanus_cli, "_load_runtime_config", lambda _paths, _args: ({"logging": {}}, []))
    monkeypatch.setattr(alphanus_cli, "_build_agent_runtime", lambda *_args, **_kwargs: (None, None, None, agent))
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
    monkeypatch.setattr(alphanus_cli, "get_app_paths", lambda: paths)
    monkeypatch.setattr(alphanus_cli, "_load_runtime_config", lambda _paths, _args: ({"logging": {}}, []))
    monkeypatch.setattr(
        alphanus_cli, "_build_agent_runtime", lambda *_args, **_kwargs: (None, _FakeMemory(), None, _FakeAgent(result, tmp_path))
    )
    monkeypatch.setattr(alphanus_cli.sys, "stdout", output)
    assert alphanus_cli._run_exec(_exec_args()) == EXIT_SUCCESS
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert all(record["schema_version"] == 1 for record in records)
    assert records[-1]["type"] == "run.completed"
    assert records[-1]["data"]["status"] == "success"


def test_exec_policy_denial_has_stable_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = io.StringIO()
    result = AgentTurnResult(status="error", content="", reasoning="", skill_exchanges=[], error="approval denied")
    monkeypatch.setattr(alphanus_cli, "get_app_paths", lambda: _paths(tmp_path))
    monkeypatch.setattr(alphanus_cli, "_load_runtime_config", lambda _paths, _args: ({"logging": {}}, []))
    monkeypatch.setattr(
        alphanus_cli, "_build_agent_runtime", lambda *_args, **_kwargs: (None, _FakeMemory(), None, _FakeAgent(result, tmp_path))
    )
    monkeypatch.setattr(alphanus_cli.sys, "stdout", output)
    assert alphanus_cli._run_exec(_exec_args()) == EXIT_POLICY_DENIED


def test_exec_rejects_empty_input_with_final_event(monkeypatch: pytest.MonkeyPatch) -> None:
    output = io.StringIO()
    monkeypatch.setattr(alphanus_cli.sys, "stdout", output)
    monkeypatch.setattr(alphanus_cli.sys, "stdin", io.StringIO(""))
    assert alphanus_cli._run_exec(_exec_args("")) == EXIT_INVALID_INPUT
    assert json.loads(output.getvalue().splitlines()[-1])["type"] == "run.completed"
