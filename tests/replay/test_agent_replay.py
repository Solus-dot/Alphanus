"""Replay tests for behavior regressions.

Fixtures in ``tests/replay/fixtures`` script model passes and expected product
outcomes. The harness keeps real Agent/Orchestrator/SkillRuntime/tool execution
in the path, but replaces network model calls with deterministic
``StreamPassResult`` objects.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, cast

import pytest

from agent.core import Agent
from agent.runtime_hooks import TurnRuntimeHooks
from agent.types import ModelStatus, StreamPassResult, ToolCall, TurnClassification
from core.memory import LexicalMemory
from core.message_types import ChatMessage
from core.project import ProjectRuntime
from core.types import JsonObject
from skills.runtime import SkillContext, SkillRuntime

FIXTURE_DIR = Path(__file__).with_name("fixtures")
REPO_ROOT = Path(__file__).resolve().parents[2]


class ReplayHooks:
    def __init__(self, agent: Agent, fixture: dict[str, Any]) -> None:
        self.agent = agent
        self.fixture = fixture
        self.passes = list(fixture.get("model_passes", []))
        self.payloads: list[dict[str, Any]] = []

    def call_with_retry(self, payload: JsonObject, stop_event, on_event, pass_id: str) -> StreamPassResult:
        self.payloads.append(dict(payload))
        if stop_event is not None and stop_event.is_set():
            return StreamPassResult(finish_reason="cancelled")
        if not self.passes:
            raise AssertionError(f"Replay fixture exhausted before {pass_id}")
        scripted = self.passes.pop(0)
        for event in scripted.get("events", []):
            if callable(on_event):
                on_event(event)
        return StreamPassResult(
            finish_reason=str(scripted.get("finish_reason", "stop")),
            content=str(scripted.get("content", "")),
            reasoning=str(scripted.get("reasoning", "")),
            tool_calls=[
                ToolCall(
                    stream_id=f"{pass_id}:{idx}",
                    index=idx,
                    id=str(item.get("id") or f"{pass_id}_call_{idx}"),
                    name=str(item["name"]),
                    arguments=dict(item.get("arguments", {})),
                )
                for idx, item in enumerate(scripted.get("tool_calls", []))
            ],
        )

    def build_skill_context(
        self,
        user_input: str,
        branch_labels: list[str],
        attachments: list[str],
        history_messages: list[ChatMessage] | None = None,
        loaded_skill_ids: list[str] | None = None,
    ) -> SkillContext:
        return self.agent.classifier.build_skill_context(
            user_input,
            branch_labels,
            attachments,
            history_messages,
            loaded_skill_ids,
        )

    def classify_context(self, ctx: SkillContext, stop_event=None) -> TurnClassification:
        raw = self.fixture.get("classification", {})
        return TurnClassification(
            time_sensitive=bool(raw.get("time_sensitive", False)),
            requires_project_action=bool(raw.get("requires_project_action", False)),
            prefer_local_project_tools=bool(raw.get("prefer_local_project_tools", False)),
            explicit_external_path=str(raw.get("explicit_external_path", "")),
            followup_kind=str(raw.get("followup_kind", "new_request")),
            source="replay",
        )

    def select_skills(self, ctx: SkillContext, stop_event):
        selected = [skill for skill in self.agent.skill_runtime.skills_by_ids(self.fixture.get("skills", [])) if skill is not None]
        return self.classify_context(ctx, stop_event=stop_event), selected


def _load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_initial_files(project_root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        target = project_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _write_fixture_skills(skills_root: Path, fixture: dict[str, Any]) -> None:
    for skill_id, files in dict(fixture.get("fixture_skills", {})).items():
        skill_dir = skills_root / str(skill_id)
        skill_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in dict(files).items():
            target = skill_dir / str(filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(content), encoding="utf-8")


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _build_agent(tmp_path: Path, fixture: dict[str, Any]) -> Agent:
    home = tmp_path / "home"
    project_root = home / "project"
    skills_root = home / "skills"
    home.mkdir()
    project_root.mkdir()
    skills_root.mkdir()
    shutil.copytree(REPO_ROOT / "bundled-skills", skills_root, dirs_exist_ok=True)
    _write_fixture_skills(skills_root, fixture)
    config = _deep_merge(
        {
            "agent": {
                "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
                "models_endpoint": "http://127.0.0.1:8080/v1/models",
                "request_timeout_s": 1,
                "readiness_timeout_s": 1,
                "readiness_poll_s": 0.01,
                "enable_thinking": True,
                "tls_verify": True,
                "max_tokens": 256,
            },
            "permissions": {"mode": "project-write", "approvals": "on-boundary", "network": False},
            "retrieval": {"enabled": False},
            "memory": {"auto_capture": False},
        },
        dict(fixture.get("config", {})),
    )
    runtime = SkillRuntime(
        skills_dir=str(home / "user-skills"),
        bundled_skills_dir=str(skills_root),
        project=ProjectRuntime(str(project_root)),
        memory=LexicalMemory(storage_path=str(tmp_path / "memory.jsonl")),
        config=config,
    )
    agent = Agent(config, runtime)
    now = time.monotonic()
    agent.llm_client.provider._store_model_status(
        ModelStatus(
            state="online",
            model_name="replay-model",
            context_window=8192,
            last_checked_at=now,
            last_success_at=now,
            endpoint=agent.models_endpoint,
        )
    )
    return agent


def _tool_results_by_name(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("type") != "tool_result":
            continue
        results.setdefault(str(event.get("name", "")), []).append(dict(event.get("result", {})))
    return results


@pytest.mark.parametrize("fixture_path", sorted(FIXTURE_DIR.glob("*.json")), ids=lambda path: path.stem)
def test_replay_core_coding_loop(fixture_path: Path, tmp_path: Path) -> None:
    fixture = _load_fixture(fixture_path)
    agent = _build_agent(tmp_path, fixture)
    project_root = Path(agent.skill_runtime.project.project_root)
    _write_initial_files(project_root, dict(fixture.get("initial_files", {})))

    hooks = ReplayHooks(agent, fixture)
    runtime_hooks = cast(TurnRuntimeHooks, hooks)
    agent.orchestrator.bind_runtime_hooks(runtime_hooks)
    agent.classifier.bind_runtime_hooks(runtime_hooks)

    events: list[dict[str, Any]] = []
    stop_event = None
    if fixture.get("cancel_on_event_type"):
        import threading

        stop_event = threading.Event()

    def on_event(event: dict[str, Any]) -> None:
        events.append(event)
        if stop_event is not None and event.get("type") == fixture.get("cancel_on_event_type"):
            stop_event.set()

    request_approval_calls: list[dict[str, Any]] = []

    def request_approval(request: dict[str, Any]) -> bool:
        request_approval_calls.append(request)
        return bool(fixture.get("request_approval", False))

    result = agent.run_turn(
        history_messages=cast(
            list[ChatMessage],
            list(fixture.get("history_messages", [{"role": "user", "content": fixture["user_input"]}])),
        ),
        user_input=str(fixture["user_input"]),
        thinking=bool(fixture.get("thinking", True)),
        loaded_skill_ids=list(fixture.get("skills", [])),
        stop_event=stop_event,
        on_event=on_event,
        request_approval=request_approval,
    )

    expected = fixture["expect"]
    assert result.status == expected["status"]
    if expected.get("content_equals") is not None:
        assert result.content == expected["content_equals"]
    if expected.get("content_contains"):
        assert str(expected["content_contains"]) in result.content
    for text in expected.get("content_not_contains", []):
        assert str(text) not in result.content

    tool_sequence = [event["name"] for event in events if event.get("type") == "tool_call"]
    assert tool_sequence == expected.get("tool_sequence", [])
    if "skill_exchange_tools" in expected:
        assert [msg.get("name") for msg in result.skill_exchanges if msg.get("role") == "tool"] == expected["skill_exchange_tools"]
    for event_type in expected.get("event_types", []):
        assert event_type in {event.get("type") for event in events}

    tool_results = _tool_results_by_name(events)
    for name, expectation in expected.get("tool_results", {}).items():
        assert tool_results[name], f"missing tool result for {name}"
        result_payload = tool_results[name][-1]
        if "ok" in expectation:
            assert result_payload.get("ok") is expectation["ok"]
        if "error_code" in expectation:
            error = result_payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == expectation["error_code"]

    for relative, content in expected.get("project_files", {}).items():
        assert (project_root / relative).read_text(encoding="utf-8") == content
    for relative in expected.get("project_paths", []):
        assert (project_root / relative).exists(), f"missing project path: {relative}"

    if fixture.get("expect_approval_call"):
        assert request_approval_calls
    assert len(hooks.passes) == int(expected.get("unconsumed_model_passes", 0))
