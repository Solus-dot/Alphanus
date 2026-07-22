from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.message_types import JSONValue
from core.runtime_protocol import (
    MAX_RUNTIME_FRAME_BYTES,
    RUNTIME_PROTOCOL_VERSION,
    RuntimeEmitter,
    RuntimeProtocolError,
    decode_runtime_frame,
)
from core.themes import ThemeLoadError, _read, available_theme_ids, reload_themes, theme_payload
from core.types import ModelStatus
from core.ui_commands import command_catalog


class _RuntimeMemory:
    def flush(self) -> None:
        return

    def close(self) -> None:
        return


class _RuntimeAgent:
    def __init__(self, root: Path) -> None:
        self.config: dict[str, object] = {"tui": {"theme": "classic"}}
        self.skill_runtime = SimpleNamespace(
            project=SimpleNamespace(project_root=root),
            skills_by_ids=lambda _ids: [],
        )

    def get_model_status(self) -> ModelStatus:
        return ModelStatus(
            state="unknown",
            model_name="test-model",
            last_error="not probed",
            endpoint="http://127.0.0.1:8080/v1/models",
        )

    def refresh_model_status(self, timeout_s: float | None = None, force: bool = False) -> ModelStatus:
        assert timeout_s == 2.0
        assert force is True
        return ModelStatus(
            state="online",
            model_name="test-model",
            context_window=32_768,
            endpoint="http://127.0.0.1:8080/v1/models",
        )


class _RecordingRuntimeAgent(_RuntimeAgent):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.histories: list[list[dict[str, object]]] = []

    def run_turn(self, **kwargs: object) -> SimpleNamespace:
        history = kwargs["history_messages"]
        assert isinstance(history, list)
        self.histories.append(history)
        current = history[-1]
        assert current["role"] == "user"
        reply = f"reply:{current['content']}"
        on_event = kwargs.get("on_event")
        if callable(on_event):
            on_event({"type": "content_token", "text": reply})
        return SimpleNamespace(
            status="done",
            content=reply,
            error="",
            skill_exchanges=[],
        )


def test_golden_runtime_requests_are_valid() -> None:
    fixture = Path(__file__).parent / "fixtures" / "runtime_protocol_v1.jsonl"
    frames = [decode_runtime_frame(line) for line in fixture.read_text(encoding="utf-8").splitlines()]
    assert [frame["type"] for frame in frames] == ["hello", "state.get", "turn.start", "turn.cancel", "shutdown"]
    assert all(frame["protocol_version"] == RUNTIME_PROTOCOL_VERSION for frame in frames)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"protocol_version": 2, "type": "hello", "request_id": "x", "data": {}}, "unsupported runtime protocol"),
        ({"protocol_version": 1, "type": "", "request_id": "x", "data": {}}, "non-empty type"),
        ({"protocol_version": 1, "type": "hello", "request_id": "", "data": {}}, "non-empty request_id"),
        ({"protocol_version": 1, "type": "hello", "request_id": "x", "data": []}, "data must be an object"),
    ],
)
def test_runtime_request_validation(payload: object, message: str) -> None:
    with pytest.raises(RuntimeProtocolError, match=message):
        decode_runtime_frame(json.dumps(payload))


def test_runtime_frames_are_size_bounded_before_json_allocation() -> None:
    with pytest.raises(RuntimeProtocolError, match="exceeds 1 MiB"):
        decode_runtime_frame("x" * (MAX_RUNTIME_FRAME_BYTES + 1))


def test_runtime_emitter_sequences_and_flushes() -> None:
    stream = io.StringIO()
    emitter = RuntimeEmitter(stream)
    emitter.emit("runtime.ready", request_id="one", data={"ok": True})
    emitter.emit("request.completed", request_id="one", data={"status": "ok"})
    frames = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [frame["sequence"] for frame in frames] == [1, 2]
    assert frames[0]["data"] == {"ok": True}


def test_runtime_server_handshake_and_shutdown(tmp_path: Path) -> None:
    from core.runtime_server import RuntimeServer

    requests = [
        {"protocol_version": 1, "type": "hello", "request_id": "hello", "data": {"min_protocol": 1, "max_protocol": 1}},
        {"protocol_version": 1, "type": "shutdown", "request_id": "bye", "data": {}},
    ]
    output = io.StringIO()
    server = RuntimeServer(
        agent=_RuntimeAgent(tmp_path),
        memory=_RuntimeMemory(),
        state_root=tmp_path / "state",
        config_path=tmp_path / "config.toml",
        input_stream=io.StringIO("".join(json.dumps(frame) + "\n" for frame in requests)),
        output_stream=output,
    )
    assert server.serve() == 0
    frames = [json.loads(line) for line in output.getvalue().splitlines()]
    assert any(frame["type"] == "runtime.ready" for frame in frames)
    completed = [frame for frame in frames if frame["type"] == "request.completed"]
    assert [(frame["request_id"], frame["data"]["status"]) for frame in completed] == [("hello", "ok"), ("bye", "ok")]


def test_runtime_snapshot_uses_latest_bounded_page(tmp_path: Path) -> None:
    from core.runtime_server import RuntimeServer

    server = RuntimeServer(
        agent=_RuntimeAgent(tmp_path),
        memory=_RuntimeMemory(),
        state_root=tmp_path / "state",
        config_path=tmp_path / "config.toml",
        input_stream=io.StringIO(),
        output_stream=io.StringIO(),
    )
    try:
        for index in range(110):
            turn = server.session.tree.add_turn(f"{index}:" + "😀" * 5_000)
            server.session.tree.complete_turn(turn.id, "界" * 5_000)
        snapshot = server._snapshot()
        assert 0 < len(snapshot["transcript"]) <= 100
        assert snapshot["transcript_offset"] + len(snapshot["transcript"]) == 110
        assert str(snapshot["transcript"][-1]["user"]).startswith("109:")
        assert snapshot["transcript"][-1]["assistant"] == "界" * 5_000
        assert "omitted" not in snapshot["transcript"][-1]["assistant"]
        assert snapshot["session"]["branch_name"] == "root"
        assert snapshot["model"]["model_name"] == "test-model"
        assert snapshot["model"]["endpoint"] == "http://127.0.0.1:8080/v1/models"
        assert len(json.dumps(snapshot, ensure_ascii=False).encode("utf-8")) < MAX_RUNTIME_FRAME_BYTES
    finally:
        server.close()


def test_runtime_snapshot_bounds_one_pathological_turn_and_emits_it(tmp_path: Path) -> None:
    from core.runtime_server import TRANSCRIPT_PAGE_BYTES, RuntimeServer, _encoded_size

    output = io.StringIO()
    server = RuntimeServer(
        agent=_RuntimeAgent(tmp_path),
        memory=_RuntimeMemory(),
        state_root=tmp_path / "state",
        config_path=tmp_path / "config.toml",
        input_stream=io.StringIO(),
        output_stream=output,
    )
    try:
        turn = server.session.tree.add_turn("🚀" * 100_000)
        activity: list[dict[str, JSONValue]] = []
        for index in range(300):
            activity.append({"kind": "reasoning", "text": "界" * 40_000})
            activity.append(
                {
                    "kind": "tool",
                    "id": f"call-{index}",
                    "name": "create_file",
                    "completed": True,
                    "preview": "print('large')\n" * 2_000,
                }
            )
        server.session.tree.complete_turn(turn.id, "答" * 500_000, "思" * 500_000, activity)

        snapshot = server._snapshot()
        item = snapshot["transcript"][-1]
        assert _encoded_size(item) <= TRANSCRIPT_PAGE_BYTES
        server.emitter.emit("state.snapshot", request_id="snapshot", data=snapshot)
        assert max(len(line.encode("utf-8")) for line in output.getvalue().splitlines()) <= MAX_RUNTIME_FRAME_BYTES
    finally:
        server.close()


def test_runtime_snapshot_preserves_reasoning_and_tool_lifecycle(tmp_path: Path) -> None:
    from core.runtime_server import RuntimeServer

    server = RuntimeServer(
        agent=_RuntimeAgent(tmp_path),
        memory=_RuntimeMemory(),
        state_root=tmp_path / "state",
        config_path=tmp_path / "config.toml",
        input_stream=io.StringIO(),
        output_stream=io.StringIO(),
    )
    try:
        turn = server.session.tree.add_turn("remember this")
        server.session.tree.complete_turn(turn.id, "done", "I should use memory")
        server.session.tree.append_skill_exchange(
            turn.id,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "store_memory", "arguments": "{}"},
                    }
                ],
            },
        )
        server.session.tree.append_skill_exchange(
            turn.id,
            {"role": "tool", "tool_call_id": "call-1", "name": "store_memory", "content": "{}"},
        )

        item = server._snapshot()["transcript"][-1]

        assert item["activity"] == [
            {"kind": "reasoning", "text": "I should use memory"},
            {"kind": "tool", "id": "call-1", "name": "store_memory", "completed": True},
        ]
    finally:
        server.close()


def test_runtime_persists_reasoning_and_tools_in_event_order(tmp_path: Path) -> None:
    from core.runtime_server import RuntimeServer

    class SequencedAgent(_RuntimeAgent):
        def run_turn(self, **kwargs: object) -> SimpleNamespace:
            on_event = kwargs["on_event"]
            assert callable(on_event)
            on_event({"type": "reasoning_token", "text": "inspect first"})
            on_event({"type": "tool_call", "id": "one", "name": "inspect", "arguments": {}})
            on_event({"type": "tool_result", "id": "one", "name": "inspect", "result": {}})
            on_event({"type": "reasoning_token", "text": "edit next"})
            on_event({"type": "tool_call", "id": "two", "name": "edit", "arguments": {}})
            on_event({"type": "tool_result", "id": "two", "name": "edit", "result": {}})
            on_event({"type": "reasoning_token", "text": "verify last"})
            on_event({"type": "content_token", "text": "done"})
            return SimpleNamespace(
                status="done",
                content="done",
                reasoning="inspect firstedit nextverify last",
                error="",
                skill_exchanges=[],
            )

    server = RuntimeServer(
        agent=SequencedAgent(tmp_path),
        memory=_RuntimeMemory(),
        state_root=tmp_path / "state",
        config_path=tmp_path / "config.toml",
        input_stream=io.StringIO(),
        output_stream=io.StringIO(),
    )
    try:
        server._turn_start("turn", {"prompt": "do work"})
        assert server.turn_thread is not None
        server.turn_thread.join(timeout=2)

        assert server._snapshot()["transcript"][-1]["activity"] == [
            {"kind": "reasoning", "text": "inspect first"},
            {"kind": "tool", "id": "one", "name": "inspect", "completed": True},
            {"kind": "reasoning", "text": "edit next"},
            {"kind": "tool", "id": "two", "name": "edit", "completed": True},
            {"kind": "reasoning", "text": "verify last"},
        ]
    finally:
        server.close()


def test_runtime_persists_create_file_preview_from_tool_arguments(tmp_path: Path) -> None:
    from core.runtime_server import RuntimeServer

    class PreviewAgent(_RuntimeAgent):
        def run_turn(self, **kwargs: object) -> SimpleNamespace:
            on_event = kwargs["on_event"]
            assert callable(on_event)
            on_event(
                {
                    "type": "tool_call",
                    "stream_id": "stream-one",
                    "id": "call-one",
                    "name": "create_file",
                    "arguments": {"filepath": "demo.py", "content": "print('hello')\n"},
                }
            )
            on_event(
                {
                    "type": "tool_result",
                    "id": "call-one",
                    "name": "create_file",
                    "result": {
                        "ok": True,
                        "data": {
                            "filepath": "demo.py",
                            "content_preview": "print('hello')\n",
                            "content_preview_truncated": False,
                        },
                    },
                }
            )
            return SimpleNamespace(
                status="done",
                content="created",
                reasoning="",
                error="",
                skill_exchanges=[],
            )

    server = RuntimeServer(
        agent=PreviewAgent(tmp_path),
        memory=_RuntimeMemory(),
        state_root=tmp_path / "state",
        config_path=tmp_path / "config.toml",
        input_stream=io.StringIO(),
        output_stream=io.StringIO(),
    )
    try:
        server._turn_start("turn", {"prompt": "create it"})
        assert server.turn_thread is not None
        server.turn_thread.join(timeout=2)

        tool = server._snapshot()["transcript"][-1]["activity"][0]
        assert tool == {
            "kind": "tool",
            "id": "call-one",
            "stream_id": "stream-one",
            "name": "create_file",
            "completed": True,
            "filepath": "demo.py",
            "preview": "print('hello')",
            "preview_truncated": False,
            "language": "",
        }
    finally:
        server.close()


def test_runtime_marks_failed_tool_results_and_bounds_streamed_reasoning(tmp_path: Path) -> None:
    from core.runtime_server import MAX_PERSISTED_REASONING_CHARS, RuntimeServer

    class FailedToolAgent(_RuntimeAgent):
        def run_turn(self, **kwargs: object) -> SimpleNamespace:
            on_event = kwargs["on_event"]
            assert callable(on_event)
            for _ in range(600):
                on_event({"type": "reasoning_token", "text": "r" * 1_000})
            on_event({"type": "tool_call", "id": "bad", "name": "shell_command", "arguments": {}})
            on_event(
                {
                    "type": "tool_result",
                    "id": "bad",
                    "name": "shell_command",
                    "result": {"ok": False, "error": {"code": "E_POLICY"}},
                }
            )
            return SimpleNamespace(status="done", content="could not run it", reasoning="", error="", skill_exchanges=[])

    server = RuntimeServer(
        agent=FailedToolAgent(tmp_path),
        memory=_RuntimeMemory(),
        state_root=tmp_path / "state",
        config_path=tmp_path / "config.toml",
        input_stream=io.StringIO(),
        output_stream=io.StringIO(),
    )
    try:
        server._turn_start("turn", {"prompt": "run it"})
        assert server.turn_thread is not None
        server.turn_thread.join(timeout=3)

        turn = server.session.tree.active_path[-1]
        assert len(turn.reasoning_content) <= MAX_PERSISTED_REASONING_CHARS + 64
        tool = next(item for item in turn.activity_trace if item.get("kind") == "tool")
        assert tool["completed"] is True
        assert tool["failed"] is True
        snapshot_tool = next(item for item in server._snapshot()["transcript"][-1]["activity"] if item["kind"] == "tool")
        assert snapshot_tool["failed"] is True
    finally:
        server.close()


def test_completed_turn_snapshot_is_not_streaming_while_worker_is_alive(tmp_path: Path) -> None:
    from core.runtime_server import RuntimeServer

    server = RuntimeServer(
        agent=_RuntimeAgent(tmp_path),
        memory=_RuntimeMemory(),
        state_root=tmp_path / "state",
        config_path=tmp_path / "config.toml",
        input_stream=io.StringIO(),
        output_stream=io.StringIO(),
    )

    class LiveWorker:
        @staticmethod
        def is_alive() -> bool:
            return True

    try:
        server.turn_thread = LiveWorker()  # type: ignore[assignment]
        server.turn_request_id = "active-turn"
        assert server._snapshot()["streaming"] is True
        assert server._snapshot(streaming=False)["streaming"] is False
        server.turn_request_id = ""
        assert server._snapshot()["streaming"] is False
    finally:
        server.turn_thread = None
        server.close()


def test_turn_remains_active_until_terminal_events_are_written(tmp_path: Path) -> None:
    from core.runtime_server import RuntimeServer

    observed_request_ids: list[str] = []

    class InspectingStream(io.StringIO):
        server: RuntimeServer | None = None

        def write(self, value: str) -> int:
            if '"type":"turn.completed"' in value and self.server is not None:
                observed_request_ids.append(self.server.turn_request_id)
            return super().write(value)

    output = InspectingStream()
    server = RuntimeServer(
        agent=_RecordingRuntimeAgent(tmp_path),
        memory=_RuntimeMemory(),
        state_root=tmp_path / "state",
        config_path=tmp_path / "config.toml",
        input_stream=io.StringIO(),
        output_stream=output,
    )
    output.server = server
    try:
        server._turn_start("active-turn", {"prompt": "hello"})
        assert server.turn_thread is not None
        server.turn_thread.join(timeout=2)

        assert observed_request_ids == ["active-turn"]
        assert server.turn_request_id == ""
    finally:
        server.close()


def test_runtime_sends_each_current_user_turn_to_the_agent(tmp_path: Path) -> None:
    from core.runtime_server import RuntimeServer

    agent = _RecordingRuntimeAgent(tmp_path)
    output = io.StringIO()
    server = RuntimeServer(
        agent=agent,
        memory=_RuntimeMemory(),
        state_root=tmp_path / "state",
        config_path=tmp_path / "config.toml",
        input_stream=io.StringIO(),
        output_stream=output,
    )
    try:
        server._turn_start("turn-one", {"prompt": "first"})
        assert server.turn_thread is not None
        server.turn_thread.join(timeout=2)
        assert not server.turn_thread.is_alive()

        server._turn_start("turn-two", {"prompt": "second"})
        assert server.turn_thread is not None
        server.turn_thread.join(timeout=2)
        assert not server.turn_thread.is_alive()

        assert [[message["content"] for message in history] for history in agent.histories] == [
            ["first"],
            ["first", "reply:first", "second"],
        ]
        completed = [frame for frame in map(json.loads, output.getvalue().splitlines()) if frame["type"] == "turn.completed"]
        assert [frame["data"]["content"] for frame in completed] == ["reply:first", "reply:second"]
    finally:
        server.close()


def test_clear_resets_persisted_and_frontend_state(tmp_path: Path) -> None:
    from core.runtime_server import RuntimeServer
    from core.ui_commands import execute_ui_command

    server = RuntimeServer(
        agent=_RuntimeAgent(tmp_path),
        memory=_RuntimeMemory(),
        state_root=tmp_path / "state",
        config_path=tmp_path / "config.toml",
        input_stream=io.StringIO(),
        output_stream=io.StringIO(),
    )
    try:
        turn = server.session.tree.add_turn("hello")
        server.session.tree.complete_turn(turn.id, "hi")
        server.pending_attachments.append((str(tmp_path / "note.txt"), "text"))

        result = execute_ui_command(server, "/clear")

        assert result["action"] == "clear"
        assert result["state_changed"] is True
        assert server._snapshot()["transcript"] == []
        assert server._snapshot()["pending_attachments"] == []
    finally:
        server.close()


def test_empty_command_states_have_visible_fallback_messages(tmp_path: Path) -> None:
    from core.runtime_server import RuntimeServer
    from core.ui_commands import execute_ui_command

    server = RuntimeServer(
        agent=_RuntimeAgent(tmp_path),
        memory=_RuntimeMemory(),
        state_root=tmp_path / "state",
        config_path=tmp_path / "config.toml",
        input_stream=io.StringIO(),
        output_stream=io.StringIO(),
    )
    try:
        assert execute_ui_command(server, "/branches")["lines"] == ["No child branches"]
        assert execute_ui_command(server, "/detach")["lines"] == ["No pending attachments"]
        assert execute_ui_command(server, "/context")["lines"] == ["Context usage is not available yet"]
    finally:
        server.close()


def test_shortcuts_are_distinct_from_full_command_help(tmp_path: Path) -> None:
    from core.runtime_server import RuntimeServer
    from core.ui_commands import execute_ui_command

    server = RuntimeServer(
        agent=_RuntimeAgent(tmp_path),
        memory=_RuntimeMemory(),
        state_root=tmp_path / "state",
        config_path=tmp_path / "config.toml",
        input_stream=io.StringIO(),
        output_stream=io.StringIO(),
    )
    try:
        help_text = "\n".join(execute_ui_command(server, "/help")["lines"])
        shortcuts_text = "\n".join(execute_ui_command(server, "/shortcuts")["lines"])

        assert "/sessions" in help_text
        assert "Ctrl+Shift+K" in shortcuts_text
        assert "/sessions" not in shortcuts_text
    finally:
        server.close()


def test_branch_command_snapshot_immediately_exposes_the_pending_branch_name(tmp_path: Path) -> None:
    from core.runtime_server import RuntimeServer

    output = io.StringIO()
    server = RuntimeServer(
        agent=_RuntimeAgent(tmp_path),
        memory=_RuntimeMemory(),
        state_root=tmp_path / "state",
        config_path=tmp_path / "config.toml",
        input_stream=io.StringIO(),
        output_stream=output,
    )
    try:
        server._command_execute("branch", {"command": "/branch X"})

        frames = [json.loads(line) for line in output.getvalue().splitlines()]
        snapshot = next(frame["data"] for frame in frames if frame["type"] == "state.snapshot")
        assert snapshot["session"]["branch_name"] == "X"
        assert snapshot["session"]["pending_branch"] is True
        assert snapshot["session"]["pending_branch_label"] == "X"

        server._command_execute("unbranch", {"command": "/unbranch"})
        frames = [json.loads(line) for line in output.getvalue().splitlines()]
        snapshots = [frame["data"] for frame in frames if frame["type"] == "state.snapshot"]
        assert snapshots[-1]["session"]["branch_name"] == "root"
        assert snapshots[-1]["session"]["pending_branch"] is False
    finally:
        server.close()


def test_status_refresh_emits_model_metadata(tmp_path: Path) -> None:
    from core.runtime_server import RuntimeServer

    output = io.StringIO()
    server = RuntimeServer(
        agent=_RuntimeAgent(tmp_path),
        memory=_RuntimeMemory(),
        state_root=tmp_path / "state",
        config_path=tmp_path / "config.toml",
        input_stream=io.StringIO(),
        output_stream=output,
    )
    try:
        server._status_refresh("status", {})
        frames = [json.loads(line) for line in output.getvalue().splitlines()]
        changed = next(frame for frame in frames if frame["type"] == "status.changed")
        assert changed["data"] == {
            "state": "online",
            "detail": "",
            "model_name": "test-model",
            "context_window": 32_768,
            "endpoint": "http://127.0.0.1:8080/v1/models",
        }
    finally:
        server.close()


def test_command_catalog_preserves_every_public_tui_command() -> None:
    commands = {item["command"].split()[0] for item in command_catalog()}
    assert {
        "/help",
        "/shortcuts",
        "/details",
        "/think",
        "/mode",
        "/clear",
        "/sessions",
        "/rename",
        "/save",
        "/file",
        "/detach",
        "/quit",
        "/branch",
        "/unbranch",
        "/branches",
        "/switch",
        "/tree",
        "/skills",
        "/reload",
        "/doctor",
        "/health",
        "/skill-on",
        "/skill-off",
        "/skill-unload",
        "/skill-unload-all",
        "/skill-reload",
        "/skill-info",
        "/memory-stats",
        "/context",
        "/audit",
        "/project-tree",
        "/theme",
        "/config",
        "/report",
        "/code",
    } <= commands


def test_builtin_themes_are_framework_neutral_and_complete() -> None:
    reload_themes()
    ids = available_theme_ids()
    assert "catppuccin-mocha" in ids
    payload = theme_payload("catppuccin-mocha")
    assert payload["colors"]["accent"] == "#cba6f7"
    assert "ratatui" not in payload


def test_optional_ratatui_theme_overrides_are_normalized() -> None:
    payload = _read(
        json.dumps(
            {
                "id": "custom",
                "title": "Custom",
                "description": "test",
                "theme": {"background": "#000000"},
                "colors": {"accent": "#ffffff"},
                "ratatui": {
                    "border_set": "double",
                    "syntax_theme": "base16-ocean.dark",
                    "styles": {
                        "selection": {
                            "foreground": "#ffffff",
                            "background": "#222222",
                            "modifiers": ["bold", "invalid"],
                        },
                        "future-style": {"foreground": "#000000"},
                    },
                    "future_field": True,
                },
            }
        ),
        source="test",
    )
    assert payload["ratatui"]["border_set"] == "double"
    assert payload["ratatui"]["styles"]["selection"]["modifiers"] == ["bold"]
    assert "future-style" not in payload["ratatui"]["styles"]
    assert payload["_warnings"]


def test_invalid_theme_shape_is_rejected() -> None:
    with pytest.raises(ThemeLoadError):
        _read("[]", source="test")
