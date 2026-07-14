from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.types import ModelStatus
from core.runtime_protocol import (
    MAX_RUNTIME_FRAME_BYTES,
    RUNTIME_PROTOCOL_VERSION,
    RuntimeEmitter,
    RuntimeProtocolError,
    decode_runtime_frame,
)
from core.themes import ThemeLoadError, _read, available_theme_ids, reload_themes, theme_payload
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
        return ModelStatus(state="unknown", last_error="not probed")


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
        assert snapshot["transcript_offset"] == 10
        assert len(snapshot["transcript"]) == 100
        assert str(snapshot["transcript"][0]["user"]).startswith("10:")
        assert len(json.dumps(snapshot, ensure_ascii=False).encode("utf-8")) < MAX_RUNTIME_FRAME_BYTES
    finally:
        server.close()


def test_command_catalog_preserves_every_public_tui_command() -> None:
    commands = {item["command"].split()[0] for item in command_catalog()}
    assert {
        "/help", "/shortcuts", "/details", "/think", "/mode", "/clear", "/sessions", "/rename", "/save",
        "/file", "/detach", "/quit", "/branch", "/unbranch", "/branches", "/switch", "/tree", "/skills",
        "/reload", "/doctor", "/health", "/skill-on", "/skill-off", "/skill-unload", "/skill-unload-all",
        "/skill-reload", "/skill-info", "/memory-stats", "/context", "/audit", "/project-tree", "/theme",
        "/config", "/report", "/code",
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
