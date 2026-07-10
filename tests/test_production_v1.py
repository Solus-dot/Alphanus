from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from agent.telemetry import TelemetryEmitter, configure_logging
from core.attachments import MAX_ATTACHMENT_BYTES, build_content, classify_attachment
from core.headless_protocol import parse_jsonl_request
from core.sandbox import MAX_SANDBOX_OUTPUT_BYTES, SandboxCommand, SandboxConfig, SandboxRunner


@pytest.mark.security
def test_attachment_limits_are_enforced_before_encoding(tmp_path: Path) -> None:
    oversized = tmp_path / "large.txt"
    with oversized.open("wb") as handle:
        handle.truncate(MAX_ATTACHMENT_BYTES + 1)
    with pytest.raises(ValueError, match="exceeds"):
        classify_attachment(str(oversized))


@pytest.mark.security
def test_attachment_symlinks_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlinks"):
        build_content("inspect", [(str(link), "text")])


@pytest.mark.security
def test_sandbox_output_is_bounded(tmp_path: Path) -> None:
    result = SandboxRunner().run(
        SandboxCommand(
            command="python3 -c \"import sys; sys.stdout.write('x' * 10000000); sys.stderr.write('y' * 10000000)\"",
            cwd=tmp_path,
            project_root=tmp_path,
            timeout_s=5,
            config=SandboxConfig(mode="danger-full-access"),
        )
    )
    assert len(result["stdout"].encode()) <= MAX_SANDBOX_OUTPUT_BYTES
    assert len(result["stderr"].encode()) <= MAX_SANDBOX_OUTPUT_BYTES
    assert result["stdout_truncated"] is True
    assert result["stderr_truncated"] is True


def test_telemetry_redacts_payload_and_rotates(tmp_path: Path) -> None:
    path = tmp_path / "runtime.jsonl"
    logger = configure_logging({"logging": {"path": str(path), "format": "json", "max_bytes": 1024, "backup_count": 2}})
    TelemetryEmitter(logger).emit("request", payload={"prompt": "private", "authorization": "Bearer secret"}, content="answer")
    for handler in logger.handlers:
        handler.flush()
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["payload"]["payload"] == "[redacted]"
    assert record["payload"]["content"] == "[redacted]"
    logging.shutdown()


def test_jsonl_input_requires_protocol_version() -> None:
    assert parse_jsonl_request('{"schema_version":1,"prompt":"hello"}')["prompt"] == "hello"
    with pytest.raises(ValueError, match="schema_version"):
        parse_jsonl_request('{"prompt":"hello"}')
