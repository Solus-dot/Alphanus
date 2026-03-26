from __future__ import annotations

import json
from pathlib import Path

from agent.classifier import TurnClassifier
from agent.llm_client import LLMClient
from agent.telemetry import TelemetryEmitter, configure_logging
from core.memory import VectorMemory
from core.skills import SkillRuntime
from core.workspace import WorkspaceManager


def _runtime(tmp_path: Path) -> SkillRuntime:
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "workspace-ops").mkdir(parents=True)
    (skills / "workspace-ops" / "SKILL.md").write_text(
        """
---
name: workspace-ops
description: workspace
version: 1.0.0
tools:
  allowed-tools:
    - create_directory
    - create_file
    - create_files
---
workspace
""".strip(),
        encoding="utf-8",
    )
    (skills / "workspace-ops" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "create_directory": {
    "capability": "workspace_write",
    "description": "Create directory",
    "parameters": {
      "type": "object",
      "properties": {"path": {"type": "string"}},
      "required": ["path"]
    }
  }
}

def execute(tool_name, args, env):
    return {"ok": True, "data": {"filepath": env.workspace.create_directory(args["path"])}, "error": None, "meta": {}}
""".strip(),
        encoding="utf-8",
    )
    return SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )


def test_telemetry_emits_json_lines_to_configured_log_file(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    logger = configure_logging({"logging": {"format": "json", "path": str(log_path), "level": "INFO"}})

    TelemetryEmitter(logger).emit("turn_classified", source="model", time_sensitive=True)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "turn_classified"
    assert payload["payload"]["source"] == "model"
    assert payload["payload"]["time_sensitive"] is True


def test_console_logging_suppresses_info_telemetry_but_keeps_file_events(tmp_path: Path, capsys) -> None:
    log_path = tmp_path / "events.jsonl"
    logger = configure_logging({"logging": {"format": "plain", "path": str(log_path), "level": "INFO"}})

    TelemetryEmitter(logger).emit("http_stream", phase="start")
    logger.warning("visible warning")

    captured = capsys.readouterr()
    assert "http_stream" not in captured.err
    assert "visible warning" in captured.err

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2
    events = [json.loads(line) for line in lines]
    assert any(item.get("event") == "http_stream" for item in events)


def test_classifier_skips_model_call_for_plain_local_workspace_task(mocker, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": True}}
    llm_client = LLMClient(cfg)
    classifier = TurnClassifier(cfg, runtime, llm_client)
    ctx = classifier.build_skill_context(
        "Make a landing page for a bakery using html css and javascript and save it in a folder called arjun",
        [],
        [],
        [],
    )

    mocker.patch.object(llm_client, "call_with_retry", side_effect=AssertionError("classifier model should not be called"))
    classifier.call_with_retry = llm_client.call_with_retry

    classification = classifier.classify(ctx)

    assert classification.used_model is False
    assert classification.workspace_scaffold_action is True
    assert classification.workspace_materialization_target == 3


def test_classifier_counts_opaque_artifact_request_as_materialization_target(mocker, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    cfg = {"agent": {"enable_structured_classification": True}}
    llm_client = LLMClient(cfg)
    classifier = TurnClassifier(cfg, runtime, llm_client)
    ctx = classifier.build_skill_context(
        "Create proposal.docx in a folder called drafts",
        [],
        [],
        [],
    )

    mocker.patch.object(llm_client, "call_with_retry", side_effect=AssertionError("classifier model should not be called"))
    classifier.call_with_retry = llm_client.call_with_retry

    classification = classifier.classify(ctx)

    assert classification.used_model is False
    assert classification.workspace_materialization_target >= 1
