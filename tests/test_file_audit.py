from __future__ import annotations

import json
from pathlib import Path

from agent.file_audit import build_file_audit_from_evidence, build_file_audit_from_skill_exchanges
from core.types import ToolExecutionRecord


def test_build_file_audit_from_evidence_tracks_file_changes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    rows = build_file_audit_from_evidence(
        [
            ToolExecutionRecord(
                name="create_file",
                args={"filepath": "src/app.py"},
                result={
                    "ok": True,
                    "data": {"filepath": str(workspace / "src/app.py"), "bytes_written": 12, "line_count": 2},
                },
            ),
            ToolExecutionRecord(
                name="edit_file",
                args={"filepath": "src/app.py"},
                result={
                    "ok": True,
                    "data": {"filepath": str(workspace / "src/app.py"), "changed": True, "changed_lines": 3},
                },
            ),
            ToolExecutionRecord(
                name="move_path",
                args={"source_path": "src/app.py", "destination_path": "src/main.py"},
                result={
                    "ok": True,
                    "data": {
                        "source_path": str(workspace / "src/app.py"),
                        "destination_path": str(workspace / "src/main.py"),
                        "kind": "file",
                    },
                },
            ),
            ToolExecutionRecord(
                name="delete_path",
                args={"path": "build"},
                result={
                    "ok": True,
                    "data": {"filepath": str(workspace / "build"), "kind": "directory", "file_count": 4},
                },
            ),
            ToolExecutionRecord(
                name="shell_command",
                args={"command": "make format"},
                result={
                    "ok": True,
                    "data": {"command": "make format", "cwd": str(workspace), "returncode": 0},
                    "meta": {"workspace_changed": True, "duration_ms": 17},
                },
            ),
        ],
        workspace_root=workspace,
    )

    assert [row["action"] for row in rows] == ["created", "edited", "moved", "deleted", "workspace_changed"]
    assert rows[0] == {
        "action": "created",
        "tool": "create_file",
        "status": "success",
        "path": "src/app.py",
        "kind": "file",
        "bytes": 12,
        "lines": 2,
    }
    assert rows[1]["path"] == "src/app.py"
    assert rows[1]["changed_lines"] == 3
    assert rows[2]["from"] == "src/app.py"
    assert rows[2]["to"] == "src/main.py"
    assert rows[3]["path"] == "build"
    assert rows[3]["file_count"] == 4
    assert rows[4]["paths_known"] is False
    assert rows[4]["command"] == "make format"


def test_file_audit_ignores_reads_and_unchanged_shell_commands(tmp_path: Path) -> None:
    rows = build_file_audit_from_evidence(
        [
            ToolExecutionRecord(
                name="read_file",
                args={"filepath": "src/app.py"},
                result={"ok": True, "data": {"filepath": "src/app.py"}},
            ),
            ToolExecutionRecord(
                name="shell_command",
                args={"command": "pwd"},
                result={"ok": True, "data": {"command": "pwd"}, "meta": {"workspace_changed": False}},
            ),
        ],
        workspace_root=tmp_path,
    )

    assert rows == []


def test_file_audit_keeps_failed_file_operation_targets(tmp_path: Path) -> None:
    rows = build_file_audit_from_evidence(
        [
            ToolExecutionRecord(
                name="edit_file",
                args={"filepath": "src/app.py"},
                result={"ok": False, "error": {"code": "E_VALIDATION", "message": "missing marker"}},
            )
        ],
        workspace_root=tmp_path,
    )

    assert rows == [
        {
            "action": "edited",
            "tool": "edit_file",
            "status": "failed",
            "path": "src/app.py",
            "error": "missing marker",
            "error_code": "E_VALIDATION",
        }
    ]


def test_file_audit_reconstructs_args_from_persisted_skill_exchanges(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    rows = build_file_audit_from_skill_exchanges(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "edit_file", "arguments": json.dumps({"filepath": "src/app.py"})},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "edit_file",
                "tool_call_id": "call_1",
                "content": json.dumps({"ok": False, "error": {"message": "stale marker"}}),
            },
        ],
        workspace_root=workspace,
    )

    assert rows == [
        {
            "action": "edited",
            "tool": "edit_file",
            "status": "failed",
            "path": "src/app.py",
            "error": "stale marker",
        }
    ]
