from __future__ import annotations

from pathlib import Path

from core.memory import VectorMemory
from core.skills import SkillContext, SkillRuntime
from core.workspace import WorkspaceManager


def test_skill_load_select_and_execute(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "s1").mkdir(parents=True)

    (skills / "s1" / "SKILL.md").write_text(
        """
---
name: s1
description: test
version: 1.0.0
tools:
  allowed-tools:
    - create_file
    - read_file
triggers:
  keywords:
    - write
  file_ext:
    - .py
---
Use carefully
""".strip(),
        encoding="utf-8",
    )
    (skills / "s1" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "create_file": {
    "capability": "workspace_write",
    "description": "Create file",
    "parameters": {
      "type": "object",
      "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}},
      "required": ["filepath", "content"]
    }
  },
  "read_file": {
    "capability": "workspace_read",
    "description": "Read file",
    "parameters": {
      "type": "object",
      "properties": {"filepath": {"type": "string"}},
      "required": ["filepath"]
    }
  }
}

def execute(tool_name, args, env):
    if tool_name == "create_file":
        path = env.workspace.create_file(args["filepath"], args["content"])
        return {"ok": True, "data": {"filepath": path}, "error": None, "meta": {}}
    if tool_name == "read_file":
        content = env.workspace.read_file(args["filepath"])
        return {"ok": True, "data": {"content": content}, "error": None, "meta": {}}
    return {"ok": False, "data": None, "error": {"code": "E_UNSUPPORTED", "message": "nope"}, "meta": {}}
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
        debug=True,
    )

    ctx = SkillContext(
        user_input="write a file",
        branch_labels=[],
        attachments=["main.py"],
        workspace_root=str(ws),
        memory_hits=[],
    )

    selected = runtime.select_skills(ctx)
    assert selected
    assert selected[0].id == "s1"

    out = runtime.execute_tool_call(
        "create_file",
        {"filepath": "hello.txt", "content": "hi"},
        selected=selected,
        ctx=ctx,
    )
    assert out["ok"] is True


def test_fail_closed_when_tool_not_allowed(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "s1").mkdir(parents=True)

    (skills / "s1" / "SKILL.md").write_text(
        """
---
name: s1
description: test
version: 1.0.0
tools:
  allowed-tools:
    - create_file
---
Workspace only
""".strip(),
        encoding="utf-8",
    )
    (skills / "s1" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "create_file": {
    "capability": "workspace_write",
    "description": "Create file",
    "parameters": {
      "type": "object",
      "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}},
      "required": ["filepath", "content"]
    }
  },
  "read_file": {
    "capability": "workspace_read",
    "description": "Read file",
    "parameters": {
      "type": "object",
      "properties": {"filepath": {"type": "string"}},
      "required": ["filepath"]
    }
  }
}

def execute(tool_name, args, env):
    if tool_name == "create_file":
        path = env.workspace.create_file(args["filepath"], args["content"])
        return {"ok": True, "data": {"filepath": path}, "error": None, "meta": {}}
    if tool_name == "read_file":
        content = env.workspace.read_file(args["filepath"])
        return {"ok": True, "data": {"content": content}, "error": None, "meta": {}}
    return {"ok": False, "data": None, "error": {"code": "E_UNSUPPORTED", "message": "nope"}, "meta": {}}
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    ctx = SkillContext(
        user_input="read a file",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )
    selected = runtime.select_skills(ctx)
    out = runtime.execute_tool_call("read_file", {"filepath": "a.txt"}, selected=selected, ctx=ctx)
    assert out["ok"] is False
    assert out["error"]["code"] == "E_UNSUPPORTED"


def test_agentskill_name_must_match_directory(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "s1").mkdir(parents=True)

    (skills / "s1" / "SKILL.md").write_text(
        """
---
name: wrong-name
description: test
version: 1.0.0
---
Body
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    assert runtime.list_skills() == []


def test_agentskill_required_tools_missing_fails_load(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "s1").mkdir(parents=True)

    (skills / "s1" / "SKILL.md").write_text(
        """
---
name: s1
description: test
version: 1.0.0
tools:
  required-tools:
    - create_file
---
Body
""".strip(),
        encoding="utf-8",
    )
    (skills / "s1" / "tools.py").write_text(
        """
TOOL_SPECS = {}

def execute(tool_name, args, env):
    return {"ok": False, "data": None, "error": {"code": "E_UNSUPPORTED", "message": "nope"}, "meta": {}}
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    assert runtime.list_skills() == []


def test_legacy_skill_toml_is_not_loaded(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "legacy").mkdir(parents=True)

    (skills / "legacy" / "skill.toml").write_text(
        """
id = "legacy"
name = "legacy"
version = "1.0.0"
description = "legacy"
enabled = true
priority = 50
""".strip(),
        encoding="utf-8",
    )
    (skills / "legacy" / "prompt.md").write_text("legacy", encoding="utf-8")

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    assert runtime.list_skills() == []


def test_command_definition_tool_executes(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "echo-skill" / "scripts").mkdir(parents=True)

    (skills / "echo-skill" / "SKILL.md").write_text(
        """
---
name: echo-skill
description: echo
version: 1.0.0
tools:
  allowed-tools:
    - echo_text
  definitions:
    - name: echo_text
      capability: utility_echo
      description: Echo text.
      command: python3 scripts/echo_text.py
      parameters:
        type: object
        properties:
          text:
            type: string
        required:
          - text
---
Echo
""".strip(),
        encoding="utf-8",
    )
    (skills / "echo-skill" / "scripts" / "echo_text.py").write_text(
        """
import json
import os
import sys

def main():
    raw = os.getenv("ALPHANUS_TOOL_ARGS_JSON") or sys.stdin.read() or "{}"
    args = json.loads(raw)
    out = {"ok": True, "data": {"text": args["text"]}, "error": None, "meta": {}}
    print(json.dumps(out))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("echo-skill")
    assert skill is not None

    ctx = SkillContext(
        user_input="echo this",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )
    out = runtime.execute_tool_call("echo_text", {"text": "hello"}, selected=[skill], ctx=ctx)
    assert out["ok"] is True
    assert out["data"]["text"] == "hello"

    out_raw = runtime.execute_tool_call("echo_text", {"_raw": '{"text":"raw-hello"}'}, selected=[skill], ctx=ctx)
    assert out_raw["ok"] is True
    assert out_raw["data"]["text"] == "raw-hello"


def test_frontmatter_metadata_format_executes(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "meta-skill" / "scripts").mkdir(parents=True)

    (skills / "meta-skill" / "SKILL.md").write_text(
        """
---
name: meta-skill
description: metadata format
allowed-tools: echo_text
metadata:
  version: "1.0.0"
  categories:
    - custom
  tags:
    - echo
  triggers:
    keywords:
      - echo
  tools:
    definitions:
      - name: echo_text
        capability: utility_echo
        description: Echo text.
        command: python3 scripts/echo_text.py
        parameters:
          type: object
          properties:
            text:
              type: string
          required:
            - text
---
Echo
""".strip(),
        encoding="utf-8",
    )
    (skills / "meta-skill" / "scripts" / "echo_text.py").write_text(
        """
import json
import os
import sys

def main():
    raw = os.getenv("ALPHANUS_TOOL_ARGS_JSON") or sys.stdin.read() or "{}"
    args = json.loads(raw)
    print(json.dumps({"text": args["text"]}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("meta-skill")
    assert skill is not None

    ctx = SkillContext(
        user_input="echo this",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )
    out = runtime.execute_tool_call("echo_text", {"text": "hello"}, selected=[skill], ctx=ctx)
    assert out["ok"] is True
    assert out["data"]["text"] == "hello"
