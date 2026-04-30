from __future__ import annotations
from pathlib import Path

import pytest

import core.skills as skills_module
from core.memory import LexicalMemory
from core.skills import SkillContext, SkillRuntime
from core.workspace import WorkspaceManager


def _tool_names(runtime: SkillRuntime, selected, ctx: SkillContext | None = None) -> list[str]:
    return [tool["function"]["name"] for tool in runtime.tools_for_turn(selected, ctx=ctx)]


def _always_available_tool_names() -> set[str]:
    return {"request_user_input", "skill_view", "skills_list"}


def test_runtime_minimal_profile_restricts_optional_tools(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    runtime = SkillRuntime(
        skills_dir="skills",
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={"runtime": {"profile": "minimal", "ask_user_tool": True}},
        debug=True,
    )
    ctx = SkillContext(
        user_input="search the latest news",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )
    runtime.skill_view("search-ops", "", ctx)
    runtime.skill_view("workspace-ops", "", ctx)
    ctx.loaded_skill_ids = ["search-ops", "workspace-ops"]
    selected = runtime.select_skills(ctx)
    names = set(runtime.allowed_tool_names(selected, ctx=ctx))

    assert "create_file" in names
    assert "read_file" in names
    assert "web_search" not in names
    assert "fetch_url" not in names
    assert "skills_list" in names
    assert "skill_view" in names


def test_permission_profile_safe_allows_read_only_workspace_tools(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    runtime = SkillRuntime(
        skills_dir="skills",
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={"capabilities": {"permission_profile": "safe"}},
        debug=True,
    )
    selected = runtime.skills_by_ids(["workspace-ops", "search-ops", "shell-ops"])
    names = set(runtime.allowed_tool_names(selected))

    assert "read_file" in names
    assert "list_files" in names
    assert "workspace_tree" in names
    assert "create_file" not in names
    assert "delete_path" not in names
    assert "web_search" not in names
    assert "shell_command" not in names


def test_permission_profile_workspace_allows_workspace_mutation_but_blocks_web_and_shell(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    runtime = SkillRuntime(
        skills_dir="skills",
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={"capabilities": {"permission_profile": "workspace"}},
        debug=True,
    )
    selected = runtime.skills_by_ids(["workspace-ops", "search-ops", "shell-ops"])
    names = set(runtime.allowed_tool_names(selected))

    assert "create_file" in names
    assert "edit_file" in names
    assert "delete_path" in names
    assert "run_checks" in names
    assert "web_search" not in names
    assert "fetch_url" not in names
    assert "shell_command" not in names


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
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        debug=True,
    )

    ctx = SkillContext(
        user_input="write a file",
        branch_labels=[],
        attachments=["main.py"],
        workspace_root=str(ws),
        memory_hits=[],
        explicit_skill_id="s1",
    )

    assert runtime.select_skills(ctx) == []
    loaded = runtime.skill_view("s1", "", ctx)
    assert loaded["skill_id"] == "s1"
    assert loaded["loaded"] is True
    selected = runtime.select_skills(ctx)
    assert [skill.id for skill in selected] == ["s1"]

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
    - write_blob
---
Workspace only
""".strip(),
        encoding="utf-8",
    )
    (skills / "s1" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "write_blob": {
    "capability": "workspace_write",
    "description": "Create file",
    "parameters": {
      "type": "object",
      "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}},
      "required": ["filepath", "content"]
    }
  },
  "read_blob": {
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
    if tool_name == "write_blob":
        path = env.workspace.create_file(args["filepath"], args["content"])
        return {"ok": True, "data": {"filepath": path}, "error": None, "meta": {}}
    if tool_name == "read_blob":
        content = env.workspace.read_file(args["filepath"])
        return {"ok": True, "data": {"content": content}, "error": None, "meta": {}}
    return {"ok": False, "data": None, "error": {"code": "E_UNSUPPORTED", "message": "nope"}, "meta": {}}
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    ctx = SkillContext(
        user_input="read a file",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )
    selected = runtime.select_skills(ctx)
    out = runtime.execute_tool_call("read_blob", {"filepath": "a.txt"}, selected=selected, ctx=ctx)
    assert out["ok"] is False
    assert out["error"]["code"] == "E_UNSUPPORTED"


def test_tools_for_turn_requires_selected_skill_for_native_tools(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "workspace-ops").mkdir(parents=True)
    (skills / "search-ops").mkdir(parents=True)

    (skills / "workspace-ops" / "SKILL.md").write_text(
        """
---
name: workspace-ops
description: workspace core tools
version: 1.0.0
tools:
  allowed-tools:
    - create_file
    - read_file
---
Workspace core.
""".strip(),
        encoding="utf-8",
    )
    (skills / "workspace-ops" / "tools.py").write_text(
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
    (skills / "search-ops" / "SKILL.md").write_text(
        """
---
name: search-ops
description: search the web
version: 1.0.0
tools:
  allowed-tools:
    - web_search
---
Search skill.
""".strip(),
        encoding="utf-8",
    )
    (skills / "search-ops" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "web_search": {
    "capability": "web_search",
    "description": "Search the web",
    "parameters": {
      "type": "object",
      "properties": {"query": {"type": "string"}},
      "required": ["query"]
    }
  }
}

def execute(tool_name, args, env):
    return {"ok": True, "data": {"query": args.get("query", "")}, "error": None, "meta": {}}
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    runtime_only_tool_names = set(_tool_names(runtime, []))
    assert runtime_only_tool_names == _always_available_tool_names()

    workspace_skill = runtime.get_skill("workspace-ops")
    search_skill = runtime.get_skill("search-ops")
    assert workspace_skill is not None and search_skill is not None
    merged_tool_names = set(_tool_names(runtime, [workspace_skill, search_skill]))
    assert merged_tool_names == {
        "create_file",
        "read_file",
        "request_user_input",
        "skill_view",
        "skills_list",
        "web_search",
    }


def test_core_tool_names_for_turn_include_model_exposed_core_tools_without_selected_skill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
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
description: workspace core tools
version: 1.0.0
tools:
  allowed-tools:
    - create_file
---
Workspace core.
""".strip(),
        encoding="utf-8",
    )
    (skills / "workspace-ops" / "tools.py").write_text(
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
  }
}

def execute(tool_name, args, env):
    return {"ok": True, "data": {"filepath": args.get("filepath", "")}, "error": None, "meta": {}}
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    ctx = SkillContext(
        user_input="save a summary to notes.md",
        branch_labels=[],
        attachments=["image.png"],
        workspace_root=str(ws),
        memory_hits=[],
    )

    monkeypatch.setattr(
        runtime,
        "model_exposed_tool_names",
        lambda: sorted(_always_available_tool_names() | {"create_file"}),
    )
    runtime._tools_schema_cache.clear()

    assert runtime.optional_tool_names([], ctx=ctx) == []
    assert runtime.core_tool_names_for_turn([], ctx=ctx) == ["create_file"]


def test_tools_for_turn_includes_generic_script_runner_for_selected_script_skill(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "script-only" / "scripts").mkdir(parents=True)

    (skills / "script-only" / "SKILL.md").write_text(
        """
---
name: script-only
description: ships a helper script for structured tasks
version: 1.0.0
---
Use the helper script when available.
""".strip(),
        encoding="utf-8",
    )
    (skills / "script-only" / "scripts" / "helper.py").write_text(
        "if __name__ == '__main__':\n    print('ok')\n",
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    skill = runtime.get_skill("script-only")
    assert skill is not None
    assert set(_tool_names(runtime, [skill])) == (_always_available_tool_names() | {"run_skill"})


def test_tools_for_turn_includes_generic_entrypoint_runner_for_structured_skill(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "report-pdf" / "scripts").mkdir(parents=True)

    (skills / "report-pdf" / "SKILL.md").write_text(
        """
---
name: report-pdf
description: Create report PDFs with a declared workflow.
produces:
  - .pdf
execution:
  entrypoints:
    - name: create_report
      description: Create a PDF report artifact.
      intents: [create]
      produces: [.pdf]
      command: python3 {skill_root}/scripts/create_report.py {workspace_root}/{filename}
      parameters:
        type: object
        properties:
          filename:
            type: string
        required:
          - filename
---
Create PDFs with the declared entrypoint.
""".strip(),
        encoding="utf-8",
    )
    (skills / "report-pdf" / "scripts" / "create_report.py").write_text(
        "from pathlib import Path\nimport sys\nPath(sys.argv[1]).write_text('pdf', encoding='utf-8')\n",
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("report-pdf")
    assert skill is not None
    ctx = SkillContext(
        user_input="Create a report PDF and save it as report.pdf",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )

    assert set(_tool_names(runtime, [skill], ctx=ctx)) == (_always_available_tool_names() | {"run_skill"})


def test_root_level_skill_helper_script_is_exposed(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "research-helper").mkdir(parents=True)

    (skills / "research-helper" / "SKILL.md").write_text(
        """
---
name: research-helper
description: ships a helper validator at the skill root
version: 1.0.0
---
Use validate_json.py when available.
""".strip(),
        encoding="utf-8",
    )
    (skills / "research-helper" / "validate_json.py").write_text(
        "if __name__ == '__main__':\n    print('ok')\n",
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    skill = runtime.get_skill("research-helper")
    assert skill is not None
    assert set(_tool_names(runtime, [skill])) == (_always_available_tool_names() | {"run_skill"})
    assert runtime._reported_skill_scripts(skill) == ["validate_json.py"]


def test_entrypoint_skill_stays_runnable_without_artifact_heuristics(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "preview-pdf" / "scripts").mkdir(parents=True)

    (skills / "preview-pdf" / "SKILL.md").write_text(
        """
---
name: preview-pdf
description: Build binary report artifacts.
produces:
  - .pdf
execution:
  entrypoints:
    - name: create_preview
      description: Create the PDF artifact for later review.
      intents: [create]
      produces: [.pdf]
      command: python3 {skill_root}/scripts/create_preview.py {workspace_root}/{filename}
      parameters:
        type: object
        properties:
          filename:
            type: string
        required:
          - filename
---
Create PDFs through the declared entrypoint.
""".strip(),
        encoding="utf-8",
    )
    (skills / "preview-pdf" / "scripts" / "create_preview.py").write_text(
        "from pathlib import Path\nimport sys\nPath(sys.argv[1]).write_text('pdf', encoding='utf-8')\n",
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("preview-pdf")
    assert skill is not None
    ctx = SkillContext(
        user_input="Create preview.pdf and review it after generation",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )

    assert set(_tool_names(runtime, [skill], ctx=ctx)) == (_always_available_tool_names() | {"run_skill"})


def test_core_tool_executes_with_selected_skill(tmp_path: Path):
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
description: workspace core tools
version: 1.0.0
tools:
  allowed-tools:
    - create_file
---
Workspace core.
""".strip(),
        encoding="utf-8",
    )
    (skills / "workspace-ops" / "tools.py").write_text(
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
  }
}

def execute(tool_name, args, env):
    path = env.workspace.create_file(args["filepath"], args["content"])
    return {"ok": True, "data": {"filepath": path}, "error": None, "meta": {}}
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    ctx = SkillContext(
        user_input="write a file",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )

    loaded = runtime.skill_view("workspace-ops", "", ctx)
    assert loaded["loaded"] is True
    selected = runtime.select_skills(ctx)
    assert [skill.id for skill in selected] == ["workspace-ops"]

    out = runtime.execute_tool_call(
        "create_file",
        {"filepath": "hello.txt", "content": "hi"},
        selected=selected,
        ctx=ctx,
    )

    assert out["ok"] is True


def test_tools_py_is_not_imported_until_execution(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "lazy-skill").mkdir(parents=True)

    (skills / "lazy-skill" / "SKILL.md").write_text(
        """
---
name: lazy-skill
description: lazy test
allowed-tools: lazy_tool
---
Lazy tool
""".strip(),
        encoding="utf-8",
    )
    (skills / "lazy-skill" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "lazy_tool": {
    "capability": "workspace_read",
    "description": "Lazy tool",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  }
}

RAISED = True
raise RuntimeError("should not import during skill load")

def execute(tool_name, args, env):
    return {"ok": True, "data": {"value": 1}, "error": None, "meta": {}}
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        debug=True,
    )
    skill = runtime.get_skill("lazy-skill")
    assert skill is not None

    ctx = SkillContext(
        user_input="run lazy tool",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )
    out = runtime.execute_tool_call("lazy_tool", {}, selected=[skill], ctx=ctx)
    assert out["ok"] is False
    assert out["error"]["code"] == "E_IO"


def test_runtimeerror_tool_message_is_preserved_without_debug(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "faulty-skill").mkdir(parents=True)

    (skills / "faulty-skill" / "SKILL.md").write_text(
        """
---
name: faulty-skill
description: tool that raises runtime errors
version: 1.0.0
tools:
  allowed-tools:
    - faulty_tool
---
faulty
""".strip(),
        encoding="utf-8",
    )

    (skills / "faulty-skill" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "faulty_tool": {
    "capability": "web_search",
    "description": "Always fails",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  }
}

def execute(tool_name, args, env):
    raise RuntimeError("Tavily API key not configured")
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        debug=False,
    )
    skill = runtime.get_skill("faulty-skill")
    assert skill is not None

    ctx = SkillContext(
        user_input="run faulty tool",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )
    out = runtime.execute_tool_call("faulty_tool", {}, selected=[skill], ctx=ctx)

    assert out["ok"] is False
    assert out["error"]["code"] == "E_IO"
    assert out["error"]["message"] == "Tavily API key not configured"


def test_select_skills_requires_explicit_session_load(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "frontend-design").mkdir(parents=True)
    (skills / "shell-ops").mkdir(parents=True)

    (skills / "frontend-design" / "SKILL.md").write_text(
        """
---
name: frontend-design
description: Create distinctive landing pages, HTML/CSS interfaces, and polished web UI.
metadata:
  tags:
    - html
    - css
    - design
---
Make the interface look great.
""".strip(),
        encoding="utf-8",
    )

    (skills / "shell-ops" / "SKILL.md").write_text(
        """
---
name: shell-ops
description: Run terminal commands in the workspace.
metadata:
  tags:
    - shell
    - terminal
---
Run shell commands carefully.
""".strip(),
        encoding="utf-8",
    )

    (skills / "shell-ops" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "shell_command": {
    "capability": "run_shell_command",
    "description": "Run a shell command",
    "parameters": {
      "type": "object",
      "properties": {"command": {"type": "string"}},
      "required": ["command"]
    }
  }
}

def execute(tool_name, args, env):
    return {"ok": True, "data": {"command": args.get("command", "")}, "error": None, "meta": {}}
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )

    ctx = SkillContext(
        user_input="Design a bold landing page in HTML and CSS for a boutique hotel.",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )

    assert runtime.select_skills(ctx) == []


def test_select_skills_does_not_auto_load_enabled_skills_for_neutral_request(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "frontend-design").mkdir(parents=True)
    (skills / "search-ops").mkdir(parents=True)

    (skills / "frontend-design" / "SKILL.md").write_text(
        """
---
name: frontend-design
description: Create distinctive landing pages, HTML/CSS interfaces, and polished web UI.
metadata:
  tags: [html, css, design]
---
Design well.
""".strip(),
        encoding="utf-8",
    )
    (skills / "search-ops" / "SKILL.md").write_text(
        """
---
name: search-ops
description: Search the web for recent information.
metadata:
  tags: [web, latest, research]
---
Search the internet.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )

    ctx = SkillContext(
        user_input="hi",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )

    assert runtime.select_skills(ctx) == []


def test_select_skills_does_not_auto_load_search_skill_without_explicit_load(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "search-ops").mkdir(parents=True)
    (skills / "frontend-design").mkdir(parents=True)

    (skills / "search-ops" / "SKILL.md").write_text(
        """
---
name: search-ops
description: Search the web and fetch page content for research and up-to-date information.
metadata:
  tags: [web, internet, latest, recent, current, news, lookup]
---
Search the internet.
""".strip(),
        encoding="utf-8",
    )
    (skills / "frontend-design" / "SKILL.md").write_text(
        """
---
name: frontend-design
description: Create distinctive landing pages, HTML/CSS interfaces, and polished web UI.
metadata:
  tags: [html, css, design]
---
Design well.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )

    ctx = SkillContext(
        user_input="what is the current situation in iran",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )

    assert runtime.select_skills(ctx) == []


def test_select_skills_ranks_loaded_skills_and_honors_top_n(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "search-ops").mkdir(parents=True)
    (skills / "frontend-design").mkdir(parents=True)

    (skills / "search-ops" / "SKILL.md").write_text(
        """
---
name: search-ops
description: Search the web for recent information.
metadata:
  tags: [web, latest, research]
---
Search the internet.
""".strip(),
        encoding="utf-8",
    )
    (skills / "frontend-design" / "SKILL.md").write_text(
        """
---
name: frontend-design
description: Create distinctive landing pages and polished web UI.
metadata:
  tags: [html, css, design]
---
Design well.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )
    ctx = SkillContext(
        user_input="look up the latest weather news on the web",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )
    runtime.skill_view("frontend-design", "", ctx)
    runtime.skill_view("search-ops", "", ctx)

    selected = runtime.select_skills(ctx, top_n=1)

    assert [skill.id for skill in selected] == ["search-ops"]


def test_tool_is_blocked_for_local_workspace_uses_capability_metadata(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "workspace-ops").mkdir(parents=True)
    (skills / "utilities").mkdir(parents=True)

    (skills / "workspace-ops" / "SKILL.md").write_text(
        """
---
name: workspace-ops
description: Workspace tools
tools:
  allowed-tools:
    - read_blob
---
Read files.
""".strip(),
        encoding="utf-8",
    )
    (skills / "workspace-ops" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "read_blob": {
    "capability": "workspace_read",
    "description": "Read file",
    "parameters": {"type": "object", "properties": {}, "required": []}
  }
}

def execute(tool_name, args, env):
    return {"ok": True, "data": {}, "error": None, "meta": {}}
""".strip(),
        encoding="utf-8",
    )
    (skills / "utilities" / "SKILL.md").write_text(
        """
---
name: utilities
description: Utility tools
tools:
  allowed-tools:
    - open_url
---
Utility helpers.
""".strip(),
        encoding="utf-8",
    )
    (skills / "utilities" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "open_url": {
    "capability": "utility_open_url",
    "description": "Open URL",
    "parameters": {"type": "object", "properties": {}, "required": []}
  }
}

def execute(tool_name, args, env):
    return {"ok": True, "data": {}, "error": None, "meta": {}}
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )

    assert runtime.tool_is_blocked_for_local_workspace("read_blob") is False
    assert runtime.tool_is_blocked_for_local_workspace("open_url") is True
    assert runtime.tool_is_blocked_for_local_workspace("request_user_input") is False
    assert runtime.tool_is_blocked_for_local_workspace("unknown_tool") is True


def test_rebase_vendor_paths_uses_discovered_skill_roots(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = home / ".codex" / "skills"
    home.mkdir()
    ws.mkdir(parents=True)
    (skills / "demo").mkdir(parents=True)
    (skills / "demo" / "SKILL.md").write_text(
        """
---
name: demo
description: Demo skill
---
Demo.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )
    skill = runtime.get_skill("demo")
    assert skill is not None

    rebased = runtime._rebase_vendor_paths(
        "Use ~/.codex/skills/demo/notes.md and ~/.claude/skills/demo/notes.md",
        skill,
    )

    assert str(skill.path) in rebased
    assert "~/.claude/skills/demo/notes.md" in rebased


def test_skill_catalog_includes_tools_for_model_routing(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "utilities").mkdir(parents=True)

    (skills / "utilities" / "SKILL.md").write_text(
        """
---
name: utilities
description: Open URLs and play songs or videos on YouTube.
allowed-tools: open_url play_youtube
metadata:
  tags: [open, youtube, play, music]
---
Utilities.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    catalog = runtime.skill_catalog_text()
    assert "utilities" in catalog
    assert "play songs or videos on YouTube" in catalog
    assert "tools: open_url, play_youtube" in catalog
    assert "location: skills/utilities" in catalog


def test_skill_with_missing_env_loads_as_unavailable(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "search-skill").mkdir(parents=True)
    monkeypatch.delenv("TEST_SKILL_KEY", raising=False)

    (skills / "search-skill" / "SKILL.md").write_text(
        """
---
name: search-skill
description: Search the internet.
requirements:
  env:
    - TEST_SKILL_KEY
---
Search.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )

    skill = runtime.get_skill("search-skill")
    assert skill is not None
    assert skill.available is False
    assert "missing env: TEST_SKILL_KEY" == skill.availability_reason
    assert runtime.enabled_skills() == []


def test_skill_with_missing_command_loads_as_unavailable(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "cmd-skill").mkdir(parents=True)

    (skills / "cmd-skill" / "SKILL.md").write_text(
        """
---
name: cmd-skill
description: Needs a missing binary.
requirements:
  commands:
    - definitely-not-a-real-binary-xyz
---
Binary required.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    skill = runtime.get_skill("cmd-skill")
    assert skill is not None
    assert skill.available is False
    assert "definitely-not-a-real-binary-xyz" in skill.availability_reason


def test_runtime_env_exposes_global_npm_root_as_node_path(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()

    class _Proc:
        returncode = 0
        stdout = "/opt/homebrew/lib/node_modules\n"
        stderr = ""

    monkeypatch.setattr(skills_module.shutil, "which", lambda name: "/opt/homebrew/bin/npm" if name == "npm" else None)
    monkeypatch.setattr(skills_module.subprocess, "run", lambda *args, **kwargs: _Proc())

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    assert runtime._proc_env_base["NODE_PATH"] == "/opt/homebrew/lib/node_modules"


def test_reload_preserves_manual_skill_toggle(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "toggle-skill").mkdir(parents=True)

    (skills / "toggle-skill" / "SKILL.md").write_text(
        """
---
name: toggle-skill
description: Toggle me.
---
Toggle.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    assert runtime.set_enabled("toggle-skill", False) is True
    runtime.load_skills()
    skill = runtime.get_skill("toggle-skill")
    assert skill is not None
    assert skill.enabled is False


def test_compose_skill_block_does_not_leave_unclosed_fence(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "docs-skill").mkdir(parents=True)

    (skills / "docs-skill" / "SKILL.md").write_text(
        """
---
name: docs-skill
description: Very long skill docs.
---
Intro line.

```python
print("alpha")
print("beta")
print("gamma")
```

Closing note.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("docs-skill")
    assert skill is not None

    block = runtime.compose_skill_block(
        [skill],
        SkillContext(
            user_input="use docs",
            branch_labels=[],
            attachments=[],
            workspace_root=str(ws),
            memory_hits=[],
        ),
        context_limit=40,
        ratio=1.0,
        hard_cap=1,
    )

    assert block.count("```") % 2 == 0


def test_agentskill_name_aliases_directory_when_names_differ(tmp_path: Path):
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
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("wrong-name")
    assert skill is not None
    assert "s1" in skill.aliases
    assert runtime.resolve_skill_reference("s1") is skill


def test_ambiguous_skill_alias_is_reported_and_not_resolved(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "alpha").mkdir(parents=True)
    (skills / "beta").mkdir(parents=True)

    (skills / "alpha" / "SKILL.md").write_text(
        """
---
name: shared
description: alpha
---
Alpha
""".strip(),
        encoding="utf-8",
    )
    (skills / "beta" / "SKILL.md").write_text(
        """
---
name: alpha
description: beta
---
Beta
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    ctx = SkillContext(
        user_input="load",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )
    with pytest.raises(FileNotFoundError):
        runtime.skill_view("alpha", "", ctx)
    report = runtime.skill_health_report()
    alpha_row = next(item for item in report if item["id"] == "shared")
    beta_row = next(item for item in report if item["id"] == "alpha")
    assert "alpha" in alpha_row["aliases"]
    assert "alpha" in alpha_row["alias_conflicts"]
    assert "alpha" in beta_row["alias_conflicts"]


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
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    listed = runtime.list_skills()
    assert len(listed) == 1
    assert listed[0].available is False
    assert listed[0].execution_allowed is False
    assert listed[0].validation_errors == ["missing required tools: create_file"]


def test_command_definition_manifest_registers_and_executes(tmp_path: Path):
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
      command: python3 scripts/echo_text.py {text}
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
import sys

def main():
    out = {"ok": True, "data": {"text": sys.argv[1]}, "error": None, "meta": {}}
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
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={"capabilities": {"shell_require_confirmation": False, "dangerously_skip_permissions": True}},
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


def test_entrypoint_non_shell_tool_is_normalized_and_runs(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "report-pdf" / "scripts").mkdir(parents=True)

    (skills / "report-pdf" / "SKILL.md").write_text(
        """
---
name: report-pdf
description: Create report files.
execution:
  entrypoints:
    - name: create_report
      tool: run_checks
      command: python3 {skill_root}/scripts/create_report.py {workspace_root}/report.txt
      parameters:
        type: object
        properties: {}
---
Create reports.
""".strip(),
        encoding="utf-8",
    )
    (skills / "report-pdf" / "scripts" / "create_report.py").write_text(
        "from pathlib import Path\nimport sys\nPath(sys.argv[1]).write_text('ok', encoding='utf-8')\n",
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={"capabilities": {"shell_require_confirmation": False, "dangerously_skip_permissions": True}},
    )
    skill = runtime.get_skill("report-pdf")
    assert skill is not None
    assert any("normalized to shell_command" in msg for msg in skill.validation_warnings)
    ctx = SkillContext(
        user_input="create report",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )
    out = runtime.execute_tool_call("run_skill", {"entrypoint": "create_report"}, selected=[skill], ctx=ctx)
    assert out["ok"] is True
    assert (ws / "report.txt").read_text(encoding="utf-8") == "ok"


def test_skill_prompt_is_loaded_lazily(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "lazy-skill").mkdir(parents=True)

    (skills / "lazy-skill" / "SKILL.md").write_text(
        """
---
name: lazy-skill
description: lazy prompt
version: 1.0.0
triggers:
  keywords:
    - lazy
---
Loaded on demand
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("lazy-skill")
    assert skill is not None
    assert skill.prompt is None

    ctx = SkillContext(
        user_input="lazy",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )
    block = runtime.compose_skill_block([skill], ctx=ctx, context_limit=1024)
    assert "Loaded on demand" in block
    assert skill.prompt == "Loaded on demand"


def test_hooks_file_is_ignored_for_runtime_surfaces(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "hooked-skill").mkdir(parents=True)
    (skills / "hooked-skill" / "SKILL.md").write_text(
        """
---
name: hooked-skill
description: has hooks file
version: 1.0.0
---
Hooked skill.
""".strip(),
        encoding="utf-8",
    )
    (skills / "hooked-skill" / "hooks.py").write_text(
        "def pre_prompt(context):\n    return 'ignored'\n",
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("hooked-skill")
    assert skill is not None
    assert skill.validation_errors == []
    assert runtime._reported_skill_scripts(skill) == []


def test_bundled_scripts_without_tool_definitions_use_generic_script_runner(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "script-only" / "scripts").mkdir(parents=True)

    (skills / "script-only" / "SKILL.md").write_text(
        """
---
name: script-only
description: ships a helper script but no executable tool contract
version: 1.0.0
---
Use the helper script if your runtime knows how.
""".strip(),
        encoding="utf-8",
    )
    (skills / "script-only" / "scripts" / "helper.py").write_text(
        "if __name__ == '__main__':\n    print('ok')\n",
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("script-only")
    assert skill is not None

    assert set(_tool_names(runtime, [skill])) == (_always_available_tool_names() | {"run_skill"})
    assert runtime._reported_skill_scripts(skill) == ["scripts/helper.py"]


def test_generic_script_runner_executes_selected_skill_script(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "script-only" / "scripts").mkdir(parents=True)

    (skills / "script-only" / "SKILL.md").write_text(
        """
---
name: script-only
description: helper script execution
version: 1.0.0
---
Use helper.py when needed.
""".strip(),
        encoding="utf-8",
    )
    (skills / "script-only" / "scripts" / "helper.py").write_text(
        """
import json
import os
import sys

def main():
    payload = json.loads(os.getenv("ALPHANUS_TOOL_ARGS_JSON") or "{}")
    print(json.dumps({"stdout": sys.stdin.read(), "payload": payload}))

if __name__ == "__main__":
    main()
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("script-only")
    assert skill is not None
    ctx = SkillContext(
        user_input="run the bundled helper",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )

    assert set(_tool_names(runtime, [skill], ctx=ctx)) == (_always_available_tool_names() | {"run_skill"})

    out = runtime.execute_tool_call(
        "run_skill",
        {"script": "helper.py", "stdin": "hello", "params": {"mode": "demo"}},
        selected=[skill],
        ctx=ctx,
    )

    assert out["ok"] is True
    assert out["data"]["skill_id"] == "script-only"
    assert out["data"]["script"] == "scripts/helper.py"
    assert out["data"]["stdout"] == "hello"
    assert out["data"]["payload"] == {"mode": "demo"}

    legacy = runtime.execute_tool_call(
        "run_skill",
        {"script": "helper.py", "args": {"mode": "demo"}},
        selected=[skill],
        ctx=ctx,
    )
    assert legacy["ok"] is False
    assert legacy["error"]["code"] == "E_VALIDATION"
    assert "Unexpected arguments: args" in legacy["error"]["message"]


def test_generic_script_runner_respects_allowed_tools_policy(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "script-wrapper" / "scripts").mkdir(parents=True)

    (skills / "script-wrapper" / "SKILL.md").write_text(
        """
---
name: script-wrapper
description: Expose only a narrow wrapper tool.
allowed-tools: wrapped_tool
---
Do not expose the raw script runner.
""".strip(),
        encoding="utf-8",
    )
    (skills / "script-wrapper" / "scripts" / "helper.py").write_text("print('ok')\n", encoding="utf-8")

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("script-wrapper")
    assert skill is not None
    ctx = SkillContext(
        user_input="run the helper script",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )

    assert set(_tool_names(runtime, [skill], ctx=ctx)) == _always_available_tool_names()

    out = runtime.execute_tool_call(
        "run_skill",
        {"script": "helper.py"},
        selected=[skill],
        ctx=ctx,
    )
    assert out["ok"] is False
    assert out["error"]["code"] == "E_POLICY"


def test_generic_entrypoint_runner_executes_structured_workflow(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "report-pdf" / "scripts").mkdir(parents=True)

    (skills / "report-pdf" / "SKILL.md").write_text(
        """
---
name: report-pdf
description: Create report PDFs with a declared workflow.
produces:
  - .pdf
execution:
  entrypoints:
    - name: create_report
      description: Create a PDF report artifact.
      intents: [create]
      produces: [.pdf]
      install:
        - python3 -c "print('install-ok')"
      verify:
        - python3 -c "print('verify-ok')"
      command: python3 {skill_root}/scripts/create_report.py {workspace_root}/{filename}
      parameters:
        type: object
        properties:
          filename:
            type: string
        required:
          - filename
---
Create PDFs with the declared entrypoint.
""".strip(),
        encoding="utf-8",
    )
    (skills / "report-pdf" / "scripts" / "create_report.py").write_text(
        "from pathlib import Path\nimport sys\nPath(sys.argv[1]).write_text('artifact', encoding='utf-8')\nprint('done')\n",
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={"capabilities": {"shell_require_confirmation": False, "dangerously_skip_permissions": True}},
    )
    skill = runtime.get_skill("report-pdf")
    assert skill is not None
    ctx = SkillContext(
        user_input="Create a report PDF and save it as report.pdf",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )

    assert set(_tool_names(runtime, [skill], ctx=ctx)) == (_always_available_tool_names() | {"run_skill"})

    out = runtime.execute_tool_call(
        "run_skill",
        {"entrypoint": "create_report", "params": {"filename": "report.pdf"}},
        selected=[skill],
        ctx=ctx,
    )

    assert out["ok"] is True
    assert out["data"]["skill_id"] == "report-pdf"
    assert out["data"]["entrypoint"] == "create_report"
    assert (ws / "report.pdf").read_text(encoding="utf-8") == "artifact"


def test_run_skill_supports_compact_execution_command_manifest(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "compact-report" / "scripts").mkdir(parents=True)

    (skills / "compact-report" / "SKILL.md").write_text(
        """
---
name: compact-report
description: Compact execution manifest.
produces:
  - .txt
execution:
  command: python3 {skill_root}/scripts/create_report.py {workspace_root}/{filename}
  cwd: workspace
  timeout_s: 20
  parameters:
    type: object
    properties:
      filename:
        type: string
    required:
      - filename
---
Generate the report through the compact execution contract.
""".strip(),
        encoding="utf-8",
    )
    (skills / "compact-report" / "scripts" / "create_report.py").write_text(
        "from pathlib import Path\nimport sys\nPath(sys.argv[1]).write_text('compact', encoding='utf-8')\n",
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={"capabilities": {"shell_require_confirmation": False, "dangerously_skip_permissions": True}},
    )
    skill = runtime.get_skill("compact-report")
    assert skill is not None
    ctx = SkillContext(
        user_input="create compact-report.txt",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )

    assert set(_tool_names(runtime, [skill], ctx=ctx)) == (_always_available_tool_names() | {"run_skill"})

    out = runtime.execute_tool_call(
        "run_skill",
        {"entrypoint": "run", "params": {"filename": "compact-report.txt"}},
        selected=[skill],
        ctx=ctx,
    )

    assert out["ok"] is True
    assert out["data"]["entrypoint"] == "run"
    assert (ws / "compact-report.txt").read_text(encoding="utf-8") == "compact"


def test_frontmatter_metadata_command_tool_is_supported(tmp_path: Path):
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
        command: python3 scripts/echo_text.py {text}
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
import sys

def main():
    print(json.dumps({"text": sys.argv[1]}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={"capabilities": {"shell_require_confirmation": False, "dangerously_skip_permissions": True}},
    )
    skill = runtime.get_skill("meta-skill")
    assert skill is not None
    ctx = SkillContext(
        user_input="echo",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )
    out = runtime.execute_tool_call("echo_text", {"text": "hello"}, selected=[skill], ctx=ctx)
    assert out["ok"] is True
    assert out["data"]["text"] == "hello"


def test_command_definition_non_json_output_returns_stdout(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "bad-output" / "scripts").mkdir(parents=True)

    (skills / "bad-output" / "SKILL.md").write_text(
        """
---
name: bad-output
description: invalid output
version: 1.0.0
tools:
  allowed-tools:
    - bad_tool
  definitions:
    - name: bad_tool
      capability: utility_bad
      description: Emit invalid output.
      command: python3 scripts/bad_tool.py
      parameters:
        type: object
        properties: {}
---
Bad output
""".strip(),
        encoding="utf-8",
    )
    (skills / "bad-output" / "scripts" / "bad_tool.py").write_text(
        """
print("definitely not json")
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={"capabilities": {"shell_require_confirmation": False, "dangerously_skip_permissions": True}},
    )
    skill = runtime.get_skill("bad-output")
    assert skill is not None

    ctx = SkillContext(
        user_input="bad",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )
    out = runtime.execute_tool_call("bad_tool", {}, selected=[skill], ctx=ctx)
    assert out["ok"] is True
    assert "definitely not json" in out["data"]["stdout"]


def test_command_tool_timeout_definition_reports_timeout(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "slow-skill" / "scripts").mkdir(parents=True)

    (skills / "slow-skill" / "SKILL.md").write_text(
        """
---
name: slow-skill
description: timeout test
version: 1.0.0
tools:
  allowed-tools:
    - slow_tool
  definitions:
    - name: slow_tool
      capability: utility_wait
      description: Sleep long enough to hit timeout.
      command: python3 scripts/slow_tool.py
      timeout-s: 1
      parameters:
        type: object
        properties: {}
---
Slow
""".strip(),
        encoding="utf-8",
    )
    (skills / "slow-skill" / "scripts" / "slow_tool.py").write_text(
        """
import time

time.sleep(2)
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={"capabilities": {"shell_require_confirmation": False, "dangerously_skip_permissions": True}},
    )
    skill = runtime.get_skill("slow-skill")
    assert skill is not None
    ctx = SkillContext(
        user_input="wait",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )
    out = runtime.execute_tool_call("slow_tool", {}, selected=[skill], ctx=ctx)
    assert out["ok"] is False
    assert out["error"]["code"] == "E_TIMEOUT"


def test_skill_health_report_includes_provenance(tmp_path: Path):
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
---
Hello
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    report = runtime.skill_health_report()

    assert report[0]["provenance"] == "repo/skills"
    assert report[0]["availability_code"] == "ready"


def test_bundled_skill_inside_workspace_repo_is_executable(tmp_path: Path):
    home = tmp_path / "home"
    repo = home / "Desktop" / "Alphanus"
    skills = repo / "skills"
    skill_dir = skills / "repo-helper"
    home.mkdir()
    skill_dir.mkdir(parents=True)

    (skill_dir / "SKILL.md").write_text(
        """
---
name: repo-helper
description: bundled repo helper
version: 1.0.0
tools:
  allowed-tools:
    - echo_text
---
Bundled repo helper.
""".strip(),
        encoding="utf-8",
    )
    (skill_dir / "tools.py").write_text(
        """
TOOL_SPECS = {
  "echo_text": {
    "capability": "utility_echo",
    "description": "Echo text",
    "parameters": {
      "type": "object",
      "properties": {"text": {"type": "string"}},
      "required": ["text"]
    }
  }
}

def execute(tool_name, args, env):
    return {"ok": True, "data": {"text": args["text"]}, "error": None, "meta": {}}
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(repo), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    skill = runtime.get_skill("repo-helper")
    assert skill is not None
    assert runtime.skill_provenance_label(skill) == "repo/skills"
    assert skill.execution_allowed is True
    assert skill.available is True
    assert set(_tool_names(runtime, [skill])) == (_always_available_tool_names() | {"echo_text"})


def test_bundled_skill_under_home_outside_workspace_is_executable(tmp_path: Path):
    home = tmp_path / "home"
    repo = home / "Desktop" / "Alphanus"
    workspace_root = home / "projects" / "demo"
    skills = repo / "skills"
    skill_dir = skills / "repo-helper"
    home.mkdir()
    workspace_root.mkdir(parents=True)
    skill_dir.mkdir(parents=True)

    (skill_dir / "SKILL.md").write_text(
        """
---
name: repo-helper
description: bundled repo helper
version: 1.0.0
---
Bundled repo helper.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(workspace_root), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    skill = runtime.get_skill("repo-helper")
    assert skill is not None
    assert runtime.skill_provenance_label(skill) == "repo/skills"
    assert skill.execution_allowed is True
    assert skill.available is True
    report = runtime.skill_health_report()
    skill_report = next(item for item in report if item["id"] == "repo-helper")
    assert skill_report["provenance"] == "repo/skills"
    assert skill_report["execution_allowed"] is True


def test_home_skill_root_is_not_discovered(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    bundled = tmp_path / "bundled"
    user_pack_root = home / ".claude"
    skill_dir = user_pack_root / "skills" / "home-helper"
    agent_dir = user_pack_root / "agents"
    home.mkdir()
    ws.mkdir()
    skill_dir.mkdir(parents=True)
    agent_dir.mkdir(parents=True)

    (skill_dir / "SKILL.md").write_text(
        """
---
name: home-helper
description: home metadata-only skill
version: 1.0.0
allowed-tools: home_tool
execution:
  entrypoints:
    - name: generate_notes
      command: python3 {skill_root}/scripts/generate_notes.py
      parameters:
        type: object
        properties: {}
tools:
  definitions:
    - name: home_tool
      capability: utility_home
      description: Home helper.
      command: python3 scripts/home_tool.py
      parameters:
        type: object
        properties: {}
---
Home helper.
""".strip(),
        encoding="utf-8",
    )
    (skill_dir / "tools.py").write_text("raise RuntimeError('should never import')\n", encoding="utf-8")
    (skill_dir / "hooks.py").write_text("raise RuntimeError('should never import')\n", encoding="utf-8")
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "generate_notes.py").write_text("print('nope')\n", encoding="utf-8")
    (skill_dir / "scripts" / "home_tool.py").write_text("print('nope')\n", encoding="utf-8")
    (agent_dir / "researcher.md").write_text(
        """
---
name: researcher
description: Research helper
---
Do work.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(bundled),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    assert runtime.get_skill("home-helper") is None
    assert all(skill.id != "home-helper" for skill in runtime.list_skills())


def test_only_bundled_root_is_used_for_discovery(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    bundled = tmp_path / "bundled"
    bundled_skill = bundled / "dup-skill"
    workspace_skill = ws / ".claude" / "skills" / "dup-skill"
    home.mkdir()
    ws.mkdir()
    bundled_skill.mkdir(parents=True)
    workspace_skill.mkdir(parents=True)

    (bundled_skill / "SKILL.md").write_text(
        """
---
name: dup-skill
description: bundled
version: 1.0.0
---
Bundled version.
""".strip(),
        encoding="utf-8",
    )
    (workspace_skill / "SKILL.md").write_text(
        """
---
name: dup-skill
description: workspace
version: 2.0.0
---
Workspace version.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(bundled),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    active = runtime.get_skill("dup-skill")
    assert active is not None
    assert active.description == "bundled"
    assert runtime.skill_provenance_label(active) == "repo/skills"
    assert len(runtime.list_skills()) == 1


def test_request_user_input_runtime_tool_uses_callback(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "asker").mkdir(parents=True)
    (skills / "asker" / "SKILL.md").write_text(
        """
---
name: asker
description: ask follow-up questions
version: 1.0.0
---
Ask for clarification when needed.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("asker")
    assert skill is not None
    ctx = SkillContext(
        user_input="use skill asker",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
        explicit_skill_id="asker",
    )

    out = runtime.execute_tool_call(
        "request_user_input",
        {"question": "Choose one", "options": ["a", "b"]},
        selected=[skill],
        ctx=ctx,
        request_user_input=lambda args: {
            "question": args["question"],
            "options": list(args.get("options", [])),
            "awaiting_user_input": True,
        },
    )

    assert out["ok"] is True
    assert out["data"]["awaiting_user_input"] is True
    assert out["data"]["options"] == ["a", "b"]


def test_blocked_python_scripts_are_reported_from_runtime_metadata(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "script-check" / "scripts").mkdir(parents=True)
    (skills / "script-check" / "SKILL.md").write_text(
        """
---
name: script-check
description: helper with optional scripts
version: 1.0.0
---
Use the bundled helper script when available.
""".strip(),
        encoding="utf-8",
    )
    (skills / "script-check" / "scripts" / "helper.py").write_text(
        "import definitely_missing_mod_xyz\n\nif __name__ == '__main__':\n    print('ok')\n",
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    skill = runtime.get_skill("script-check")
    assert skill is not None
    assert runtime._skill_runnable_scripts(skill) == []
    assert runtime._reported_skill_scripts(skill) == []
    assert runtime._blocked_skill_scripts(skill) == [
        {
            "script": "scripts/helper.py",
            "reason": "missing python modules: definitely_missing_mod_xyz",
        }
    ]
    assert set(_tool_names(runtime, [skill])) == _always_available_tool_names()


def test_python_script_without_main_guard_stays_exposed(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "script-check" / "scripts").mkdir(parents=True)
    (skills / "script-check" / "SKILL.md").write_text(
        """
---
name: script-check
description: helper with optional scripts
version: 1.0.0
---
Use the bundled helper script when available.
""".strip(),
        encoding="utf-8",
    )
    (skills / "script-check" / "scripts" / "helper.py").write_text(
        "print('ok')\n",
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    skill = runtime.get_skill("script-check")
    assert skill is not None
    assert runtime._skill_runnable_scripts(skill) == ["scripts/helper.py"]
    assert runtime._reported_skill_scripts(skill) == ["scripts/helper.py"]


def test_runnable_scripts_cache_reuses_scan_and_invalidates_on_reload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "script-check" / "scripts").mkdir(parents=True)
    (skills / "script-check" / "SKILL.md").write_text(
        """
---
name: script-check
description: helper with optional scripts
version: 1.0.0
---
Use the bundled helper script when available.
""".strip(),
        encoding="utf-8",
    )
    (skills / "script-check" / "scripts" / "helper.py").write_text("print('ok')\n", encoding="utf-8")

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    skill = runtime.get_skill("script-check")
    assert skill is not None
    calls = {"count": 0}
    original = runtime._python_script_missing_modules

    def counted(skill_obj, rel_script):
        calls["count"] += 1
        return original(skill_obj, rel_script)

    monkeypatch.setattr(runtime, "_python_script_missing_modules", counted)

    assert runtime._skill_runnable_scripts(skill) == ["scripts/helper.py"]
    assert runtime._skill_runnable_scripts(skill) == ["scripts/helper.py"]
    assert calls["count"] == 0

    runtime.load_skills()
    skill = runtime.get_skill("script-check")
    assert skill is not None
    assert runtime._skill_runnable_scripts(skill) == ["scripts/helper.py"]
    assert calls["count"] == 1


def test_configured_python_executable_controls_script_availability(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "script-check" / "scripts").mkdir(parents=True)
    (skills / "script-check" / "SKILL.md").write_text(
        """
---
name: script-check
description: helper with optional scripts
version: 1.0.0
---
Use the bundled helper script when available.
""".strip(),
        encoding="utf-8",
    )
    (skills / "script-check" / "scripts" / "helper.py").write_text(
        "if __name__ == '__main__':\n    print('ok')\n",
        encoding="utf-8",
    )

    missing_python = tmp_path / "missing-python"
    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={"skills": {"python_executable": str(missing_python)}},
    )

    skill = runtime.get_skill("script-check")
    assert skill is not None
    assert runtime._skill_runnable_scripts(skill) == []
    assert runtime._blocked_skill_scripts(skill) == [
        {
            "script": "scripts/helper.py",
            "reason": f"missing interpreter: {missing_python}",
        }
    ]


def test_skill_view_uses_single_active_skill_when_file_path_is_provided(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "doc-helper").mkdir(parents=True)
    (skills / "doc-helper" / "SKILL.md").write_text(
        """
---
name: doc-helper
description: helper for docs
version: 1.0.0
---
Read the bundled README when needed.
""".strip(),
        encoding="utf-8",
    )
    (skills / "doc-helper" / "README.md").write_text("# hello\n", encoding="utf-8")

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("doc-helper")
    assert skill is not None
    ctx = SkillContext(
        user_input="read the doc helper resource",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )

    out = runtime.execute_tool_call(
        "skill_view",
        {"name": "doc-helper", "file_path": "README.md"},
        selected=[skill],
        ctx=ctx,
    )

    assert out["ok"] is True
    assert out["data"]["skill_id"] == "doc-helper"
    assert out["data"]["file_path"] == "README.md"


def test_obsolete_runtime_tools_are_not_exposed(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "alpha").mkdir(parents=True)
    (skills / "alpha" / "scripts").mkdir(parents=True)
    (skills / "alpha" / "SKILL.md").write_text(
        """
---
name: alpha
description: alpha helper
version: 1.0.0
---
Use the helper script.
""".strip(),
        encoding="utf-8",
    )
    (skills / "alpha" / "scripts" / "helper.py").write_text("print('ok')\n", encoding="utf-8")

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    alpha = runtime.get_skill("alpha")
    assert alpha is not None
    ctx = SkillContext(
        user_input="use alpha",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )

    tool_names = {item["function"]["name"] for item in runtime.tools_for_turn([alpha], ctx=ctx)}
    assert "run_skill" in tool_names
    assert "read_skill_resource" not in tool_names
    assert "run_skill_command" not in tool_names
    assert "spawn_skill_agent" not in tool_names
    assert runtime.tool_registration("read_skill_resource") is None
    assert runtime.tool_registration("run_skill_command") is None
    assert runtime.tool_registration("spawn_skill_agent") is None


def test_runtime_tool_schema_requires_skill_id_only_when_multiple_active_skills_are_available(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    for skill_id in ("alpha", "beta"):
        (skills / skill_id).mkdir(parents=True)
        (skills / skill_id / "SKILL.md").write_text(
            f"""
---
name: {skill_id}
description: {skill_id} helper
version: 1.0.0
---
```bash
echo {skill_id}
```
""".strip(),
            encoding="utf-8",
        )
        (skills / skill_id / "README.md").write_text(f"# {skill_id}\n", encoding="utf-8")

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    alpha = runtime.get_skill("alpha")
    beta = runtime.get_skill("beta")
    assert alpha is not None and beta is not None
    ctx = SkillContext(
        user_input="use both skills",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )

    single_tools = runtime.tools_for_turn([alpha], ctx=ctx)
    multi_tools = runtime.tools_for_turn([alpha, beta], ctx=ctx)

    def tool_params(tools, name):
        for item in tools:
            fn = item.get("function", {})
            if fn.get("name") == name:
                return fn.get("parameters", {})
        raise AssertionError(f"missing tool schema for {name}")

    single_view = tool_params(single_tools, "skill_view")
    multi_view = tool_params(multi_tools, "skill_view")
    assert single_view == multi_view


def test_tools_for_turn_cache_key_ignores_context_fingerprint(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "script-skill").mkdir(parents=True)
    (skills / "script-skill" / "SKILL.md").write_text(
        """
---
name: script-skill
description: artifact-specific helper
version: 1.0.0
---
Use the bundled helper script when available.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("script-skill")
    assert skill is not None

    docx_ctx = SkillContext(
        user_input="set up report.docx",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )
    png_ctx = SkillContext(
        user_input="create image.png",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )

    calls = {"count": 0}

    def fake_tool_schemas(names, selected=None, ctx=None):
        calls["count"] += 1
        return [{"ctx": getattr(ctx, "user_input", ""), "names": list(names)}]

    runtime._tool_schemas = fake_tool_schemas  # type: ignore[method-assign]

    docx_tools = runtime.tools_for_turn([skill], ctx=docx_ctx)
    png_tools = runtime.tools_for_turn([skill], ctx=png_ctx)

    assert calls["count"] == 1
    assert docx_tools == png_tools
    assert docx_tools[0]["ctx"] == "set up report.docx"
