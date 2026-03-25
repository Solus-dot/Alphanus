from __future__ import annotations
from pathlib import Path

import core.skills as skills_module
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
    out = runtime.execute_tool_call("read_blob", {"filepath": "a.txt"}, selected=selected, ctx=ctx)
    assert out["ok"] is False
    assert out["error"]["code"] == "E_UNSUPPORTED"


def test_tools_for_turn_includes_core_tools_without_selected_skill(tmp_path: Path):
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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    core_tool_names = [tool["function"]["name"] for tool in runtime.tools_for_turn([])]
    assert core_tool_names == ["create_file", "read_file"]

    search_skill = runtime.get_skill("search-ops")
    assert search_skill is not None
    merged_tool_names = [tool["function"]["name"] for tool in runtime.tools_for_turn([search_skill])]
    assert merged_tool_names == ["create_file", "load_skill", "read_file", "web_search"]


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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    skill = runtime.get_skill("script-only")
    assert skill is not None
    tools = runtime.tools_for_turn([skill])
    tool_names = [tool["function"]["name"] for tool in tools]
    assert tool_names == ["load_skill", "run_skill_script"]
    script_tool = tools[1]["function"]
    assert "scripts/helper.py" in script_tool["description"]
    assert script_tool["parameters"]["properties"]["script"]["enum"] == ["scripts/helper.py"]


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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
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

    tools = runtime.tools_for_turn([skill], ctx=ctx)
    tool_names = [tool["function"]["name"] for tool in tools]
    assert tool_names == ["load_skill", "run_skill_entrypoint"]
    entry_tool = tools[1]["function"]
    assert "report-pdf:create_report" in entry_tool["description"]
    assert entry_tool["parameters"]["properties"]["entrypoint"]["enum"] == ["create_report"]


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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    skill = runtime.get_skill("research-helper")
    assert skill is not None
    tools = runtime.tools_for_turn([skill])
    assert [tool["function"]["name"] for tool in tools] == ["load_skill", "run_skill_script"]
    contract = runtime.load_skill_contract("research-helper")
    assert contract.scripts == ["validate_json.py"]
    assert contract.resources == []


def test_produces_entrypoint_still_matches_create_plus_review_request(tmp_path: Path):
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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
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

    assert runtime._skill_supports_artifact(skill, [".pdf"], intents=runtime.task_intents(ctx)) is True
    tools = runtime.tools_for_turn([skill], ctx=ctx)
    assert [tool["function"]["name"] for tool in tools] == ["load_skill", "run_skill_entrypoint"]


def test_core_tool_executes_without_selected_skill(tmp_path: Path):
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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    ctx = SkillContext(
        user_input="write a file",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )

    out = runtime.execute_tool_call(
        "create_file",
        {"filepath": "hello.txt", "content": "hi"},
        selected=[],
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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
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


def test_prompt_only_skill_can_be_selected_from_description_metadata(tmp_path: Path):
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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={"skills": {"selection_mode": "heuristic", "max_active_skills": 1}},
    )

    ctx = SkillContext(
        user_input="Design a bold landing page in HTML and CSS for a boutique hotel.",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )

    selected = runtime.select_skills(ctx)
    assert selected
    assert selected[0].id == "frontend-design"


def test_heuristic_selection_skips_zero_score_skills(tmp_path: Path):
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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={"skills": {"selection_mode": "heuristic", "max_active_skills": 2}},
    )

    ctx = SkillContext(
        user_input="hi",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )

    assert runtime.select_skills(ctx) == []


def test_time_sensitive_query_prefers_search_skill(tmp_path: Path):
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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={"skills": {"selection_mode": "heuristic", "max_active_skills": 2}},
    )

    ctx = SkillContext(
        user_input="what is the current situation in iran",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )

    selected = runtime.select_skills(ctx)
    assert selected
    assert selected[0].id == "search-ops"


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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={"skills": {"selection_mode": "heuristic", "max_active_skills": 1}},
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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("script-only")
    assert skill is not None

    tools = runtime.tools_for_turn([skill])
    assert [tool["function"]["name"] for tool in tools] == ["load_skill", "run_skill_script"]
    contract = runtime.load_skill_contract("script-only")
    assert contract.scripts == ["scripts/helper.py"]


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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
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

    out = runtime.execute_tool_call(
        "run_skill_script",
        {"script": "helper.py", "stdin": "hello", "args": {"mode": "demo"}},
        selected=[skill],
        ctx=ctx,
    )

    assert out["ok"] is True
    assert out["data"]["skill_id"] == "script-only"
    assert out["data"]["script"] == "scripts/helper.py"
    assert out["data"]["stdout"] == "hello"
    assert out["data"]["payload"] == {"mode": "demo"}


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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
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

    assert [tool["function"]["name"] for tool in runtime.tools_for_turn([skill], ctx=ctx)] == ["load_skill"]

    out = runtime.execute_tool_call(
        "run_skill_script",
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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
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

    out = runtime.execute_tool_call(
        "run_skill_entrypoint",
        {"entrypoint": "create_report", "params": {"filename": "report.pdf"}},
        selected=[skill],
        ctx=ctx,
    )

    assert out["ok"] is True
    assert out["data"]["skill_id"] == "report-pdf"
    assert out["data"]["entrypoint"] == "create_report"
    assert len(out["data"]["install_results"]) == 1
    assert len(out["data"]["verify_results"]) == 1
    assert (ws / "report.pdf").read_text(encoding="utf-8") == "artifact"


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


def test_invalid_command_tool_output_maps_to_e_protocol(tmp_path: Path):
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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
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
    assert out["ok"] is False
    assert out["error"]["code"] == "E_PROTOCOL"


def test_command_timeout_maps_to_e_timeout(tmp_path: Path):
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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    report = runtime.skill_health_report()

    assert report[0]["source_tier"] == "bundled"
    assert report[0]["availability_code"] == "ready"


def test_runtime_discovers_pack_agents_and_rebases_agent_paths(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    bundled = tmp_path / "bundled"
    pack_root = ws / ".claude"
    skill_dir = pack_root / "skills" / "frontend-design"
    agent_dir = pack_root / "agents"
    home.mkdir()
    ws.mkdir()
    skill_dir.mkdir(parents=True)
    agent_dir.mkdir(parents=True)

    (skill_dir / "SKILL.md").write_text(
        """
---
name: frontend-design
description: frontend design skill
version: 1.0.0
---
Use ~/.codex/skills/frontend-design/templates/mockup.html
""".strip(),
        encoding="utf-8",
    )
    (agent_dir / "researcher.md").write_text(
        """
---
name: researcher
description: Research helper
---
Read ~/.codex/skills/frontend-design/SKILL.md and coordinate with ~/.codex/agents/researcher.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(bundled),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    skill = runtime.resolve_skill_reference("front")
    assert skill is not None
    assert skill.id == "frontend-design"

    assert runtime.get_agent("researcher") is not None

    contract = runtime.load_agent_contract("researcher", skill_id="frontend-design")
    assert str(skill_dir) in contract.prompt
    assert str((agent_dir / "researcher.md").resolve()) in contract.prompt

    report = runtime.skill_health_report()
    assert report[0]["agents"] == ["researcher"]
    assert report[0]["source_tier"] == "workspace/local"


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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
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


def test_load_skill_contract_exposes_declared_commands(tmp_path: Path):
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
description: helper for documents
version: 1.0.0
---
Use these commands:

```bash
npm install -g docx
python scripts/build.py proposal.md proposal.docx
```
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    contract = runtime.load_skill_contract("doc-helper")
    assert contract.commands == ["npm install -g docx", "python scripts/build.py proposal.md proposal.docx"]


def test_shell_workflow_commands_keep_inline_install_and_ignore_javascript_fence(tmp_path: Path):
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
description: helper for documents
version: 1.0.0
---
Install: `npm install -g docx`

```javascript
const width = 9360;
```

```
category: International Product
```

```bash
python scripts/office/validate.py doc.docx
```
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    contract = runtime.load_skill_contract("doc-helper")
    assert contract.commands == ["npm install -g docx", "python scripts/office/validate.py doc.docx"]
    assert contract.install_commands == ["npm install -g docx"]
    assert contract.verify_commands == ["python scripts/office/validate.py doc.docx"]


def test_load_skill_contract_reports_blocked_python_scripts(tmp_path: Path):
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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    skill = runtime.get_skill("script-check")
    assert skill is not None
    assert runtime._skill_runnable_scripts(skill) == []
    contract = runtime.load_skill_contract("script-check")
    assert contract.scripts == []
    assert contract.blocked_scripts == [
        {
            "script": "scripts/helper.py",
            "reason": "missing python modules: definitely_missing_mod_xyz",
        }
    ]
    assert [tool["function"]["name"] for tool in runtime.tools_for_turn([skill])] == ["load_skill"]


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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    skill = runtime.get_skill("script-check")
    assert skill is not None
    assert runtime._skill_runnable_scripts(skill) == ["scripts/helper.py"]
    contract = runtime.load_skill_contract("script-check")
    assert contract.scripts == ["scripts/helper.py"]


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
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={"skills": {"python_executable": str(missing_python)}},
    )

    skill = runtime.get_skill("script-check")
    assert skill is not None
    assert runtime._skill_runnable_scripts(skill) == []
    contract = runtime.load_skill_contract("script-check")
    assert contract.blocked_scripts == [
        {
            "script": "scripts/helper.py",
            "reason": f"missing interpreter: {missing_python}",
        }
    ]


def test_run_skill_command_rejects_undeclared_command(tmp_path: Path):
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
description: helper for documents
version: 1.0.0
---
```bash
npm install -g docx
```
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("doc-helper")
    assert skill is not None
    ctx = SkillContext(
        user_input="use skill doc-helper",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
        explicit_skill_id="doc-helper",
    )

    out = runtime.execute_tool_call(
        "run_skill_command",
        {"skill_id": "doc-helper", "command": "create-proposal"},
        selected=[skill],
        ctx=ctx,
    )

    assert out["ok"] is False
    assert out["error"]["code"] == "E_POLICY"
