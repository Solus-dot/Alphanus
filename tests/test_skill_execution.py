from __future__ import annotations

from pathlib import Path

import pytest

from core.memory import LexicalMemory
from core.project import ProjectRuntime
from skills.runtime import SkillContext, SkillRuntime


def _tool_names(runtime: SkillRuntime, selected, ctx: SkillContext | None = None) -> list[str]:
    return [tool["function"]["name"] for tool in runtime.tools_for_turn(selected, ctx=ctx)]


def _always_available_tool_names() -> set[str]:
    return {"request_user_input", "skill_view", "skills_list"}


def test_bundled_skill_allowed_tools_match_registered_tool_specs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    runtime = SkillRuntime(
        skills_dir=str(repo_root / "bundled-skills"),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )

    for skill_dir in sorted((repo_root / "bundled-skills").iterdir()):
        if not (skill_dir / "tools.py").exists():
            continue
        skill = runtime.get_skill(skill_dir.name)
        assert skill is not None
        assert skill.validation_errors == []
        assert set(skill.allowed_tools) == set(runtime._reported_skill_tools(skill))


def test_skill_index_warns_against_namespaced_skill_tool_calls(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    runtime = SkillRuntime(
        skills_dir="bundled-skills",
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )

    index = runtime.compose_skill_index()

    assert "load it with skill_view(name)" in index
    assert "before saying file tools are unavailable" in index
    assert "Do not call namespaced skill tools" in index
    assert "exact unqualified function names" in index


def test_project_write_mode_exposes_selected_tools(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    runtime = SkillRuntime(
        skills_dir="bundled-skills",
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={"permissions": {"mode": "project-write", "approvals": "on-boundary", "network": False}},
        debug=True,
    )
    ctx = SkillContext(
        user_input="search the latest news",
        branch_labels=[],
        attachments=[],
        project_root=str(ws),
        memory_hits=[],
    )
    runtime.skill_view("search-ops", "", ctx)
    runtime.skill_view("project-ops", "", ctx)
    ctx.loaded_skill_ids = ["search-ops", "project-ops"]
    selected = runtime.select_skills(ctx)
    names = set(runtime.allowed_tool_names(selected, ctx=ctx))

    assert "create_file" in names
    assert "read_file" in names
    assert "web_search" in names
    assert "fetch_url" in names
    assert "skills_list" in names
    assert "skill_view" in names


def test_read_only_mode_allows_read_only_project_tools(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    runtime = SkillRuntime(
        skills_dir="bundled-skills",
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={"permissions": {"mode": "read-only", "approvals": "on-boundary", "network": False}},
        debug=True,
    )
    selected = runtime.skills_by_ids(["project-ops", "search-ops", "shell-ops"])
    names = set(runtime.allowed_tool_names(selected))

    assert "read_file" in names
    assert "list_files" in names
    assert "find_files" in names
    assert "project_tree" in names
    assert "create_file" not in names
    assert "delete_path" not in names
    assert "shell_command" not in names


def test_project_write_mode_allows_project_mutation_web_and_shell(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    runtime = SkillRuntime(
        skills_dir="bundled-skills",
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={"permissions": {"mode": "project-write", "approvals": "on-boundary", "network": False}},
        debug=True,
    )
    selected = runtime.skills_by_ids(["project-ops", "search-ops", "shell-ops"])
    names = set(runtime.allowed_tool_names(selected))

    assert "create_file" in names
    assert "edit_file" in names
    assert "delete_path" in names
    assert "run_checks" not in names
    assert "web_search" in names
    assert "fetch_url" in names
    assert "shell_command" in names


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
    "capability": "project_write",
    "description": "Create file",
    "parameters": {
      "type": "object",
      "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}},
      "required": ["filepath", "content"]
    }
  },
  "read_file": {
    "capability": "project_read",
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
        path = env.project.create_file(args["filepath"], args["content"])
        return {"ok": True, "data": {"filepath": path}, "error": None, "meta": {}}
    if tool_name == "read_file":
        content = env.project.read_file(args["filepath"])
        return {"ok": True, "data": {"content": content}, "error": None, "meta": {}}
    return {"ok": False, "data": None, "error": {"code": "E_UNSUPPORTED", "message": "nope"}, "meta": {}}
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        debug=True,
    )

    ctx = SkillContext(
        user_input="write a file",
        branch_labels=[],
        attachments=["main.py"],
        project_root=str(ws),
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
Project only
""".strip(),
        encoding="utf-8",
    )
    (skills / "s1" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "write_blob": {
    "capability": "project_write",
    "description": "Create file",
    "parameters": {
      "type": "object",
      "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}},
      "required": ["filepath", "content"]
    }
  },
  "read_blob": {
    "capability": "project_read",
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
        path = env.project.create_file(args["filepath"], args["content"])
        return {"ok": True, "data": {"filepath": path}, "error": None, "meta": {}}
    if tool_name == "read_blob":
        content = env.project.read_file(args["filepath"])
        return {"ok": True, "data": {"content": content}, "error": None, "meta": {}}
    return {"ok": False, "data": None, "error": {"code": "E_UNSUPPORTED", "message": "nope"}, "meta": {}}
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    ctx = SkillContext(
        user_input="read a file",
        branch_labels=[],
        attachments=[],
        project_root=str(ws),
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
    (skills / "project-ops").mkdir(parents=True)
    (skills / "search-ops").mkdir(parents=True)

    (skills / "project-ops" / "SKILL.md").write_text(
        """
---
name: project-ops
description: project core tools
version: 1.0.0
tools:
  allowed-tools:
    - create_file
    - read_file
---
Project core.
""".strip(),
        encoding="utf-8",
    )
    (skills / "project-ops" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "create_file": {
    "capability": "project_write",
    "description": "Create file",
    "parameters": {
      "type": "object",
      "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}},
      "required": ["filepath", "content"]
    }
  },
  "read_file": {
    "capability": "project_read",
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
        path = env.project.create_file(args["filepath"], args["content"])
        return {"ok": True, "data": {"filepath": path}, "error": None, "meta": {}}
    if tool_name == "read_file":
        content = env.project.read_file(args["filepath"])
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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    runtime_only_tool_names = set(_tool_names(runtime, []))
    assert runtime_only_tool_names == _always_available_tool_names()

    project_skill = runtime.get_skill("project-ops")
    search_skill = runtime.get_skill("search-ops")
    assert project_skill is not None and search_skill is not None
    merged_tool_names = set(_tool_names(runtime, [project_skill, search_skill]))
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
    (skills / "project-ops").mkdir(parents=True)

    (skills / "project-ops" / "SKILL.md").write_text(
        """
---
name: project-ops
description: project core tools
version: 1.0.0
tools:
  allowed-tools:
    - create_file
---
Project core.
""".strip(),
        encoding="utf-8",
    )
    (skills / "project-ops" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "create_file": {
    "capability": "project_write",
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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    ctx = SkillContext(
        user_input="save a summary to notes.md",
        branch_labels=[],
        attachments=["image.png"],
        project_root=str(ws),
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


def test_core_tool_executes_with_selected_skill(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "project-ops").mkdir(parents=True)

    (skills / "project-ops" / "SKILL.md").write_text(
        """
---
name: project-ops
description: project core tools
version: 1.0.0
tools:
  allowed-tools:
    - create_file
---
Project core.
""".strip(),
        encoding="utf-8",
    )
    (skills / "project-ops" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "create_file": {
    "capability": "project_write",
    "description": "Create file",
    "parameters": {
      "type": "object",
      "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}},
      "required": ["filepath", "content"]
    }
  }
}

def execute(tool_name, args, env):
    path = env.project.create_file(args["filepath"], args["content"])
    return {"ok": True, "data": {"filepath": path}, "error": None, "meta": {}}
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    ctx = SkillContext(
        user_input="write a file",
        branch_labels=[],
        attachments=[],
        project_root=str(ws),
        memory_hits=[],
    )

    loaded = runtime.skill_view("project-ops", "", ctx)
    assert loaded["loaded"] is True
    selected = runtime.select_skills(ctx)
    assert [skill.id for skill in selected] == ["project-ops"]

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
    "capability": "project_read",
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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        debug=True,
    )
    skill = runtime.get_skill("lazy-skill")
    assert skill is not None

    ctx = SkillContext(
        user_input="run lazy tool",
        branch_labels=[],
        attachments=[],
        project_root=str(ws),
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
    raise RuntimeError("SearXNG base URL not configured")
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        debug=False,
    )
    skill = runtime.get_skill("faulty-skill")
    assert skill is not None

    ctx = SkillContext(
        user_input="run faulty tool",
        branch_labels=[],
        attachments=[],
        project_root=str(ws),
        memory_hits=[],
    )
    out = runtime.execute_tool_call("faulty_tool", {}, selected=[skill], ctx=ctx)

    assert out["ok"] is False
    assert out["error"]["code"] == "E_IO"
    assert out["error"]["message"] == "SearXNG base URL not configured"


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
description: Run terminal commands in the project.
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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )

    ctx = SkillContext(
        user_input="Design a bold landing page in HTML and CSS for a boutique hotel.",
        branch_labels=[],
        attachments=[],
        project_root=str(ws),
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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )

    ctx = SkillContext(
        user_input="hi",
        branch_labels=[],
        attachments=[],
        project_root=str(ws),
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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )

    ctx = SkillContext(
        user_input="what is the current situation in iran",
        branch_labels=[],
        attachments=[],
        project_root=str(ws),
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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )
    ctx = SkillContext(
        user_input="look up the latest weather news on the web",
        branch_labels=[],
        attachments=[],
        project_root=str(ws),
        memory_hits=[],
    )
    runtime.skill_view("frontend-design", "", ctx)
    runtime.skill_view("search-ops", "", ctx)

    selected = runtime.select_skills(ctx, top_n=1)

    assert [skill.id for skill in selected] == ["search-ops"]
