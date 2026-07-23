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

def test_tool_is_blocked_for_local_project_uses_capability_metadata(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "project-ops").mkdir(parents=True)
    (skills / "local-search").mkdir(parents=True)
    (skills / "search-ops").mkdir(parents=True)
    (skills / "utilities").mkdir(parents=True)

    (skills / "project-ops" / "SKILL.md").write_text(
        """
---
name: project-ops
description: Project tools
tools:
  allowed-tools:
    - read_blob
---
Read files.
""".strip(),
        encoding="utf-8",
    )
    (skills / "project-ops" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "read_blob": {
    "capability": "project_read",
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
    - search_project_files
---
Utility helpers.
""".strip(),
        encoding="utf-8",
    )
    (skills / "local-search" / "SKILL.md").write_text(
        """
---
name: local-search
description: Local search tools
tools:
  allowed-tools:
    - search_local_files
---
Search project files.
""".strip(),
        encoding="utf-8",
    )
    (skills / "local-search" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "search_local_files": {
    "capability": "local_search",
    "description": "Search local files",
    "parameters": {"type": "object", "properties": {}, "required": []}
  }
}

def execute(tool_name, args, env):
    return {"ok": True, "data": {}, "error": None, "meta": {}}
""".strip(),
        encoding="utf-8",
    )
    (skills / "search-ops" / "SKILL.md").write_text(
        """
---
name: search-ops
description: Local retrieval tools
tools:
  allowed-tools:
    - retrieve_knowledge
    - retrieval_stats
---
Retrieve indexed project knowledge.
""".strip(),
        encoding="utf-8",
    )
    (skills / "search-ops" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "retrieve_knowledge": {
    "capability": "knowledge_retrieve",
    "description": "Retrieve local knowledge",
    "parameters": {"type": "object", "properties": {}, "required": []}
  },
  "retrieval_stats": {
    "capability": "retrieval_stats",
    "description": "Inspect the local retrieval index",
    "parameters": {"type": "object", "properties": {}, "required": []}
  }
}

def execute(tool_name, args, env):
    return {"ok": True, "data": {}, "error": None, "meta": {}}
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
  },
  "search_project_files": {
    "capability": "utility_file_search",
    "description": "Search project filenames",
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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )

    assert runtime.tool_is_blocked_for_local_project("read_blob") is False
    assert runtime.tool_is_blocked_for_local_project("search_local_files") is False
    assert runtime.tool_is_blocked_for_local_project("retrieve_knowledge") is False
    assert runtime.tool_is_blocked_for_local_project("retrieval_stats") is False
    assert runtime.tool_is_blocked_for_local_project("search_project_files") is False
    assert runtime.tool_is_blocked_for_local_project("open_url") is True
    assert runtime.tool_is_blocked_for_local_project("request_user_input") is False
    assert runtime.tool_is_blocked_for_local_project("unknown_tool") is True


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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    skill = runtime.get_skill("cmd-skill")
    assert skill is not None
    assert skill.available is False
    assert "definitely-not-a-real-binary-xyz" in skill.availability_reason


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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("docs-skill")
    assert skill is not None

    block = runtime.compose_skill_block(
        [skill],
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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    ctx = SkillContext(
        user_input="load",
        branch_labels=[],
        attachments=[],
        project_root=str(ws),
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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    listed = runtime.list_skills()
    assert len(listed) == 1
    assert listed[0].available is False
    assert listed[0].execution_allowed is False
    assert listed[0].validation_errors == ["missing required tools: create_file"]


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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("lazy-skill")
    assert skill is not None
    assert skill.prompt is None

    block = runtime.compose_skill_block([skill], context_limit=1024)
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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("hooked-skill")
    assert skill is not None
    assert skill.validation_errors == []


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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    report = runtime.skill_health_report()

    assert report[0]["provenance"] == "user/skills"
    assert report[0]["availability_code"] == "ready"


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
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    assert runtime.get_skill("home-helper") is None
    assert all(skill.id != "home-helper" for skill in runtime.list_skills())


def test_only_bundled_root_is_used_for_discovery(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    bundled = tmp_path / "bundled"
    bundled_skill = bundled / "dup-skill"
    project_skill = ws / ".claude" / "skills" / "dup-skill"
    home.mkdir()
    ws.mkdir()
    bundled_skill.mkdir(parents=True)
    project_skill.mkdir(parents=True)

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
    (project_skill / "SKILL.md").write_text(
        """
---
name: dup-skill
description: project
version: 2.0.0
---
Project version.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(bundled),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    active = runtime.get_skill("dup-skill")
    assert active is not None
    assert active.description == "bundled"
    assert runtime.skill_provenance_label(active) == "user/skills"
    assert len(runtime.list_skills()) == 1


def test_runtime_discovers_user_bundled_and_configured_skill_roots(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    user_root = tmp_path / "user-skills"
    bundled_root = tmp_path / "bundled-skills"
    configured_root = tmp_path / "extra-skills"
    home.mkdir()
    ws.mkdir()
    for root, skill_id, description in (
        (user_root, "user-helper", "user helper"),
        (bundled_root, "bundled-helper", "bundled helper"),
        (configured_root, "configured-helper", "configured helper"),
    ):
        skill_dir = root / skill_id
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"""
---
name: {skill_id}
description: {description}
version: 1.0.0
---
{description}.
""".strip(),
            encoding="utf-8",
        )

    runtime = SkillRuntime(
        skills_dir=str(user_root),
        bundled_skills_dir=str(bundled_root),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={"skills": {"paths": [str(configured_root)]}},
    )

    user_skill = runtime.get_skill("user-helper")
    bundled_skill = runtime.get_skill("bundled-helper")
    configured_skill = runtime.get_skill("configured-helper")
    assert user_skill is not None
    assert bundled_skill is not None
    assert configured_skill is not None
    assert runtime.skill_provenance_label(user_skill) == "user/skills"
    assert runtime.skill_provenance_label(bundled_skill) == "bundled"
    assert runtime.skill_provenance_label(configured_skill) == "configured"


def test_user_skill_root_overrides_bundled_duplicate(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    user_root = tmp_path / "user-skills"
    bundled_root = tmp_path / "bundled-skills"
    home.mkdir()
    ws.mkdir()
    for root, description in ((user_root, "user version"), (bundled_root, "bundled version")):
        skill_dir = root / "dup-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"""
---
name: dup-skill
description: {description}
version: 1.0.0
---
{description}.
""".strip(),
            encoding="utf-8",
        )

    runtime = SkillRuntime(
        skills_dir=str(user_root),
        bundled_skills_dir=str(bundled_root),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )

    active = runtime.get_skill("dup-skill")
    assert active is not None
    assert active.description == "user version"
    assert runtime.skill_provenance_label(active) == "user/skills"
    assert any("duplicate skill id" in item for item in active.validation_errors)


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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("asker")
    assert skill is not None
    ctx = SkillContext(
        user_input="use skill asker",
        branch_labels=[],
        attachments=[],
        project_root=str(ws),
        memory_hits=[],
        explicit_skill_id="asker",
    )

    out = runtime.execute_tool_call(
        "request_user_input",
        {"question": "Choose one", "options": ["a", "b"]},
        selected=[skill],
        ctx=ctx,
        request_user_input=lambda args: (
            lambda raw_options: {
                "question": args["question"],
                "options": list(raw_options) if isinstance(raw_options, list) else [],
                "awaiting_user_input": True,
            }
        )(args.get("options")),
    )

    assert out["ok"] is True
    assert out["data"]["awaiting_user_input"] is True
    assert out["data"]["options"] == ["a", "b"]


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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("doc-helper")
    assert skill is not None
    ctx = SkillContext(
        user_input="read the doc helper resource",
        branch_labels=[],
        attachments=[],
        project_root=str(ws),
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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    alpha = runtime.get_skill("alpha")
    beta = runtime.get_skill("beta")
    assert alpha is not None and beta is not None
    ctx = SkillContext(
        user_input="use both skills",
        branch_labels=[],
        attachments=[],
        project_root=str(ws),
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
        bundled_skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
    )
    skill = runtime.get_skill("script-skill")
    assert skill is not None

    docx_ctx = SkillContext(
        user_input="set up report.docx",
        branch_labels=[],
        attachments=[],
        project_root=str(ws),
        memory_hits=[],
    )
    png_ctx = SkillContext(
        user_input="create image.png",
        branch_labels=[],
        attachments=[],
        project_root=str(ws),
        memory_hits=[],
    )

    calls = {"count": 0}

    def fake_tool_schemas(names, selected=None, ctx=None):
        calls["count"] += 1
        return [{"ctx": getattr(ctx, "user_input", ""), "names": list(names)}]

    runtime._tool_schema_builder.build = fake_tool_schemas  # type: ignore[method-assign]

    docx_tools = runtime.tools_for_turn([skill], ctx=docx_ctx)
    png_tools = runtime.tools_for_turn([skill], ctx=png_ctx)

    assert calls["count"] == 1
    assert docx_tools == png_tools
    assert docx_tools[0]["ctx"] == "set up report.docx"


def test_unexpected_tool_exception_message_preserves_exception_type(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "explode-skill").mkdir(parents=True)
    (skills / "explode-skill" / "SKILL.md").write_text(
        """
---
name: explode-skill
description: tool raises unexpected exception
version: 1.0.0
allowed-tools:
  - explode_tool
---
Explode.
""".strip(),
        encoding="utf-8",
    )
    (skills / "explode-skill" / "tools.py").write_text(
        """
TOOL_SPECS = {
  "explode_tool": {
    "capability": "project_read",
    "description": "Explodes",
    "parameters": {"type": "object", "properties": {}, "required": []}
  }
}

def execute(tool_name, args, env):
    raise ArithmeticError("missing config")
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
    skill = runtime.get_skill("explode-skill")
    assert skill is not None
    ctx = SkillContext(
        user_input="explode",
        branch_labels=[],
        attachments=[],
        project_root=str(ws),
        memory_hits=[],
    )

    out = runtime.execute_tool_call("explode_tool", {}, selected=[skill], ctx=ctx)

    assert out["ok"] is False
    assert out["error"]["code"] == "E_IO"
    assert "Tool raised ArithmeticError" in out["error"]["message"]
