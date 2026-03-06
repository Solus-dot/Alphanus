# Alphanus Skill Guide (Simple Version)

This guide shows the easiest way to add a new skill in Alphanus.

The key idea:

- `skill.toml` controls **when** a skill is active and **what capabilities** it is allowed to use.
- `tools.py` defines **what tools exist** and **how each tool executes**.

You usually do **not** edit `core/skills.py` when adding normal new skills/tools.

---

## 1. Minimal Skill Structure

Create:

```text
skills/<skill-id>/
  skill.toml
  prompt.md
  tools.py
```

Optional:

- `hooks.py` for policy checks or prompt augmentation

---

## 2. Basic Example Skill (Notes Skill)

We will create a basic skill that saves a note to a file in the workspace.

## Step 1: Create the folder

```bash
mkdir -p skills/notes-skill
```

## Step 2: Add `skill.toml`

```toml
schema_version = "1.0.0"
id = "notes-skill"
name = "Notes Skill"
version = "1.0.0"
description = "Save short notes into workspace files"
enabled = true
priority = 50

[triggers]
keywords = ["note", "remember this", "save note"]
file_ext = []
capabilities = ["workspace_write"]
```

## Step 3: Add `prompt.md`

```md
Use this skill when the user asks to save a short note.

Rules:
- Keep note content concise.
- Store notes under `notes/` in the workspace.
```

## Step 4: Add `tools.py`

```python
from typing import Any, Dict

TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "save_note": {
        "capability": "workspace_write",
        "description": "Save a note to notes/<filename>.txt in workspace",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["filename", "content"]
        }
    }
}


def execute(tool_name: str, args: Dict[str, Any], env) -> Dict[str, Any]:
    if tool_name == "save_note":
        filename = args["filename"].strip().replace("/", "_")
        path = f"notes/{filename}.txt"
        written = env.workspace.create_file(path, args["content"])
        return {
            "ok": True,
            "data": {"filepath": written},
            "error": None,
            "meta": {},
        }

    return {
        "ok": False,
        "data": None,
        "error": {"code": "E_UNSUPPORTED", "message": f"Unsupported tool: {tool_name}"},
        "meta": {},
    }
```

## Step 5: Reload in TUI

```text
/skill reload
/skills
/skill info notes-skill
```

---

## 3. How `skill.toml` Works

`skill.toml` is parsed by the runtime and turned into a `SkillManifest`.

Where in code:

- Loader: `core/skills.py` (`load_skills`)
- Selection/scoring: `core/skills.py` (`score_skills`, `select_skills`)

Field-by-field behavior:

1. `schema_version`
- Used for compatibility checks.
- Major-version mismatch is rejected.

2. `id`
- Unique skill ID.
- Should match folder name.

3. `enabled`
- If `false`, skill is ignored.
- Can be toggled at runtime via `/skill on|off <id>`.

4. `priority`
- Base score for selection.
- Higher priority means more likely to be selected.

5. `[triggers].keywords`
- If user input contains these, score gets boosted.

6. `[triggers].file_ext`
- If input/attachments mention these extensions, score gets boosted.

7. `[triggers].capabilities`
- Policy gate for tools.
- A tool is only allowed if its `TOOL_SPECS[tool].capability` is listed here.

Important: capability mismatch causes fail-closed policy errors (`E_POLICY`).

---

## 4. How `tools.py` Works

`tools.py` is loaded dynamically per skill.

It must provide:

1. `TOOL_SPECS`
2. `execute(tool_name, args, env)`

### 4.1 `TOOL_SPECS`

`TOOL_SPECS` is what the model sees in the OpenAI-compatible `tools` array.

For each tool, you define:

- `capability`: policy capability required
- `description`: what the tool does
- `parameters`: JSON Schema object for tool arguments

### 4.2 `execute(...)`

`execute(...)` runs when the model calls the tool.

Arguments:

- `tool_name`: selected tool name
- `args`: parsed JSON arguments from tool call
- `env`: execution environment

`env` contains:

- `env.workspace`: workspace adapter (`create_file`, `read_file`, `run_shell_command`, etc.)
- `env.memory`: memory adapter (`add_memory`, `search`, etc.)
- `env.config`: global config dict
- `env.confirm_shell`: callback for shell confirmation (if needed)
- `env.debug`: debug mode boolean

Return must be a normalized envelope:

```json
{
  "ok": true,
  "data": {},
  "error": null,
  "meta": {"duration_ms": 12}
}
```

On failure:

```json
{
  "ok": false,
  "data": null,
  "error": {"code": "E_POLICY", "message": "..."},
  "meta": {"duration_ms": 4}
}
```

Common error codes:

- `E_POLICY`
- `E_VALIDATION`
- `E_TIMEOUT`
- `E_IO`
- `E_NOT_FOUND`
- `E_UNSUPPORTED`

---

## 5. Runtime Flow (What Actually Happens)

1. Runtime loads enabled skills from `skills/*/skill.toml`.
2. Runtime loads each skill's `prompt.md` and optional `tools.py`.
3. For a user turn, runtime scores and selects top skills.
4. Agent composes one system message + selected skill guidance.
5. Agent exposes only tools from selected skills that pass capability policy.
6. Model emits `tool_calls`.
7. Runtime dispatches tool call to the owning skill's `tools.py::execute`.
8. Tool result is appended to history and model continues.

---

## 6. Troubleshooting

Tool not visible:

1. Confirm `tools.py` has valid `TOOL_SPECS` and callable `execute`.
2. Confirm skill is enabled.
3. Confirm `/skill reload` was run.
4. Confirm capability in `skill.toml` matches tool capability.

Tool called but denied:

1. Capability mismatch (`skill.toml` vs `TOOL_SPECS`).
2. Skill not selected for that prompt.
3. Hook `pre_action` denied it.

---

## 7. Quick Checklist

1. Create `skills/<id>/skill.toml`.
2. Create `skills/<id>/prompt.md`.
3. Create `skills/<id>/tools.py`.
4. Ensure capability alignment (`skill.toml` <-> `TOOL_SPECS`).
5. Reload skills in TUI.
6. Test with a real prompt.
