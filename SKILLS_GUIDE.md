# Alphanus Skill Guide

Alphanus follows the `agentskills.io` skill format.

## 1. Required Layout

```text
skills/<skill-id>/
  SKILL.md
  tools.py      # optional unless required-tools is set
  hooks.py      # optional
```

Only `SKILL.md` is supported. Legacy `skill.toml` / `prompt.md` is no longer loaded.

## 2. Minimal `SKILL.md`

```md
---
name: notes-skill
description: Save short notes in the workspace.
version: 1.0.0
categories:
  - coding
tags:
  - note
  - save
tools:
  allowed-tools:
    - save_note
x-alphanus:
  triggers:
    keywords:
      - note
      - save note
    file_ext:
      - .md
---
Use this skill when the user asks to save short notes.

Rules:
- Keep note content concise.
- Store notes under `notes/`.
```

## 3. Frontmatter Fields

Standard fields used:

- `name` (required): must match directory name.
- `description` (required)
- `version` (optional semver)
- `categories` (optional)
- `tags` (optional)
- `tools` (optional)

`tools` supports:

- `allowed-tools`: expose only these tools from `tools.py`
- `required-tools`: fail skill load if any are missing in `TOOL_SPECS`
- `disable-model-invocation`: keep skill loaded but hide tools from model

Alphanus extension (`x-alphanus`) supports lightweight selection hints only:

- `enabled` (optional bool, default `true`)
- `triggers.keywords` (optional)
- `triggers.file_ext` (optional)

No per-skill priority or capability gate is used.

## 4. `tools.py` Contract

`tools.py` must export:

- `TOOL_SPECS: Dict[str, Dict[str, Any]]`
- `execute(tool_name: str, args: Dict[str, Any], env) -> Dict[str, Any]`

Minimal shape:

```python
from typing import Any, Dict

TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "save_note": {
        "capability": "workspace_write",
        "description": "Save a note",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["filename", "content"],
        },
    }
}


def execute(tool_name: str, args: Dict[str, Any], env) -> Dict[str, Any]:
    if tool_name != "save_note":
        return {
            "ok": False,
            "data": None,
            "error": {"code": "E_UNSUPPORTED", "message": f"Unsupported tool: {tool_name}"},
            "meta": {},
        }

    path = env.workspace.create_file(f"notes/{args['filename']}.txt", args["content"])
    return {"ok": True, "data": {"filepath": path}, "error": None, "meta": {}}
```

## 5. Runtime Behavior

1. Runtime loads `skills/*/SKILL.md`.
2. Runtime validates required fields and name/directory match.
3. Runtime loads `tools.py` and enforces allowlist/required tool rules.
4. Skill selection uses keyword and file-extension trigger matches.
5. Selected tools are exposed and dispatched for tool calls.

## 6. TUI Commands

- `/skills`
- `/skill reload`
- `/skill on <id>`
- `/skill off <id>`
- `/skill info <id>`

## 7. Troubleshooting

Skill not loading:

1. Confirm `SKILL.md` exists and starts/ends frontmatter with `---`.
2. Confirm `name` and `description` are present.
3. Confirm `name` equals directory name.
4. Confirm categories are valid enum values.
5. If `required-tools` is set, confirm each tool exists in `TOOL_SPECS`.

Tool not visible:

1. Confirm skill is enabled.
2. Confirm tool exists in `TOOL_SPECS`.
3. If `allowed-tools` is set, confirm the tool is listed.
4. Run `/skill reload`.
