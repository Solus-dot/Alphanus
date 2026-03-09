# Alphanus Skill Guide

Alphanus follows the `agentskills.io` skill format with command-backed tool execution.

## 1. Required Layout

```text
skills/<skill-id>/
  SKILL.md
  scripts/      # optional command handlers referenced from SKILL.md
  tools.py      # optional legacy fallback adapter
  hooks.py      # optional
```

Only `SKILL.md` is required. Legacy `skill.toml` / `prompt.md` is not loaded.

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
  definitions:
    - name: save_note
      capability: workspace_write
      description: Save a note.
      command: python3 scripts/ops.py save_note
      timeout-s: 30
      parameters:
        type: object
        properties:
          filename:
            type: string
          content:
            type: string
        required:
          - filename
          - content
x-alphanus:
  triggers:
    keywords:
      - note
      - save note
    file_ext:
      - .md
---
Use this skill when the user asks to save short notes.
```

## 3. Frontmatter Fields

Standard fields:

- `name` (required): must match directory name
- `description` (required)
- `version` (optional semver)
- `categories` (optional)
- `tags` (optional)
- `tools` (optional)

`tools` supports:

- `allowed-tools`: expose only these tool names
- `required-tools`: fail skill load if any required tool is missing
- `disable-model-invocation`: keep skill loaded but hide tools from the model
- `definitions`: command-backed tool definitions (preferred)

Command definition fields:

- `name` (required)
- `capability` (required)
- `description` (required)
- `parameters` JSON schema object (required)
- `command` bash command string (required)
- `timeout-s` integer seconds (optional, default `30`)
- `confirm-arg` argument key requiring interactive approval (optional)

Alphanus extension (`x-alphanus`) supports lightweight selection hints:

- `enabled` (optional bool, default `true`)
- `triggers.keywords` (optional)
- `triggers.file_ext` (optional)

No per-skill priority or capability gate is used.

## 4. Command Execution Contract

Runtime executes each tool definition `command` through bash in the skill directory.

Runtime-provided environment variables:

- `ALPHANUS_TOOL_NAME`
- `ALPHANUS_TOOL_ARGS_JSON`
- `ALPHANUS_WORKSPACE_ROOT`
- `ALPHANUS_HOME_ROOT`
- `ALPHANUS_MEMORY_PATH`
- `ALPHANUS_MEMORY_MODEL`
- `ALPHANUS_MEMORY_BACKEND`
- `ALPHANUS_MEMORY_EAGER_LOAD`
- `ALPHANUS_CONFIG_JSON`

The command must print a single JSON result object (or print diagnostics and the JSON result on the final line):

```json
{"ok": true, "data": {}, "error": null, "meta": {}}
```

## 5. Legacy `tools.py` Fallback

Legacy adapters are still supported:

- `TOOL_SPECS: Dict[str, Dict[str, Any]]`
- `execute(tool_name: str, args: Dict[str, Any], env) -> Dict[str, Any]`

Use this only when command-backed tools are not practical.

## 6. Runtime Behavior

1. Runtime loads `skills/*/SKILL.md`.
2. Runtime validates required fields and name/directory match.
3. Runtime loads command definitions (`tools.definitions`) first.
4. Runtime optionally loads legacy `tools.py` and merges non-duplicate tools.
5. Runtime enforces allowlist/required tool rules.
6. Skill selection uses keyword and file-extension trigger matches.
7. Selected tools are exposed and dispatched for tool calls.

## 7. TUI Commands

- `/skills`
- `/skill reload`
- `/skill on <id>`
- `/skill off <id>`
- `/skill info <id>`

## 8. Troubleshooting

Skill not loading:

1. Confirm `SKILL.md` exists and frontmatter starts/ends with `---`.
2. Confirm `name` and `description` are present.
3. Confirm `name` equals directory name.
4. Confirm categories are valid enum values.
5. If `required-tools` is set, confirm each tool is present.
6. If using command tools, confirm each `definitions` item has all required fields.

Tool not visible:

1. Confirm skill is enabled.
2. Confirm tool name is in `allowed-tools` (if set).
3. Confirm command script is executable in your environment.
4. Confirm command prints JSON result.
5. Run `/skill reload`.
