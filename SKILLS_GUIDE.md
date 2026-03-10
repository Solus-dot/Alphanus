# Alphanus Skills Guide

Alphanus uses an AgentSkills-style `SKILL.md` manifest with command-backed tool execution.

## 1. Folder Layout

```text
skills/<skill-id>/
  SKILL.md              # required
  scripts/              # optional command handlers referenced by SKILL.md
  tools.py              # optional legacy fallback adapter
  hooks.py              # optional pre_prompt/pre_action/post_response
```

Only `SKILL.md` is required.

## 2. Required Frontmatter

```md
---
name: skill-name
description: What this skill does and when to use it.
---
```

Constraints used in this project:
- `name` must match directory name.
- `name` should be lowercase letters, numbers, and hyphens.
- `description` must be non-empty.

## 3. Supported Optional Frontmatter

```md
---
name: workspace-ops
description: Read and write workspace files safely.
license: Apache-2.0
compatibility: "requires local python3"
allowed-tools: create_file edit_file read_file
metadata:
  version: "1.0.0"
  categories: [coding]
  tags: [file, workspace]
  enabled: true
  triggers:
    keywords: ["file", "edit"]
    file_ext: [".py", ".md"]
  tools:
    required-tools: [create_file]
    disable-model-invocation: false
    definitions:
      - name: create_file
        capability: workspace_write
        description: Create or overwrite a workspace file.
        command: python3 scripts/create_file.py
        timeout-s: 30
        parameters:
          type: object
          properties:
            filepath: { type: string }
            content: { type: string }
          required: [filepath, content]
---
```

Top-level fields recognized by runtime/parser:
- `name` (required)
- `description` (required)
- `license` (optional)
- `compatibility` (optional)
- `allowed-tools` (optional; list, comma-delimited, or space-delimited)
- `required-tools` (optional)
- `tools` (optional)
- `metadata` (optional mapping)

## 4. Command Tool Contract

Preferred tool path is `metadata.tools.definitions` (or `tools.definitions` fallback).

Required per definition:
- `name`
- `capability`
- `description`
- `command`
- `parameters` (JSON schema object)

Optional:
- `timeout-s` (default `30`)
- `confirm-arg` (argument key that requires interactive approval)

Runtime executes `command` in the skill directory with `shell=True` and sends args via stdin and env vars.

Environment variables injected:
- `ALPHANUS_TOOL_NAME`
- `ALPHANUS_TOOL_ARGS_JSON`
- `ALPHANUS_WORKSPACE_ROOT`
- `ALPHANUS_HOME_ROOT`
- `ALPHANUS_MEMORY_PATH`
- `ALPHANUS_MEMORY_MODEL`
- `ALPHANUS_MEMORY_BACKEND`
- `ALPHANUS_MEMORY_EAGER_LOAD`
- `ALPHANUS_CONFIG_JSON`

Tool output contract:
- Final stdout line must be valid JSON object.
- Preferred normalized envelope:

```json
{"ok": true, "data": {}, "error": null, "meta": {}}
```

## 5. Legacy `tools.py` Fallback

Still supported when command definitions are absent.

`tools.py` must expose:
- `TOOL_SPECS: dict[str, dict]`
- `execute(tool_name: str, args: dict, env) -> dict`

Command-defined tools and `tools.py` tools are merged; duplicates are skipped.

## 6. Hooks

Optional `hooks.py` can define:
- `pre_prompt(context) -> str | None`
- `pre_action(context, action_name, args) -> tuple[bool, str]`
- `post_response(context, text) -> None`

Hook failures are non-fatal and do not crash the turn.

## 7. Runtime Flow

1. Runtime loads `skills/*/SKILL.md`.
2. Frontmatter is parsed by `core/skill_parser.py` using `PyYAML`.
3. Manifest is validated (`name`, `description`, semver shape for version, etc.).
4. Command tools are registered.
5. Optional legacy `tools.py` tools are registered.
6. Skills are selected per turn (`selection_mode`, triggers, limits).
7. Selected tools are exposed to the model via OpenAI `tools` payload.
8. Tool calls execute via `core/skills.py` and normalized results return to agent loop.

## 8. TUI Skill Commands

- `/skills`
- `/skill reload`
- `/skill on <id>`
- `/skill off <id>`
- `/skill info <id>`

## 9. Troubleshooting

Skill does not load:
1. Ensure `SKILL.md` begins and ends frontmatter with `---`.
2. Ensure `name` equals directory name.
3. Ensure `description` is non-empty.
4. Ensure `metadata` is a mapping if present.
5. Ensure every command definition has all required keys.

Tool call fails with "no JSON output":
1. Ensure script prints JSON on the final stdout line.
2. Ensure required args are present.
3. Ensure script exits `0` on success.

`shell_command` confirmation issues:
1. Check `confirm-arg` is configured correctly.
2. Check runtime permission flags in `capabilities` config.
