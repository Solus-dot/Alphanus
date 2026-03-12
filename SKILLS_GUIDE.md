# Alphanus Skills Guide

Alphanus uses an AgentSkills-style `SKILL.md` manifest with two supported execution paths:
- native `tools.py` handlers for in-process tools
- command definitions for script-backed tools

## 1. Folder Layout

```text
skills/<skill-id>/
  SKILL.md              # required
  tools.py              # optional native in-process tool handlers
  scripts/              # optional command handlers referenced by SKILL.md
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

## 4. Native `tools.py` Contract

Preferred for bundled/high-frequency tools.

`tools.py` must expose:
- `TOOL_SPECS: dict[str, dict]`
- `execute(tool_name: str, args: dict, env) -> dict | Any`

Each `TOOL_SPECS` entry must include:
- `capability`
- `description`
- `parameters` (JSON schema object)

Important runtime behavior:
- `TOOL_SPECS` is read lazily from source at skill-load time without importing the module.
- The Python module itself is imported only on first execution.
- After first execution, the loaded module is cached.
- Native tools receive `ToolExecutionEnv` directly:
  - `workspace`
  - `memory`
  - `config`
  - `debug`
  - `confirm_shell`

This is the path used by the current built-in `workspace-ops`, `memory-rag`, `shell-ops`, and part of `utilities`.

## 5. Command Tool Contract

Used for external/script-backed tools and desktop-side effects.

Definitions live in `metadata.tools.definitions` (or `tools.definitions` fallback).

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

Native `tools.py` tools and command-defined tools are merged; duplicates are skipped.

## 6. Choosing a Path

Use `tools.py` when:
- the skill is first-party/bundled
- low latency matters
- the tool needs direct access to `workspace` / `memory`
- you want simpler testing and fewer subprocess failure modes

Use command tools when:
- the skill is intended to be shared externally
- the implementation should stay language-agnostic
- subprocess isolation is useful
- the tool performs desktop-side effects or depends on external CLIs

## 7. Hooks

Optional `hooks.py` can define:
- `pre_prompt(context) -> str | None`
- `pre_action(context, action_name, args) -> tuple[bool, str]`
- `post_response(context, text) -> None`

Hook failures are non-fatal and do not crash the turn.

## 8. Runtime Flow

1. Runtime loads `skills/*/SKILL.md`.
2. Frontmatter is parsed by `core/skill_parser.py` using `PyYAML`.
3. Manifest is validated (`name`, `description`, semver shape for version, etc.).
4. `tools.py` metadata is read lazily from source when present.
5. Command tools are registered.
6. Native and command tools are merged into one registry.
7. Native `tools.py` modules are imported only on first execution.
8. Skills are selected per turn (`selection_mode`, triggers, limits).
9. Selected tools are exposed to the model via OpenAI `tools` payload.
10. Tool calls execute via `core/skills.py` and normalized results return to agent loop.

## 9. TUI Skill Commands

- `/skills`
- `/skill reload`
- `/skill on <id>`
- `/skill off <id>`
- `/skill info <id>`

## 10. Troubleshooting

Skill does not load:
1. Ensure `SKILL.md` begins and ends frontmatter with `---`.
2. Ensure `name` equals directory name.
3. Ensure `description` is non-empty.
4. Ensure `metadata` is a mapping if present.
5. Ensure every command definition has all required keys.
6. Ensure `tools.py` contains a literal `TOOL_SPECS` mapping and an `execute()` function.

Tool call fails with "no JSON output":
1. Ensure script prints JSON on the final stdout line.
2. Ensure required args are present.
3. Ensure script exits `0` on success.

`shell_command` confirmation issues:
1. For command-backed tools, check `confirm-arg` is configured correctly.
2. For native shell tools, ensure confirmation logic is implemented in `tools.py`.
3. Check runtime permission flags in `capabilities` config.
