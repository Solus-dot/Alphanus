# Alphanus Skills Guide

Alphanus skills use an AgentSkills-style `SKILL.md` manifest. Runtime exposure is session-driven: load a skill, then use its tools or `run_skill`.

## 1. Discovery and Loading

Skill roots are discovered from:
- configured repo `skills/` root only

Enablement and loading are separate:
- `/skill-on <id>` / `/skill-off <id>` control global enablement
- `skill_view(name)` loads the skill for the active session
- loaded skill ids persist with the session
- `/skill-unload <id>` and `/skill-unload-all` remove session-loaded skills

Skills are expected to live at `<repo>/skills/<skill-id>/SKILL.md`.

## 2. Layout

Minimum:

```text
skills/<skill-id>/
  SKILL.md
```

Optional:

```text
skills/<skill-id>/
  SKILL.md
  tools.py
  scripts/
  references/
  templates/
  assets/
```

Script candidates are `.py`, `.sh`, `.js`, `.mjs`. `tools.py`, `hooks.py`, and `__init__.py` are never treated as runnable scripts.

## 3. Required Frontmatter

```md
---
name: workspace-ops
description: Read and write workspace files safely.
---
```

Rules:
- `name` required
- `description` required
- `name` must match directory name
- optional mapping fields (`metadata`, `tools`, `execution`) must be mappings

## 4. Optional Frontmatter

Common fields:
- `version` / `metadata.version`
- `format` / `metadata.format` (adapter hint)
- `tags`, `categories`, `produces`
- `allowed-tools`, `required-tools`
- `user-invocable`, `disable-model-invocation`
- `requirements.os`, `requirements.env`, `requirements.commands`
- `execution.entrypoints` (structured runnable commands)
- `execution.dependencies`, `execution.install`, `execution.verify`

Adapter selection details:
- adapter defaults to `agentskills`
- adapter can be set directly with `format` or `metadata.format`
- vendor requirement blocks under `metadata.openclaw`, `metadata.claude`, or `metadata.opencode` can select the adapter and add requirement constraints

## 5. Executable Surfaces

### `tools.py` native tools

Contract:
- expose `TOOL_SPECS` (static dict literal)
- expose `execute(tool_name, args, env)`

Notes:
- parsed lazily, imported on first execution
- native tools are only callable when the owning skill is selected (session-loaded)

### `run_skill`

Unified executor for skill entrypoints and bundled scripts.

Parameters:
- `skill_id` (required when multiple selected skills match)
- exactly one of `entrypoint` or `script`
- `params` (object payload)
- `argv` (optional script argv list)
- `stdin` (optional script stdin)
- `timeout_s` (optional timeout override)

## 6. Runtime Tools

Always available:
- `skills_list`
- `skill_view`
- `request_user_input` (unless `runtime.ask_user_tool=false`)
- `run_skill` (exposed when at least one selected skill has entrypoints or runnable scripts)

Conditionally available by selected skills:
- native tools from `tools.py`

## 7. Command Tool Definitions

`tools.definitions` command tools in frontmatter are not supported.
Use `tools.py` or `execution.entrypoints` for executable behavior.

## 8. Troubleshooting

Skill not loading:
1. Check frontmatter delimiters and required fields.
2. Ensure `name` matches directory name.
3. Check `validation_errors` from `/skills` or `/doctor`.

Tool not callable:
1. Ensure the skill is enabled and loaded in the current session.
2. Ensure `allowed-tools` includes the tool name when set.
3. Check `disable-model-invocation` if invoking from model tool calls.

`run_skill` not callable:
1. Ensure selected skill exposes a valid entrypoint or runnable script.
2. Provide `skill_id` when multiple selected skills match.
3. Provide exactly one of `entrypoint` or `script`.
