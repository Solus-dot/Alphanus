# Alphanus Skills Guide

Alphanus uses an AgentSkills-style `SKILL.md` format, but the runtime surface is broader than plain `tools.py`.

Today, a trusted skill can contribute behavior through:
- `tools.py` native tools
- `execution.entrypoints` exposed via the generic `run_skill_entrypoint` runtime tool
- bundled runnable scripts exposed via `run_skill_script`
- documented shell/python workflow commands exposed via `run_skill_command`
- optional `hooks.py`
- optional companion agents discovered from the surrounding skill pack

Important limitation:
- frontmatter `tools.definitions` command tools are parsed for compatibility, but they are currently blocked with `disabled_pending_safe_runner`

## 1. Discovery, Trust, and Shadowing

Skill discovery scans:
- upward from the workspace for `skills/`, `.claude/skills`, `.agents/skills`, and `.opencode/skills`
- local skill roots under `.alphanus/skills`, `~/.claude/skills`, `~/.agents/skills`, and `~/.config/opencode/skills`
- the bundled repo `skills/` root
- any extra roots from `skills.load.extra_dirs`

Source priority is:
1. `workspace/local`
2. `bundled`
3. `user/local`
4. `external/local`

If two skills share the same `name`, the higher-priority one wins and the lower-priority one is reported as `shadowed`.

Executable trust policy:
- `workspace/local` and `bundled` roots are trusted and may execute tools, scripts, entrypoints, hooks, and pack agents
- `user/local` and `external/local` roots are metadata-only; executable surfaces are blocked and reported in `/skills` and `/doctor`

## 2. Folder Layout

Minimum skill layout:

```text
skills/<skill-id>/
  SKILL.md
```

Common optional files:

```text
skills/<skill-id>/
  SKILL.md
  tools.py
  hooks.py
  scripts/
  references/
  templates/
  assets/
  helper.py
  validate.sh
```

Notes:
- `SKILL.md` is required.
- Runnable scripts can live under `scripts/` or at the skill root.
- Root-level script candidates are limited to `.py`, `.sh`, `.js`, and `.mjs`.
- `tools.py`, `hooks.py`, and `__init__.py` are not treated as runnable scripts.
- Companion agents are discovered from the surrounding pack root, not from inside a single skill directory. Pack roots may contain `agents/`, `agents-codex/`, `.claude/agents`, `.agents/agents`, or `.config/opencode/agents`.

## 3. Required Frontmatter

Minimum valid manifest:

```md
---
name: workspace-ops
description: Read and write workspace files safely.
---
```

Rules enforced by the parser:
- `name` is required
- `description` is required
- `name` must match the directory name exactly
- `metadata`, when present, must be a mapping
- semantic versions must match semver shape when provided

## 4. Common Optional Frontmatter

Frequently used fields:

```md
---
name: report-pdf
description: Create report PDFs with a declared workflow.
version: 1.0.0
compatibility: requires local python3
tags: [pdf, reports]
categories: [documents]
produces: [.pdf]
allowed-tools: run_skill_entrypoint
required-tools: create_file
user-invocable: true
disable-model-invocation: false
argument-hint: filename=report.pdf
requirements:
  os: [darwin, linux]
  env: [OPENAI_API_KEY]
  commands: [python3]
execution:
  dependencies:
    python: [reportlab]
    commands: [soffice]
  install:
    - uv pip install reportlab
  verify:
    - python3 scripts/check_env.py
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
        required: [filename]
      timeout-s: 30
      cwd: workspace
---
```

Recognized top-level/frontmatter paths include:
- `name`
- `description`
- `version` or `metadata.version`
- `compatibility`
- `tags`
- `categories`
- `produces` or `artifacts`
- `allowed-tools`
- `required-tools`
- `user-invocable`
- `disable-model-invocation`
- `argument-hint`
- `requirements.os`
- `requirements.env`
- `requirements.commands`
- `execution.dependencies.python`
- `execution.dependencies.commands`
- `execution.install`
- `execution.verify`
- `execution.entrypoints`
- `triggers.keywords`
- `triggers.file_ext`
- `format` / vendor-extension metadata such as `openclaw`, `claude`, and `opencode`

## 5. Native `tools.py`

This is the primary executable tool path for bundled and workspace-local skills.

Contract:
- `tools.py` must expose `TOOL_SPECS`
- `tools.py` must expose `execute(tool_name: str, args: dict, env)`

Each `TOOL_SPECS` entry must include:
- `capability`
- `description`
- `parameters` as a JSON-schema-like object

Runtime behavior:
- `TOOL_SPECS` is read lazily from source before import
- the module is imported only on first execution
- the module is cached after first import
- `TOOL_SPECS` must be a static dictionary literal
- dynamic spec construction is not supported

Native tools receive `ToolExecutionEnv`, which includes:
- `workspace`
- `memory`
- `config`
- `debug`
- `confirm_shell`
- `spawn_skill_agent`
- `request_user_input`

## 6. `execution.entrypoints`

Structured execution entrypoints are the preferred way to declare artifact workflows.

Each entrypoint supports:
- `name`
- `description`
- `command`
- `parameters`
- `intents`
- `produces`
- `install`
- `verify`
- `timeout-s`
- `cwd` (`workspace` or `skill`)

Current constraints:
- only `tool: shell_command` is accepted
- `parameters` must be a mapping
- `timeout-s` must be `> 0`
- `cwd` must be `workspace` or `skill`

Runtime behavior:
- relevant entrypoints are exposed through the generic `run_skill_entrypoint` tool
- if multiple active skills expose relevant entrypoints, the model must supply `skill_id`
- entrypoint relevance is filtered by task intent and requested artifact extension when possible

## 7. Bundled Scripts

CLI-style scripts can be exposed without writing a native tool.

What counts as a runnable script:
- `scripts/*.py`
- `scripts/*.sh`
- `scripts/*.js`
- `scripts/*.mjs`
- root-level `.py`, `.sh`, `.js`, `.mjs` files

What the runtime checks before exposure:
- interpreter exists
- Python helper modules referenced by the script are available
- the script stays inside the skill root
- the script is from a trusted executable root

Relevant scripts are exposed through the generic `run_skill_script` tool.

## 8. Documented Workflow Commands

Alphanus can also extract declared shell/python workflow commands from the skill prompt.

Commands are discovered from:
- fenced shell blocks such as ```` ```bash ````
- inline backticked commands in the prompt when they look like real workflow commands

Examples that can become runtime-visible:
- `uv sync`
- `python3 scripts/create_report.py output.pdf`
- `npm install`

Runtime behavior:
- visible commands are exposed through `run_skill_command`
- only commands already documented by the skill are allowed
- commands are executed with argv parsing, not a shell string runner
- shell control operators and fallback chaining are still blocked by workspace policy

This path is useful for packaged setup/build flows when a full entrypoint contract would be overkill.

## 9. Runtime Skill Tools

The runtime always registers several non-skill-specific tools:
- `skills_list`
- `skill_view`
- `skill_manage`
- `request_user_input` unless `runtime.ask_user_tool = false`

Additional runtime tools appear when relevant:
- `read_skill_resource`
- `run_skill_command`
- `run_skill_script`
- `run_skill_entrypoint`
- `spawn_skill_agent`

What they do:
- `skills_list`: list enabled skills with minimal metadata
- `skill_view`: load a skill's full prompt into the session or read a specific file inside the skill root
- `skill_manage`: create/edit/patch/delete workspace-local skills under `.alphanus/skills`
- `read_skill_resource`: read a bundled file from an active skill
- `run_skill_command`: run one declared shell/python workflow command from an active skill
- `run_skill_script`: run one bundled script from an active skill
- `run_skill_entrypoint`: run one declared `execution.entrypoints` contract from an active skill
- `spawn_skill_agent`: start, inspect, or wait for a companion agent
- `request_user_input`: ask the user a structured follow-up question and pause the workflow

## 10. Session Loading Model

Skill enablement and session loading are separate.

Global enablement:
- `/skill-on <id>` enables a skill
- `/skill-off <id>` disables a skill

Session loading:
- `skill_view(name)` loads the skill prompt and appends the skill id to `loaded_skill_ids`
- loaded skill ids are persisted with the chat session
- `/skill-unload <id>` and `/skill-unload-all` remove session-loaded skills

Practical consequence:
- a skill can be enabled but not loaded into the current session
- optional runtime surfaces such as `run_skill_script` or `run_skill_entrypoint` only matter once the skill is active for the turn

## 11. Hooks

Optional `hooks.py` may define:
- `pre_prompt(context) -> str | None`
- `pre_action(context, action_name, args) -> tuple[bool, str]`
- `post_response(context, text) -> None`

Hook failures are logged and treated as non-fatal.

## 12. Command Tool Definitions

AgentSkills-style frontmatter command tools are still parsed:

```yaml
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
        required: [text]
```

Current runtime status:
- definitions are parsed and validated
- the skill is marked with `blocked_features = ["command_tools"]`
- `validation_errors` includes `command_tools are disabled_pending_safe_runner`
- no executable tool is registered for that definition

If you need an executable path today, use `tools.py`, `execution.entrypoints`, bundled scripts, or a documented workflow command instead.

## 13. Troubleshooting

Skill does not load:
1. Ensure `SKILL.md` starts and ends frontmatter with `---`.
2. Ensure `name` matches the directory name.
3. Ensure `description` is non-empty.
4. Ensure `metadata`, `tools`, `triggers`, and `execution` values are mappings when present.
5. Ensure entrypoints use valid `parameters`, `timeout-s`, and `cwd`.

Skill shows as blocked:
1. Check whether the root is `user/local` or `external/local`; those are metadata-only.
2. Check whether a higher-priority skill with the same name shadowed it.
3. Check `validation_errors`, `validation_warnings`, and `blocked_features` in `/skills` or `/doctor`.

Script does not appear:
1. Ensure it is a CLI-style `.py`, `.sh`, `.js`, or `.mjs` file.
2. Ensure the interpreter exists.
3. Ensure Python dependencies are importable for that configured Python executable.
4. Ensure the skill is in a trusted executable root.

Entrypoint is not exposed:
1. Ensure the skill is active for the turn.
2. Ensure `allowed-tools` does not accidentally exclude `run_skill_entrypoint`.
3. Ensure the current task intent and requested artifact type still match the entrypoint.

`run_skill_command` is rejected:
1. The command was probably not declared in the skill prompt.
2. Only documented commands exposed for the current turn are allowed.
3. The command still goes through shell safety policy.

`request_user_input` does not appear:
1. Check `runtime.ask_user_tool`.
2. The runtime tool may be disabled in config even though the skill itself is fine.
