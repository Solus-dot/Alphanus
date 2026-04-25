<p align="center">
  <h1 align="center">Alphanus</h1>
  <p align="center">
    Local-first coding assistant with a Textual TUI, persistent memory, session-aware skills, and a branchable conversation tree.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="Status: alpha">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/backend-llama.cpp-black" alt="Backend: llama.cpp">
  <img src="https://img.shields.io/badge/platform-local--first-lightgrey" alt="Local first">
</p>

<p align="center">
  Built for running against a local <code>llama.cpp</code> server. Your workspace stays local. Secrets stay in environment variables.
</p>

---

## Overview

Alphanus is an experimental coding assistant for local use.

It combines a terminal UI, a streaming chat-completions loop, persistent lexical memory, explicit skills, and workspace-aware tooling into a system that is meant to be inspectable and hackable rather than hidden behind a hosted product.

The current implementation is built around `llama.cpp` and a Textual interface, with support for named sessions, branching conversation history, configurable skills, and controlled workspace or shell operations.

---

## Features

- Streaming agent loop with separate reasoning and content handling
- `llama.cpp` `/v1` API integration with OpenAI-style `tool_calls`
- Textual UI with live tool previews, code block popups, config editing, and support-bundle export
- Dual command palettes: slash palette (`/` or `Ctrl+P`) and quick palette (`Ctrl+K`) with actionable commands, sessions, files, and skills
- Switchable built-in themes with runtime picker support (`/theme`) and persisted `tui.theme` preference
- Runtime profiles: `standard` (default) and `minimal` (minimal reliable mode)
- Named autosaved sessions with a branchable conversation tree
- Persistent lexical memory with score thresholds and storage recovery safeguards
- Explicit session-loaded skills with enable and disable controls
- Web search support for time-sensitive answers
- Workspace-aware tooling with confirmation gates and explicit session-loaded skill execution
- First-class per-turn trace journal (pass payloads, selected skills, tool calls/results, timings)

---

## What's New

Recent user-facing additions include:

- `Ctrl+K` quick palette with an actionable catalog (commands, session switching, file attach, and skill load or unload)
- Built-in theme system (`classic`, `soft`, `catppuccin-mocha`, `catppuccin-macchiato`, `tokyonight-moon`, `gruvbox-dark-soft`)
- `/theme` command in the TUI to preview/apply themes and persist the selection
- Scoped `init` setup (`all`, `workspace`, `model`, `search`, `theme`) plus `--theme` support for non-interactive setup
- `runtime.profile` with `minimal` mode for a smaller, predictable tool surface
- Capability permission profiles (`safe`, `workspace`, `full`) for tool-scope hardening
- Turn-level trace capture in the journal (`turn_trace`) for replay and debugging

---

## Status

Alphanus is currently **alpha** software.

Current priorities are:

- local-first workflows
- power-user usability
- explicit runtime behavior
- safety boundaries around tools and shell access

Expect rough edges, changing configuration, and incomplete backend support.

---

## Requirements

- Python `>=3.11`
- [`uv`](https://docs.astral.sh/uv/)
- A running `llama.cpp` server exposing:
  - `http://127.0.0.1:8080/v1/chat/completions`
  - `http://127.0.0.1:8080/v1/models`

Backend support is currently focused on `llama.cpp`. Other inference backends are not yet supported or documented.

---

## Quick Start

Install dependencies:

```bash
uv sync --extra dev
```

Set any required environment variables:

* `TAVILY_API_KEY` when `search.provider = "tavily"`
* `BRAVE_SEARCH_API_KEY` when `search.provider = "brave"`
* `ALPHANUS_AUTH_HEADER` for authenticated model endpoints (`Header-Name: value`)
* `AUTH_HEADER` as a fallback if `ALPHANUS_AUTH_HEADER` is unset

Run Alphanus:

```bash
uv run alphanus init
uv run alphanus doctor
uv run alphanus
```

Useful flags:

```bash
uv run alphanus --debug
uv run alphanus --dangerously-skip-permissions
uv run alphanus doctor --json --debug
uv run alphanus run --debug
uv run alphanus init theme --non-interactive --theme catppuccin-macchiato
```

Notes:

* `.env` is loaded from `~/.alphanus/.env` without overriding already-set environment variables
* `uv run alphanus` requires `~/.alphanus/config/global_config.json`; if missing, run `uv run alphanus init`
* `uv run alphanus doctor` validates config, env, endpoint reachability, and workspace permissions

---

## Initialize the New System

The runtime now separates app state from your workspace files.

State lives in:

* `~/.alphanus/config/global_config.json` (config)
* `~/.alphanus/.env` (environment secrets)
* `~/.alphanus/sessions/` (sessions)
* `~/.alphanus/memory/` (memory)

Workspace files remain at `workspace.path` (default `~/Desktop/Alphanus-Workspace`).

Interactive setup:

```bash
uv run alphanus init
```

Initialize only one section:

```bash
uv run alphanus init theme
```

Non-interactive setup:

```bash
uv run alphanus init --non-interactive \
  --workspace-path ~/Desktop/Alphanus-Workspace \
  --model-endpoint http://127.0.0.1:8080/v1/chat/completions \
  --models-endpoint http://127.0.0.1:8080/v1/models \
  --search-provider tavily \
  --theme catppuccin-mocha
```

Init section values: `all`, `workspace`, `model`, `search`, `theme`.

Validate then launch:

```bash
uv run alphanus doctor
uv run alphanus
```

Machine-readable doctor output:

```bash
uv run alphanus doctor --json
```

---

## Configuration

Global config is stored at:

* `~/.alphanus/config/global_config.json`

At load and save time, Alphanus:

* merges missing keys from built-in defaults
* normalizes types and clamps invalid values
* strips secret-like fields from disk and from the `/config` editor
* enforces same-host `model_endpoint` and `models_endpoint` unless `agent.allow_cross_host_endpoints = true`

A trimmed example:

```json
{
  "schema_version": "1.0.0",
  "agent": {
    "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
    "models_endpoint": "http://127.0.0.1:8080/v1/models",
    "request_timeout_s": 180,
    "readiness_timeout_s": 30,
    "enable_thinking": true,
    "allow_cross_host_endpoints": false,
    "max_tokens": null
  },
  "workspace": {
    "path": "~/Desktop/Alphanus-Workspace"
  },
  "capabilities": {
    "permission_profile": "full"
  },
  "memory": {
    "min_score_default": 0.3,
    "recall_min_score_default": 0.18,
    "replace_min_score_default": 0.72,
    "backup_revisions": 2
  },
  "context": {
    "keep_last_n": 10
  },
  "runtime": {
    "ask_user_tool": true,
    "profile": "standard"
  },
  "tui": {
    "theme": "catppuccin-mocha"
  },
  "search": {
    "provider": "tavily"
  }
}
```

Important runtime notes:

* `agent.request_timeout_s` is a stream idle timeout, not a total-turn deadline
* `agent.max_tokens` defaults to `null`, so no explicit `max_tokens` cap is sent unless configured
* memory persistence is fixed at `~/.alphanus/memory/` (`events.jsonl` source-of-truth, generated `facts.md`)
* session persistence is fixed at `~/.alphanus/sessions/`
* workspace content remains rooted at `workspace.path` (default `~/Desktop/Alphanus-Workspace`)
* Internal context budget defaults exist at runtime, but several budget controls are intentionally hidden from the editable config file and TUI editor
* `search.provider` currently supports `tavily` and `brave`
* `tui.theme` currently supports: `classic`, `soft`, `catppuccin-mocha`, `catppuccin-macchiato`, `tokyonight-moon`, `gruvbox-dark-soft` (`catppuccin` alias maps to `catppuccin-mocha`)
* `runtime.profile` supports `standard` and `minimal` (`safe` and `minimal_reliable` normalize to `minimal`; `workspace` and `full` normalize to `standard`)
* `capabilities.permission_profile` supports `safe`, `workspace`, and `full`; aliases `minimal`/`readonly` -> `safe` and `standard` -> `workspace`
* Collaboration mode is session-scoped (`execute` or `plan`) and can be toggled with `/mode`
* Skills are discovered from the configured repo `skills/` root only
* `runtime.ask_user_tool` gates structured follow-up question flows via `request_user_input`

---

## Runtime Profiles

`runtime.profile` controls model-visible tool scope:

* `standard` (default): core tools plus optional skill tools for loaded and selected skills
* `minimal`: only core workspace tools and skill discovery tools (`skills_list`, `skill_view`), plus `request_user_input` when `runtime.ask_user_tool` is enabled

Use `minimal` when you want deterministic behavior with a reduced tool surface.

---

## Collaboration Modes

Session-scoped collaboration mode controls whether turns are execution-oriented or planning-oriented:

* `execute` (default): normal tool behavior
* `plan`: read-only + ask mode; mutating tools, shell execution, and `run_skill` are blocked with policy errors

Use `/mode` to inspect the current setting, or `/mode plan` and `/mode execute` to switch.

---

## Turn Trace and Observability

Each turn journal now includes:

* `timing`: `started_at`, `finished_at`, `elapsed_ms`, and pass count
* `turn_trace.passes`: per-pass model payload, selected skills, exposed tools, finish reason, and pass timings
* `turn_trace.tool_calls`: tool call arguments and timestamps
* `turn_trace.tool_results`: tool outputs, policy-block markers, and completion timestamps

Turn journals are persisted with session data and included in `/report` support bundles.

Doctor and support reports also include harness-level runtime metrics:

* `task_completion_rate`
* `human_interruption_rate`
* `tool_failure_rate`
* `avg_tool_loop_depth`
* `first_token_latency_ms_avg`

---

## Built-in Skills

Bundled skills include:

* `workspace-ops` — create, read, edit, move, and delete workspace paths; search code; render the workspace tree; and run approved verification commands
* `memory-rag` — store, recall, forget, inspect, and export lexical memories
* `search-ops` — web search and page fetch for current information
* `shell-ops` — confirmed workspace shell commands
* `utilities` — weather lookup, home file search, URL open, and YouTube helpers

See [SKILLS_GUIDE.md](./SKILLS_GUIDE.md) for details.

---

## Skills Model

Alphanus separates skill discovery, enabling, and session loading.

Important behavior:

* Skill discovery uses the configured repo `skills/` root only
* Skills are expected to live under `<repo>/skills/<skill-id>/SKILL.md`
* `/skill-on` and `/skill-off` control whether a skill is globally enabled
* Skills become session-loaded when the runtime calls `skill_view(name)`
* Session-loaded skills persist with the session and can be removed with `/skill-unload` or `/skill-unload-all`
* Native `tools.py` tools are available only when the owning skill is loaded in the active session
* Executable skills run through unified `run_skill` (entrypoint or bundled script), scoped to loaded skills for the current turn

---

## TUI Commands

### Conversation and navigation

* `/help`
* `/keyboard-shortcuts` (`/shortcuts`, `/keymap`, `/keys`)
* `/details`
* `/think`
* `/mode [plan|execute]`
* `/clear`
* `/sessions`
* `/rename <name>`
* `/save [name]`
* `/file [path]`
* `/image [path]` (alias of `/file`)
* `/detach [n|last|all]`
* `/code [n|last]`
* `/quit`, `/exit`, `/q`

### Branching

* `/branch [label]`
* `/unbranch`
* `/branches`
* `/switch <n>`
* `/tree`

### Skills and diagnostics

* `/skills`
* `/reload`
* `/doctor`
* `/skill-on <id>`
* `/skill-off <id>`
* `/skill-unload <id>`
* `/skill-unload-all`
* `/skill-reload`
* `/skill-info <id>`

### Memory, workspace, and support

* `/memory-stats`
* `/context`
* `/workspace-tree`
* `/theme`
* `/config`
* `/report [file]`

### Keyboard shortcuts

* `F1` shows the keyboard shortcut reference
* `Ctrl+K` opens the quick palette
* `Ctrl+P` or `/` opens the slash-command palette
* `Ctrl+F` opens the file picker
* `Ctrl+G` focuses the composer
* `F2` toggles tool details
* `F3` toggles thinking mode
* `Ctrl+U` clears the full draft
* `Ctrl+Shift+K` deletes from the cursor to the end of the line

Session notes:

* Startup opens a session picker before chat input
* Sessions are stored under `~/.alphanus/sessions/`
* The active session is autosaved after each turn
* Loaded skill IDs are persisted with the session

---

## Memory

Memory is lexical and persistent.

Notes:

* `memory.min_score_default` sets the base lexical threshold for direct memory searches
* `memory.recall_min_score_default` and `memory.replace_min_score_default` tune skill-level recall and replace-query behavior
* `memory.backup_revisions` controls how many previous memory snapshots are kept as `.bakN` files
* storage is fixed at `~/.alphanus/memory/` (`events.jsonl` is source-of-truth, `facts.md` is regenerated on flush)
* Retrieval is lexical-only
* For corrections, prefer `replace_query` or `replace_ids` when calling memory tools

---

## Search

Search is intended for time-sensitive or current-information tasks.

Notes:

* `search-ops` supports `tavily` and `brave`
* Time-sensitive answers are expected to use fetched source text; otherwise Alphanus declines to speculate
* Search credentials are environment variables only
* Default per-turn tool budgets are:

  * `web_search=2`
  * `fetch_url=2`
  * `recall_memory=2`

---

## Workspace and Shell Safety

Workspace and shell operations are intentionally constrained.

* `delete_path` supports both files and directories
* `run_checks` is limited to approved verification runners:

  * `pytest`
  * `ruff`
  * `mypy`
  * `pyright`
  * `eslint`
  * `tsc`
  * `vitest`
  * `jest`
  * `tox`
  * `nox`
  * `uv run <approved-runner>`
* Shell commands require confirmation by default
* Shell-control operators and dangerous patterns are blocked

---

## Testing

Run the test suite with:

```bash
uv run pytest
```

Recommended quality checks:

```bash
uv run ruff check src tests
uv run pyright
uv run vulture src tests
```

---

## Architecture Notes

Current architecture changes focused on reducing coupling:

* `Agent` no longer monkey-patches classifier/orchestrator lambdas during config reload; runtime behavior is wired through explicit runtime hooks
* `SkillRuntime` delegates inventory loading, process-env construction, and tool execution to dedicated services
* `interface.py` is slimmer and delegates input and palette/theme behavior to focused runtime modules

---

## Project Direction

Alphanus is aimed at users who want a coding assistant that is:

* local-first
* explicit in how it loads tools and skills
* inspectable at runtime
* usable from a terminal UI
* comfortable to extend and modify

It is not trying to hide the system behind a chat box. The intent is to make the assistant's runtime behavior visible and controllable.
