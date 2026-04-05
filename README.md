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

It combines a terminal UI, a streaming chat-completions loop, persistent semantic memory, explicit skills, and workspace-aware tooling into a system that is meant to be inspectable and hackable rather than hidden behind a hosted product.

The current implementation is built around `llama.cpp` and a Textual interface, with support for named sessions, branching conversation history, configurable skills, and controlled workspace or shell operations.

---

## Features

- Streaming agent loop with separate reasoning and content handling
- `llama.cpp` `/v1` API integration with OpenAI-style `tool_calls`
- Textual UI with live tool previews, code block popups, config editing, and support-bundle export
- Named autosaved sessions with a branchable conversation tree
- Persistent semantic memory with automatic re-embedding on model change
- Explicit session-loaded skills with enable and disable controls
- Web search support for time-sensitive answers
- Workspace-aware tooling with confirmation gates and executable skill trust boundaries

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
````

Set any required environment variables:

* `TAVILY_API_KEY` when `search.provider = "tavily"`
* `BRAVE_SEARCH_API_KEY` when `search.provider = "brave"`
* `ALPHANUS_AUTH_HEADER` for authenticated model endpoints (`Header-Name: value`)
* `AUTH_HEADER` as a fallback if `ALPHANUS_AUTH_HEADER` is unset

Run Alphanus:

```bash
uv run alphanus
```

Useful flags:

```bash
uv run alphanus --debug
uv run alphanus --dangerously-skip-permissions
```

Notes:

* `.env` at the repo root is loaded automatically on startup without overriding already-set environment variables
* Startup performs a `/v1/models` readiness handshake before the first turn
* `uv run main.py` still works in a repo checkout, but `uv run alphanus` is the primary entrypoint

---

## Configuration

Global config is stored at:

* `config/global_config.json` in a repo checkout
* `~/.alphanus/config/global_config.json` for an installed package

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
  "memory": {
    "path": "./memories/memory.pkl",
    "model_name": "BAAI/bge-small-en-v1.5",
    "eager_load_encoder": false,
    "allow_model_download": true
  },
  "context": {
    "keep_last_n": 10
  },
  "search": {
    "provider": "tavily"
  }
}
```

Important runtime notes:

* `agent.request_timeout_s` is a stream idle timeout, not a total-turn deadline
* `agent.max_tokens` defaults to `null`, so no explicit `max_tokens` cap is sent unless configured
* Internal context budget defaults exist at runtime, but several budget controls are intentionally hidden from the editable config file and TUI editor
* `search.provider` currently supports `tavily` and `brave`
* `skills.load.extra_dirs`, `skills.load.watch`, and `skills.load.upward_scan` control skill discovery
* `agents.enable_skill_agents` and `runtime.ask_user_tool` gate companion-agent and structured follow-up flows

---

## Built-in Skills

Bundled skills include:

* `workspace-ops` — create, read, edit, move, and delete workspace paths; search code; render the workspace tree; and run approved verification commands
* `memory-rag` — store, recall, forget, inspect, and export semantic memories
* `search-ops` — web search and page fetch for current information
* `shell-ops` — confirmed workspace shell commands
* `utilities` — weather lookup, home file search, URL open, and YouTube helpers

See [SKILLS_GUIDE.md](./SKILLS_GUIDE.md) for details.

---

## Skills Model

Alphanus separates skill discovery, enabling, and session loading.

Important behavior:

* Skill discovery scans upward from the workspace for `skills/`, `.claude/skills`, `.agents/skills`, and `.opencode/skills`
* It also checks user-local skill roots under the home directory
* Shadowing priority is:

```text
workspace/local > bundled > user/local > external/local
```

* Only bundled and workspace-local roots are trusted executable roots
* Home and external roots are visible in `/skills` and `/doctor`, but executable surfaces are blocked
* `/skill-on` and `/skill-off` control whether a skill is globally enabled
* Skills become session-loaded when the runtime calls `skill_view(name)`
* Session-loaded skills persist with the session and can be removed with `/skill-unload` or `/skill-unload-all`
* Core tools from enabled trusted skills remain model-exposed; optional skill runtime surfaces are added when a skill is loaded for the session

---

## TUI Commands

### Conversation and navigation

* `/help`
* `/keyboard-shortcuts` (`/shortcuts`, `/keymap`, `/keys`)
* `/details`
* `/think`
* `/clear`
* `/sessions`
* `/new [name]`
* `/rename <name>`
* `/save [name]`
* `/load`
* `/file [path]`
* `/image [path]`
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
* `/config`
* `/report [file]`

### Keyboard shortcuts

* `F1` shows the keyboard shortcut reference
* `Ctrl+P` opens the slash-command palette
* `Ctrl+G` focuses the composer
* `F2` toggles tool details
* `F3` toggles thinking mode
* `Ctrl+U` clears the full draft
* `Ctrl+K` deletes from the cursor to the end of the line

Session notes:

* Startup opens a session picker before chat input
* Sessions are stored under `.alphanus/sessions/` in the current workspace
* The active session is autosaved after each turn
* Loaded skill IDs are persisted with the session

---

## Memory

Memory is semantic and persistent.

Notes:

* Recommended embedding model: `BAAI/bge-small-en-v1.5`
* `memory.allow_model_download` controls whether uncached embedding weights may be downloaded on first use
* Existing memories are automatically re-embedded when you switch models
* Retrieval is semantic-only
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

---

## Live Smoke Checks

Run an end-to-end smoke pass against your current model server:

```bash
uv run python scripts/live_smoke.py
```

Optional scenarios:

```bash
uv run python scripts/live_smoke.py --include-browser
uv run python scripts/live_smoke.py --include-browser --json
```

Notes:

* `--include-browser` exercises `open_url` and `play_youtube`
* The script uses a temporary workspace and temporary memory file
* Browser scenarios are optional because they trigger desktop-side effects

---

## Project Direction

Alphanus is aimed at users who want a coding assistant that is:

* local-first
* explicit in how it loads tools and skills
* inspectable at runtime
* usable from a terminal UI
* comfortable to extend and modify

It is not trying to hide the system behind a chat box. The intent is to make the assistant's runtime behavior visible and controllable.