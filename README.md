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
  <img src="https://img.shields.io/badge/platform-local--first-lightgrey" alt="Local first">
</p>

<p align="center">
  OpenAI-compatible endpoints, explicit tools, inspectable runtime behavior.
</p>

---

## Table of Contents

- [What Alphanus Is](#what-alphanus-is)
- [Quick Start](#quick-start)
- [Model Endpoint Setup](#model-endpoint-setup)
- [Major Features](#major-features)
  - [Sessions and Branching](#sessions-and-branching)
  - [Collaboration Modes](#collaboration-modes)
  - [Skills and Tool Loading](#skills-and-tool-loading)
  - [Workspace Operations](#workspace-operations)
  - [Memory](#memory)
  - [Search](#search)
  - [Safety and Permission Profiles](#safety-and-permission-profiles)
  - [Turn Trace and Diagnostics](#turn-trace-and-diagnostics)
- [Command Cheat Sheet](#command-cheat-sheet)
- [Configuration](#configuration)
- [State Layout](#state-layout)
- [Development](#development)
- [Status and Scope](#status-and-scope)

---

## What Alphanus Is

Alphanus is an experimental local-first coding assistant.

It combines:

- a Textual terminal UI
- a streaming model loop
- explicit tool execution
- session-loaded skills
- lexical memory
- branching conversation history

The system is designed to be inspectable and controllable at runtime rather than hidden behind a hosted UI.

---

## Quick Start

### Requirements

- Python `>=3.11`
- [`uv`](https://docs.astral.sh/uv/)
- An OpenAI-compatible endpoint exposing:
  - `GET /v1/models`
  - `POST /v1/chat/completions` and/or `POST /v1/responses`

### Install

```bash
uv sync --extra dev
```

### Initialize, verify, run

```bash
uv run alphanus init
uv run alphanus doctor
uv run alphanus
```

### Required env vars (as needed)

- `ALPHANUS_API_KEY` for authenticated model endpoints
- `ALPHANUS_AUTH_HEADER` for advanced custom auth header override
- `AUTH_HEADER` fallback if `ALPHANUS_AUTH_HEADER` is unset
- `TAVILY_API_KEY` when `search.provider = "tavily"`
- `BRAVE_SEARCH_API_KEY` when `search.provider = "brave"`

---

## Model Endpoint Setup

Alphanus supports three endpoint modes:

- `chat` (default)
- `responses`
- `auto` (responses-first with fallback to chat when unsupported)

Interactive setup:

```bash
uv run alphanus init
```

Non-interactive setup example:

```bash
uv run alphanus init --non-interactive \
  --workspace-path ~/Desktop/Alphanus-Workspace \
  --base-url https://api.openai.com \
  --endpoint-mode auto \
  --api-key "$ALPHANUS_API_KEY" \
  --model-endpoint https://api.openai.com/v1/chat/completions \
  --responses-endpoint https://api.openai.com/v1/responses \
  --models-endpoint https://api.openai.com/v1/models \
  --search-provider tavily \
  --theme catppuccin-mocha
```

Notes:

- `init` writes API key secrets to `~/.alphanus/.env`
- config stores API key references (for example `env:ALPHANUS_API_KEY`) instead of plaintext keys
- local backends that do not require auth can run with no API key set

---

## Major Features

### Sessions and Branching

- Named sessions are autosaved after each turn
- A conversation can branch at any point and switch between branches
- Tree view makes alternate solution paths explicit
- Loaded skill IDs are persisted with each session

Branching commands:

- `/branch [label]`
- `/unbranch`
- `/branches`
- `/switch <n>`
- `/tree`

### Collaboration Modes

Session mode controls how turns behave:

- `execute`: normal agent execution with tools
- `plan`: read-only/ask-first mode; mutating tools and shell execution are blocked

Commands:

- `/mode`
- `/mode plan`
- `/mode execute`

### Skills and Tool Loading

- Skills are discovered under `<repo>/skills/<skill-id>/SKILL.md`
- Skills can be enabled/disabled globally
- Tools from a skill are only available when that skill is loaded in the active session
- Skill execution is explicit through `run_skill` and tool schemas

Useful commands:

- `/skills`
- `/skill-on <id>`
- `/skill-off <id>`
- `/skill-unload <id>`
- `/skill-unload-all`
- `/skill-reload`
- `/skill-info <id>`

### Workspace Operations

`workspace-ops` covers create/read/edit/move/delete with guardrails.

Current capabilities include:

- section-scoped read/edit by line bounds and anchors
- regex-based edits
- ripgrep-backed code search with optional context lines
- workspace tree rendering
- verification command runner (`run_checks`) with approved commands

### Memory

Memory is lexical and persistent.

- storage path: `~/.alphanus/memory/`
- source of truth: `events.jsonl`
- generated index file: `facts.md`
- configurable score thresholds and backup revisions

Relevant commands:

- `/memory-stats`

### Search

Search is intended for time-sensitive queries.

- providers: `tavily`, `brave`
- policy: if evidence is insufficient, Alphanus declines to speculate
- default per-turn budgets:
  - `web_search=2`
  - `fetch_url=2`
  - `recall_memory=2`

### Safety and Permission Profiles

Runtime safety knobs:

- `runtime.profile`: `standard` or `minimal`
- `capabilities.permission_profile`: `safe`, `workspace`, `full`
- shell commands require confirmation by default
- dangerous shell patterns are blocked

### Turn Trace and Diagnostics

Per-turn journal includes:

- `timing`: start/end/elapsed/pass count
- `turn_trace.passes`: payloads, selected skills, exposed tools, finish reason
- `turn_trace.tool_calls`: arguments + timestamps
- `turn_trace.tool_results`: outputs + policy-block markers + timings

`/doctor` and support bundles include harness metrics such as:

- task completion rate
- tool failure rate
- avg tool loop depth
- first token latency average

---

## Command Cheat Sheet

### Conversation

- `/help`
- `/details`
- `/think`
- `/clear`
- `/quit`, `/exit`, `/q`

### Sessions

- `/sessions`
- `/rename <name>`
- `/save [name]`

### Files and Attachments

- `/file [path]`
- `/image [path]`
- `/detach [n|last|all]`
- `/code [n|last]`

### Skills and Runtime

- `/skills`
- `/reload`
- `/doctor`
- `/theme`
- `/config`
- `/report [file]`

### Keyboard Shortcuts

- `F1`: shortcuts help
- `Ctrl+K`: quick palette
- `Ctrl+P` or `/`: slash-command palette
- `Ctrl+F`: file picker
- `Ctrl+G`: focus composer
- `F2`: toggle tool details
- `F3`: toggle thinking mode
- `Ctrl+U`: clear draft
- `Ctrl+Shift+K`: delete to end of line

---

## Configuration

Global config path:

- `~/.alphanus/config/global_config.json`

Config behavior:

- missing keys merged from defaults
- types normalized and invalid values clamped
- secret-like fields stripped from disk and `/config` editor
- endpoint host policy enforced unless `agent.allow_cross_host_endpoints = true`

Model-related config keys:

- `agent.base_url`
- `agent.model_endpoint`
- `agent.responses_endpoint`
- `agent.models_endpoint`
- `agent.endpoint_mode`
- `agent.api_key` (env reference recommended)
- `agent.api_key_env`
- `agent.auth_header_template`

Trimmed config example:

```json
{
  "schema_version": "1.0.0",
  "agent": {
    "base_url": "http://127.0.0.1:8080",
    "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
    "responses_endpoint": "http://127.0.0.1:8080/v1/responses",
    "models_endpoint": "http://127.0.0.1:8080/v1/models",
    "endpoint_mode": "chat",
    "api_key": "env:ALPHANUS_API_KEY",
    "api_key_env": "ALPHANUS_API_KEY",
    "request_timeout_s": 180,
    "readiness_timeout_s": 30,
    "allow_cross_host_endpoints": false,
    "max_tokens": null
  },
  "workspace": {
    "path": "~/Desktop/Alphanus-Workspace"
  },
  "runtime": {
    "profile": "standard",
    "ask_user_tool": true
  },
  "capabilities": {
    "permission_profile": "full"
  }
}
```

---

## State Layout

Alphanus keeps runtime state separate from your workspace files.

- `~/.alphanus/config/global_config.json` (config)
- `~/.alphanus/.env` (secrets)
- `~/.alphanus/sessions/` (sessions)
- `~/.alphanus/memory/` (memory)

Workspace content remains under `workspace.path`.

---

## Development

Run tests:

```bash
uv run pytest
```

Optional checks:

```bash
uv run ruff check src tests
uv run pyright
uv run vulture src tests
uv run python tools/benchmark_hotpaths.py --loops 5
```

Autoresearch-style optimization loop (whole program, functionality-gated):

```bash
# initialize baseline
uv run python tools/autoresearch_optimize.py init --description "baseline"

# run after each refactor attempt
uv run python tools/autoresearch_optimize.py evaluate --description "simplify X"
```

The loop keeps changes only when tests pass and tracked LOC decreases. See `optimization/program.md` for the workflow.

---

## Status and Scope

Alphanus is currently alpha.

Current focus:

- local-first workflows
- runtime transparency
- safer tool execution boundaries
- strong terminal UX for coding tasks

Expect active iteration and occasional config/schema evolution.
