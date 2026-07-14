<p align="center">
  <h1 align="center">Alphanus</h1>
  <p align="center">
    Alpha coding assistant with a Rust/Ratatui TUI, a JSONL automation interface, transactional state, and workspace-guarded tools.
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
  - [Terminal UI](#terminal-ui)
  - [Collaboration Modes](#collaboration-modes)
  - [Skills and Tool Loading](#skills-and-tool-loading)
  - [Built-In Skills](#built-in-skills)
  - [Project Operations](#project-operations)
  - [Memory](#memory)
  - [Search](#search)
  - [Safety and Permission Modes](#safety-and-permission-modes)
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

- a Rust/Ratatui terminal UI
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
- macOS or Linux (Windows is intentionally unsupported in 0.2.0)
- [`uv`](https://docs.astral.sh/uv/)
- Rust `1.85+` and Cargo when installing from source (release-wheel users do not need Rust)
- An OpenAI-compatible endpoint exposing:
  - `GET /v1/models`
  - optional `GET /slots` or `GET /props` for loaded-model/context metadata on local backends
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

Desktop permission checks can be run separately:

```bash
uv run alphanus init permissions
```

### Required env vars (as needed)

- `ALPHANUS_API_KEY` for authenticated model endpoints
- `ALPHANUS_AUTH_HEADER` for advanced custom auth header override
- `TAVILY_API_KEY` when `search.provider = "tavily"` or `search.fallback_provider = "tavily"`
- `ALPHANUS_EMBEDDINGS_API_KEY` when optional OpenAI-compatible retrieval embeddings use an authenticated endpoint

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
  --project-root ~/Desktop/Alphanus-Project \
  --base-url https://api.openai.com \
  --endpoint-mode auto \
  --model-endpoint https://api.openai.com/v1/chat/completions \
  --responses-endpoint https://api.openai.com/v1/responses \
  --models-endpoint https://api.openai.com/v1/models \
  --search-provider searxng \
  --search-fallback-provider tavily \
  --searxng-base-url http://127.0.0.1:8888 \
  --theme catppuccin-mocha
```

Notes:

- `init` writes an owner-only, versioned `~/.alphanus/config/config.toml`
- credentials are read from environment variables only; CLI flags, TOML secrets, and `.env` loading are rejected
- config stores API key references (for example `env:ALPHANUS_API_KEY`) instead of plaintext keys
- local backends that do not require auth can run with no API key set
- model status prefers loaded-model metadata from local `/slots` or `/props` endpoints when available; `/v1/models` is used as the general fallback
- the TUI does not keep the inference server warm with continuous idle pings; status is refreshed on startup, explicit checks, config/model changes, and transport failures
- screenshot capture depends on OS permissions outside Alphanus; `alphanus init permissions` reports the platform-specific setup and opens the macOS Screen Recording settings pane during interactive setup

---

## Major Features

### Headless automation

`alphanus exec` is a public, versioned JSONL interface. Protocol records are written only to stdout and diagnostics only to stderr.

```bash
printf '%s\n' '{"schema_version":1,"prompt":"summarize this repository"}' \
  | uv run alphanus exec --input jsonl --approval-policy deny
```

Exit codes distinguish model failure, policy denial, invalid input/configuration, cancellation, and internal failure. Boundary approvals are denied by default in non-interactive mode.

### Sessions and Branching

- Named sessions are autosaved after each turn
- A conversation can branch at any point and switch between branches
- The conversation tree can be opened on demand for branch navigation and turn inspection
- Loaded skill IDs are persisted with each session

Branching commands:

- `/branch [label]`
- `/unbranch`
- `/branches`
- `/switch <n>`
- `/tree`

### Terminal UI

The TUI keeps the primary view focused on the transcript and composer.

The Python agent runtime and Rust frontend communicate over a private, versioned JSONL protocol. The public `alphanus exec` JSONL protocol is unchanged.

- Bottom metadata shows model, thinking state, session, branch, endpoint, LLM state, and context usage
- Keyboard hints sit beside the input instead of a persistent top header
- The conversation tree and inspector live in a hidden split that opens only when requested
- Saved sessions can be searched by title, messages, branch labels, and tool names from `/sessions`
- Tool execution details, reasoning text, streamed file previews, and themes remain available in the transcript flow
- Keyboard and mouse navigation cover panels, trees, palettes, sessions, themes, approvals, file selection, modal actions, scrolling, and focus changes
- Transcript, code, and config views support explicit OSC 52 copy actions; native terminal selection remains available with Shift-drag

Useful controls:

- `Tab` / `Shift+Tab`: cycle active panels
- `Ctrl+B`: toggle the conversation split
- `Ctrl+L`: open and focus the tree split
- `Ctrl+H`: focus the transcript
- `Ctrl+G`: focus the composer
- `PgUp` / `PgDn`: scroll transcript
- `j` / `k`, `Enter` / `o`, `[` / `]`, `g` / `G`: navigate the tree split

### Collaboration Modes

Session mode controls how turns behave:

- `execute`: normal agent execution with tools
- `plan`: read-only/ask-first mode; mutating tools and shell execution are blocked

Commands:

- `/mode`
- `/mode plan`
- `/mode execute`

### Skills and Tool Loading

- Python skill runtime code lives in `src/skills/`
- Built-in shipped skills live in `bundled-skills/`
- Downloaded AgentSkills can be dropped into `<repo>/skills/<skill-id>/SKILL.md` in a checkout, or `~/.alphanus/skills/<skill-id>/SKILL.md` when installed
- Extra skill roots can be configured with `skills.paths`
- User skills are loaded before bundled skills, so a downloaded skill can override a bundled skill with the same id
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

### Built-In Skills

Bundled skills cover project work, shell execution, search, memory, and a small set of desktop/local utilities.

Core project/runtime skills:

- `project-ops`: read, create, edit, move, delete, find, tree, and code-search files inside the configured project
- `shell-ops`: run confirmed shell commands from the project with visible stdout/stderr previews
- `search-ops`: web search and page fetch through configured providers
- `memory-rag`: recall and save stable local facts
- `utilities`: weather, URL/Youtube helpers, and simple local utility lookups
- `git`: inspect and operate on Git repositories with path policy checks

Desktop and local-inspection skills:

- `app-control`: list, open, focus, and quit desktop applications; open/focus/quit require explicit confirmation
- `browser-control`: open URLs/searches with confirmation and inspect the current browser page where supported
- `local-search`: search filenames and text under the project root, skipping sensitive/protected paths
- `document-tools`: extract text/tables from TXT, CSV, PDF, and DOCX; PDF/DOCX require optional dependencies
- `screenshot-ocr`: capture screenshots with confirmation and OCR explicit image paths when OCR tooling is installed

Skill discovery is intentionally explicit. `/skills` shows the installed catalog, `/skill-info <id>` shows detailed metadata, and `skill_view` loads a skill's full instructions/tools into a session.

### Themes

- Built-in themes are JSON files under `src/tui/theme_specs/`
- Custom themes can be added as `~/.alphanus/themes/<theme-id>.json`
- Extra theme directories can be supplied with `ALPHANUS_THEME_PATHS` using the platform path separator
- Existing theme JSON is interpreted without modification: `theme` and `colors` provide semantic Ratatui colors, while legacy syntax-theme fields remain accepted
- An optional `ratatui` object can set `border_set` (`plain`, `rounded`, `double`, or `thick`), `syntax_theme`, and semantic `styles` with foreground/background colors and modifiers
- Invalid Ratatui overrides produce a warning and fall back to the theme's semantic colors; unknown fields are ignored for forward compatibility

### Project Operations

`project-ops` covers create/read/edit/move/delete with guardrails.

Current capabilities include:

- section-scoped read/edit by line bounds and anchors
- regex-based edits
- filename/path discovery with `find_files`
- ripgrep-backed code search with optional context lines
- bounded project tree rendering from the root or a selected directory
- path-safe move/delete operations inside the project
- symlink-aware listing/tree rendering; directory symlinks are shown but not traversed by the tree
- no command runner; use `shell-ops` only when shell output itself is required

### Memory

Memory is lexical and persistent.

- storage path: `~/.alphanus/memory/`
- source of truth: `events.jsonl`
- generated index file: `facts.md`
- configurable score thresholds and backup revisions

Relevant commands:

- `/memory-stats`
- `uv run alphanus retrieval stats`
- `uv run alphanus retrieval reset --yes`

### Search

Search is intended for time-sensitive queries and feeds the local retrieval index when pages are fetched.

- primary provider: `searxng`
- fallback provider: optional `tavily` using `TAVILY_API_KEY`
- `search.searxng_base_url` is required for SearXNG; if it is missing or unreachable, Alphanus can use Tavily when `search.fallback_provider = "tavily"`
- export the configured Tavily environment variable before launching Alphanus; secrets are never persisted by `init`
- retrieval store: SQLite FTS under the Alphanus state root, usually `~/.alphanus/retrieval/index.sqlite` unless `ALPHANUS_APP_ROOT` is set
- fetched pages are indexed; search result snippets alone are not persisted
- optional dense retrieval uses an OpenAI-compatible embeddings endpoint when `retrieval.embeddings.enabled` is true
- safe automatic memory capture stores only obvious stable preference/project facts and skips secret-like text
- policy: if evidence is insufficient, Alphanus declines to speculate
- default per-turn budgets:
  - `web_search=2`
  - `fetch_url=2`
  - `recall_memory=2`

Configure a local SearXNG instance:

```bash
uv run alphanus init search --non-interactive \
  --search-provider searxng \
  --search-fallback-provider tavily \
  --tavily-api-key "$TAVILY_API_KEY" \
  --searxng-base-url http://127.0.0.1:8888

uv run alphanus doctor
```

Use Tavily directly without running SearXNG:

```bash
export TAVILY_API_KEY="tvly-..."

uv run alphanus init search --non-interactive \
  --search-provider tavily \
  --tavily-api-key "$TAVILY_API_KEY"
```

### Safety and Permission Modes

Runtime safety knobs:

- `permissions.mode`: `read-only`, `project-write`, or `danger-full-access`
- `permissions.approvals`: `on-boundary`
- `permissions.network`: `false` by default
- `sandbox.backend`: `auto`, with fail-closed setup checks by default
- project root detection uses the enclosing git repository, falling back to launch `cwd`; pass `--project-root` for an explicit per-run override
- project file tools treat relative paths as project-relative, and can operate on explicit absolute paths outside the project when those paths are not protected
- shell commands run through the platform sandbox in `project-write` and require approval at the configured boundary
- protected internal state such as `.alphanus` is blocked before execution, including common shell expansion paths
- direct `.git` writes and project root deletion are blocked by policy
- desktop actions such as app launch, browser open, and screenshot capture require explicit tool-level confirmation when they can affect the local machine
- macOS requires Screen Recording permission for the terminal app or launcher that runs Alphanus; Linux requires `gnome-screenshot` or `scrot`; Windows is not supported

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
- `/shortcuts`, `/keymap`, `/keys`
- `/details`
- `/think`
- `/mode [plan|execute]`
- `/clear`
- `/quit`, `/exit`, `/q`

### Sessions

- `/sessions`
- `/rename <name>`
- `/save [name]`

### Branching

- `/branch [label]`
- `/unbranch`
- `/branches`
- `/switch <n>`
- `/tree`

### Files and Attachments

- `/file [path]`
- `/detach [n|last|all]`
- `/code [n|last]`

### Skills and Runtime

- `/skills`
- `/reload`
- `/doctor`
- `/health`
- `/skill-on <id>`
- `/skill-off <id>`
- `/skill-unload <id>`
- `/skill-unload-all`
- `/skill-reload`
- `/skill-info <id>`
- `/memory-stats`
- `/context`
- `/project-tree`
- `/theme`
- `/config`
- `/report [file]`

### Keyboard Shortcuts

- `F1`: shortcuts help
- `Ctrl+K`: quick palette
- `Ctrl+P` or `/`: slash-command palette
- `Ctrl+F`: file picker
- `Ctrl+B`: toggle conversation split
- `Ctrl+G`: focus composer
- `Ctrl+H`: focus transcript
- `Ctrl+L`: open/focus tree split
- `Tab` / `Shift+Tab`: cycle panels
- `PgUp` / `PgDn`: scroll transcript
- `F2`: toggle tool details
- `F3`: toggle thinking mode
- `Esc`: clear input or stop stream
- `Backspace` on empty input: remove last attachment
- `Ctrl+Backspace`: remove last attachment
- `Ctrl+Shift+Backspace`: clear attachments
- `Ctrl+U`: clear draft
- `Ctrl+Shift+K`: delete to end of line

---

## Configuration

Global config path:

- `~/.alphanus/config/config.toml`

Config behavior:

- missing keys merged from defaults
- types normalized and invalid values clamped
- secret-like fields stripped from disk and `/config` editor
- endpoint host policy enforced unless `agent.allow_cross_host_endpoints = true`
- normalized config is projected into an internal typed v2 runtime model for new subsystem code
- config and session storage use major schema v2; v1 files are intentionally rejected instead of migrated

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
  "project": {
    "root_strategy": "git-or-cwd"
  },
  "permissions": {
    "mode": "project-write",
    "approvals": "on-boundary",
    "network": false
  },
  "sandbox": {
    "backend": "auto",
    "fail_closed": true
  },
  "runtime": {
    "ask_user_tool": true
  }
}
```

---

## State Layout

Alphanus keeps runtime state separate from your project files.

- `~/.alphanus/config/config.toml` (versioned, owner-only config)
- environment variables or OS credential injection (secrets; never stored by Alphanus)
- `~/.alphanus/sessions/sessions.db` and `~/.alphanus/memory/memory.db` (WAL-enabled SQLite state)
- `~/.alphanus/sessions/` (sessions)
- `~/.alphanus/memory/` (memory)

Project content remains under `project.path`.

---

## Development

Run the required local quality gate:

```bash
uv run pytest
uv run ruff check src tests bundled-skills tools
uv run pyright
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --no-default-features
```

Performance benchmarks are part of the test suite and can be run directly:

```bash
uv run pytest tests/test_performance_benchmarks.py
```

---

## Status and Scope

Alphanus is currently alpha.

Current focus:

- local-first workflows
- runtime transparency
- safer tool execution boundaries
- strong terminal UX for coding tasks

Expect active iteration and occasional config/schema evolution.
