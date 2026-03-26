# Alphanus

Alphanus is a local-first coding assistant with a Textual TUI, an OpenAI-compatible chat-completions loop, modular `SKILL.md` skills, persistent vector memory, and a branchable conversation tree.

Status:
- public alpha / power-user tooling
- local-first
- secrets are environment variables (not editable config fields)

## Current State

Current behavior and architecture:
- streaming agent loop with separate reasoning/content rendering
- OpenAI-style `tool_calls` execution
- all enabled skills are active from the first turn
- bundled workspace, shell, memory, search, and utility skills
- workspace-safe file operations and constrained verification runners
- persistent vector memory with transformer embeddings and deterministic hash fallback
- branchable conversation tree with named multi-session save/load and inactive-branch compaction
- TUI support for config editing, live tool previews, support bundle export, and code-block popups

Bundled skills:
- `workspace-ops`: create/read/edit/search/delete workspace files and run checks
- `memory-rag`: store, recall, forget, inspect, and export persistent memories
- `search-ops`: web search plus page fetch for up-to-date answers
- `shell-ops`: confirmed workspace shell commands
- `utilities`: weather lookup, home file search, URL open, and YouTube open/play helpers

## Requirements

- Python `>=3.11`
- [uv](https://docs.astral.sh/uv/)

## Quick Start

1. Install dependencies:

```bash
uv sync --extra dev
```

2. Start your model server externally. Default endpoints:
- `http://127.0.0.1:8080/v1/chat/completions`
- `http://127.0.0.1:8080/v1/models`

3. Run Alphanus:

```bash
uv run main.py
```

Useful flags:
- `--debug` enables HTTP debug logging to `logs/http-debug.jsonl`
- `--dangerously-skip-permissions` disables interactive shell approval prompts

Optional environment variables:
- `TAVILY_API_KEY` for `search-ops` when `search.provider = "tavily"`
- `BRAVE_SEARCH_API_KEY` for `search-ops` when `search.provider = "brave"`
- `ALPHANUS_AUTH_HEADER` for authenticated model endpoints (`Header-Name: value`)
- `AUTH_HEADER` fallback if `ALPHANUS_AUTH_HEADER` is unset

Notes:
- `.env` at repo root is auto-loaded on startup (without overriding already-set environment variables).
- On startup, Alphanus performs a models-endpoint handshake and prints readiness status.

### Tavily Setup for Web Search

Use this when you want `search-ops` to run public web lookups via Tavily.

1. Get a Tavily API key from [tavily.com](https://tavily.com/).
2. Set the key in your shell or `.env`:

```bash
export TAVILY_API_KEY="tvly-..."
```

3. Ensure your global config uses Tavily:

```json
{
  "search": {
    "provider": "tavily"
  }
}
```

4. Start Alphanus:

```bash
uv run main.py
```

5. In chat, ask a time-sensitive question (example: `latest NVIDIA earnings summary`) and Alphanus will use `web_search`/`fetch_url` via `search-ops`.

## Runtime Notes

- `agent.max_tokens` defaults to `null`, so no explicit `max_tokens` cap is sent unless configured.
- The system prompt includes the current local date.
- Endpoint host policy is enforced: `agent.model_endpoint` and `agent.models_endpoint` must share host unless `allow_cross_host_endpoints = true`.
- The TUI `ctx:` indicator and `/context` command are based on inference-engine metadata from `agent.models_endpoint` when available, not on app-side config fallbacks.
- `context.context_limit`, `context.safety_margin`, and `agent.context_budget_max_tokens` are not treated as authoritative model capacity in the UI.
- Shell commands require confirmation by default.
- Current-info answers are expected to come from fetched web evidence; otherwise Alphanus will decline to speculate.
- `delete_path` supports both files and directories.
- `run_checks` is intentionally constrained to approved verification runners (`pytest`, `ruff`, `mypy`, `pyright`, `eslint`, `tsc`, `vitest`, `jest`, `tox`, `nox`) and `uv run <approved-runner>`.

## TUI Commands

Conversation and view:
- `/help`
- `/keyboard-shortcuts` (`/shortcuts`, `/keymap`, `/keys`)
- `/details`
- `/think`
- `/clear`
- `/sessions`
- `/new [name]`
- `/rename <name>`
- `/save [name]`
- `/load`
- `/code [n|last]`
- `/quit`, `/exit`, `/q`

Attachments:
- `/file <path>` (image or text)
- `/image <path>` (alias of `/file`)

Branching:
- `/branch [label]`
- `/unbranch`
- `/branches`
- `/switch <n>`
- `/tree`

Skills and diagnostics:
- `/skills`
- `/reload`
- `/skill on <id>`
- `/skill off <id>`
- `/skill reload`
- `/skill info <id>`
- `/doctor`

Memory, workspace, and support:
- `/memory stats`
- `/context` (inference engine context usage, when reported)
- `/workspace tree`
- `/config`
- `/report [file]`
- `/export`
- `/import`

Keyboard shortcuts:
- `F1` or `?` shows the keyboard shortcut reference
- `Ctrl+P` opens the slash-command palette
- `Ctrl+G` focuses the composer
- `F2` toggles live tool details
- `F3` toggles thinking mode
- `Ctrl+U` clears the full draft
- `Ctrl+K` deletes from cursor to end of line

Session notes:
- Startup opens a session picker before chat input so you can load an existing session or start a new one immediately.
- Sessions are stored under `.alphanus/sessions/` in the current workspace.
- Managed exports are stored under `.alphanus/exports/` in the current workspace.
- Each saved session keeps its own `ConvTree`, active node, and branch structure.
- The active session is autosaved to disk on each turn.
- `/save` persists the active session and can optionally rename it.
- `/load` opens a picker of saved sessions.
- `/export` writes the current session tree into `.alphanus/exports`.
- `/import` opens a picker of stored exports and imports the selected export as a new session.

## Config

Global config lives at `config/global_config.json`. Missing keys are merged from built-in defaults at startup.
Config values are normalized and type-checked at load/save time, with invalid values falling back to safe defaults.
Secret-like keys are stripped from config (credentials must stay in environment variables).
Internal pruning defaults are not surfaced in the user-facing config file or config editor.
`agent.request_timeout_s` is a stream idle timeout, not a total turn deadline. Active streaming can run longer than this value as long as chunks keep arriving.

Built-in defaults:

```json
{
  "schema_version": "1.0.0",
  "agent": {
    "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
    "models_endpoint": "http://127.0.0.1:8080/v1/models",
    "request_timeout_s": 180,
    "readiness_timeout_s": 30,
    "readiness_poll_s": 0.5,
    "enable_thinking": true,
    "tls_verify": true,
    "ca_bundle_path": "",
    "allow_cross_host_endpoints": false,
    "max_tokens": null,
    "max_action_depth": 10,
    "max_tool_result_chars": 12000,
    "max_reasoning_chars": 20000,
    "compact_tool_results_in_history": false,
    "compact_tool_result_tools": []
  },
  "workspace": {
    "path": "~/Desktop/Alphanus-Workspace"
  },
  "memory": {
    "path": "./memories/memory.pkl",
    "embedding_backend": "transformer",
    "model_name": "BAAI/bge-small-en-v1.5",
    "eager_load_encoder": false,
    "allow_model_download": true
  },
  "context": {
    "keep_last_n": 10
  },
  "capabilities": {
    "shell_require_confirmation": true,
    "dangerously_skip_permissions": false
  },
  "skills": {
    "strict_capability_policy": false
  },
  "tools": {
    "core_exposure_policy": "coding_core"
  },
  "search": {
    "provider": "tavily"
  },
  "tui": {
    "chat_log_max_lines": 5000,
    "tree_compaction": {
      "enabled": true,
      "inactive_assistant_char_limit": 12000,
      "inactive_tool_argument_char_limit": 5000,
      "inactive_tool_content_char_limit": 8000
    }
  }
}
```

The TUI also exposes `/config`, which opens a modal editor for global config. Secret values are intentionally omitted there.

## Memory and RAM Tuning

Notes:
- `memory.embedding_backend: "transformer"` is the default and recommended semantic mode.
- `memory.embedding_backend: "hash"` is the lowest-RAM fallback.
- Recommended transformer model: `BAAI/bge-small-en-v1.5`.
- `memory.allow_model_download` controls whether uncached embedding weights may be downloaded on first use.
- Existing memories are automatically re-embedded when you switch backend/model, so recall stays consistent.
- `tui.chat_log_max_lines` bounds RichLog memory growth.
- Tree compaction is lossy for inactive branches; disable it if you need full historical payload fidelity when switching back.

## Search and Current-Info Behavior

- `search-ops` supports `tavily` and `brave` providers.
- Time-sensitive answers require fetched-source evidence; otherwise Alphanus declines to speculate.
- Search credentials are environment variables only.
- Search loops are intentionally short (default per-turn tool budgets: `web_search=2`, `fetch_url=2`).

## Skills

See [SKILLS_GUIDE.md](./SKILLS_GUIDE.md).

Skills live under `skills/<skill-id>/SKILL.md` and can expose tools via:
- `metadata.tools.definitions` command entries
- `tools.py` (`TOOL_SPECS` + `execute`) native tool modules
- optional `hooks.py` (`pre_prompt`, `pre_action`, `post_response`)

Skill loading also supports requirement gating (`os`, `env`, `commands`) and marks blocked skills in `/skills` and `/doctor`.

## Tests

```bash
uv run pytest
```

## Live Smoke Checks

Run a real end-to-end smoke pass against your currently running model server:

```bash
uv run python scripts/live_smoke.py
```

Optional scenarios:

```bash
uv run python scripts/live_smoke.py --include-browser
uv run python scripts/live_smoke.py --include-browser --json
```

Notes:
- `--include-browser` exercises `open_url` and `play_youtube`.
- The script uses a temporary workspace and temporary memory file, then runs real `Agent.run_turn(...)` scenarios against the live endpoint.
- Browser scenarios are optional because they trigger desktop-side effects.
