# Alphanus

Alphanus is a local-first coding assistant with a Textual TUI, an OpenAI-compatible chat-completions client, a modular `SKILL.md` skill runtime, persistent vector memory, and a branchable conversation tree.

Status:
- public alpha / power-user tooling
- local-first
- secrets come from environment variables, not editable config fields

## Current State

Current behavior and architecture:
- streaming agent loop with separate reasoning/content rendering
- OpenAI-style `tool_calls` execution loop
- model-routed skill selection by default (`skills.selection_mode = "model"`)
- built-in workspace, shell, memory, search, and utility skills
- workspace-safe read/write/delete tools plus constrained verification runners
- persistent vector memory with transformer embeddings and deterministic hash fallback
- branchable conversation tree with named multi-session save/load plus inactive-branch compaction
- TUI support for config editing, live tool previews, support bundle export, and code-block popups

Bundled skills:
- `workspace-ops`: create/read/edit/search/delete workspace files and run checks
- `memory-rag`: store, recall, forget, inspect, and export persistent memories
- `search-ops`: web search plus page fetch for current or verified answers
- `shell-ops`: confirmed workspace shell commands
- `utilities`: weather lookup, home file search, URL open, and YouTube open/play helpers

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
- `TAVILY_API_KEY` for `search-ops`
- `BRAVE_SEARCH_API_KEY` for `search-ops` when `search.provider = "brave"`
- `ALPHANUS_AUTH_HEADER` for authenticated model endpoints (`Header-Name: value`)

At startup, Alphanus performs a models-endpoint handshake and prints readiness status, for example:
- `waiting for endpoint <...>/v1/models handshake...`

## Runtime Notes

- `agent.max_tokens` defaults to `null`, so no explicit `max_tokens` cap is sent unless configured.
- The system prompt includes the current local date.
- The app validates that `agent.model_endpoint` and `agent.models_endpoint` share a host unless `allow_cross_host_endpoints` is enabled.
- Shell commands require confirmation by default.
- Current-info answers are expected to come from fetched web evidence; otherwise the agent will decline to speculate.
- Workspace deletes are handled through `delete_path`, which supports both files and directories.

## TUI Commands

Conversation and view:
- `/help`
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
- `/file <path>`
- `/image <path>`

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
- `/workspace tree`
- `/config`
- `/report [file]`
- `/export`
- `/import`

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

Global config lives at `config/global_config.json`. Missing keys are merged from the built-in defaults at startup.

Key defaults:

```json
{
  "agent": {
    "model_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
    "models_endpoint": "http://127.0.0.1:8080/v1/models",
    "enable_thinking": true,
    "max_tokens": null,
    "context_budget_max_tokens": 2048,
    "max_action_depth": 10
  },
  "memory": {
    "embedding_backend": "transformer",
    "model_name": "BAAI/bge-small-en-v1.5",
    "allow_model_download": true
  },
  "skills": {
    "selection_mode": "model",
    "max_active_skills": 2
  },
  "tools": {
    "core_exposure_policy": "coding_core"
  },
  "search": {
    "provider": "tavily"
  }
}
```

The TUI also exposes `/config`, which opens a modal editor for the global config. Secret values are intentionally omitted there.

## Memory and RAM Tuning

Edit `config/global_config.json`:

```json
{
  "memory": {
    "embedding_backend": "transformer",
    "model_name": "BAAI/bge-small-en-v1.5",
    "allow_model_download": true
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

Notes:
- `memory.embedding_backend: "transformer"` is the default and recommended semantic mode.
- `memory.embedding_backend: "hash"` is the lowest-RAM fallback.
- Recommended transformer model: `BAAI/bge-small-en-v1.5`.
- `memory.allow_model_download` controls whether the app may fetch uncached embedding weights on first use.
- Existing memories are automatically re-embedded when you switch backend/model so recall stays consistent.
- `hash` mode is a low-resource fallback, not the recommended public-facing quality mode.
- `tui.chat_log_max_lines` bounds RichLog memory growth.
- Tree compaction is lossy for inactive branches; disable it if you need full historical payload fidelity when switching back.

## Search And Current-Info Behavior

- `search-ops` supports `tavily` and `brave` providers.
- Time-sensitive answers require fetched-source evidence; otherwise Alphanus will decline to speculate.
- Configure search credentials with environment variables only.
- Search loops are intentionally short: search first, fetch only what is needed, then answer from the gathered evidence.

## Skills

See [SKILLS_GUIDE.md](./SKILLS_GUIDE.md).

Skills live under `skills/<skill-id>/SKILL.md` and can expose tools via:
- `metadata.tools.definitions` command entries (preferred)
- `tools.py` (`TOOL_SPECS` + `execute`) native tool modules
- optional `hooks.py` for `pre_prompt`, `pre_action`, and `post_response`

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
- The script uses a temporary workspace under the repo and a temporary memory file, then runs real `Agent.run_turn(...)` scenarios against the live endpoint.
- Browser scenarios are optional because they trigger desktop-side effects.
