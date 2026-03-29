# Alphanus

Alphanus is a local-first coding assistant with a Textual TUI, a `llama.cpp`-targeted streaming chat-completions loop, explicit session-loaded skills, semantic memory, and a branchable conversation tree.

Status:
- alpha / power-user tooling
- local-first
- secrets stay in environment variables, not editable config fields

## Current State

Current behavior and architecture:
- streaming agent loop with separate reasoning/content handling
- `llama.cpp` `/v1` API integration with OpenAI-style `tool_calls` execution
- Textual UI with live tool previews, code-block popups, config editing, and support-bundle export
- named autosaved sessions with a branchable `ConvTree`
- semantic transformer memory with persistent storage and automatic re-embedding on model change
- current-info discipline: time-sensitive answers are expected to use fetched web evidence
- explicit session-loaded skills via `skill_view(...)`, plus skill enable/disable controls in the TUI
- trusted executable skills from bundled and workspace-local roots; untrusted home/external roots are metadata-only

Bundled skills:
- `workspace-ops`: create/read/edit/move/delete workspace paths, search code, render the workspace tree, and run approved verification commands
- `memory-rag`: store, recall, forget, inspect, and export semantic memories
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

2. Start a `llama.cpp` server that exposes:
- `http://127.0.0.1:8080/v1/chat/completions`
- `http://127.0.0.1:8080/v1/models`

Current backend scope:
- supported/tested: `llama.cpp`
- other inference backends are not yet supported or documented

3. Set environment variables as needed:
- `TAVILY_API_KEY` when `search.provider = "tavily"`
- `BRAVE_SEARCH_API_KEY` when `search.provider = "brave"`
- `ALPHANUS_AUTH_HEADER` for authenticated model endpoints (`Header-Name: value`)
- `AUTH_HEADER` as a fallback if `ALPHANUS_AUTH_HEADER` is unset

4. Run Alphanus:

```bash
uv run alphanus
```

Useful flags:
- `--debug` writes HTTP debug logs to `logs/http-debug.jsonl`
- `--dangerously-skip-permissions` disables shell approval prompts

Notes:
- `.env` at the repo root is auto-loaded on startup without overriding already-set environment variables.
- Startup performs a `/v1/models` readiness handshake before the first turn.
- `uv run main.py` still works from a repo checkout, but `uv run alphanus` is now the primary entrypoint.

## Configuration

Global config lives at `config/global_config.json` in a repo checkout, or `~/.alphanus/config/global_config.json` for an installed package. At load/save time Alphanus:
- merges missing keys from built-in defaults
- normalizes types and clamps invalid values
- strips secret-like fields from disk and from the `/config` editor
- enforces same-host `model_endpoint` and `models_endpoint` unless `agent.allow_cross_host_endpoints = true`

Important runtime notes:
- `agent.request_timeout_s` is a stream idle timeout, not a total-turn deadline.
- `agent.max_tokens` defaults to `null`, so no explicit `max_tokens` cap is sent unless you configure one.
- Internal context-budget defaults exist at runtime, but `agent.context_budget_max_tokens`, `context.context_limit`, and `context.safety_margin` are intentionally hidden from the editable config file and the TUI editor.
- `search.provider` supports `tavily` and `brave`.
- `skills.load.extra_dirs`, `skills.load.watch`, and `skills.load.upward_scan` control skill discovery.
- `agents.enable_skill_agents` and `runtime.ask_user_tool` gate companion-agent and structured follow-up flows.

The checked-in config file is intentionally smaller than the full runtime defaults:

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
    "max_action_depth": 10
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
  "capabilities": {
    "shell_require_confirmation": true,
    "dangerously_skip_permissions": false
  },
  "skills": {
    "strict_capability_policy": false,
    "load": {
      "extra_dirs": [],
      "watch": true,
      "upward_scan": true
    },
    "compat": {
      "vendor_extensions": "major"
    }
  },
  "agents": {
    "enable_skill_agents": true
  },
  "runtime": {
    "ask_user_tool": true
  },
  "tools": {},
  "search": {
    "provider": "tavily"
  },
  "logging": {
    "level": "INFO",
    "format": "json",
    "path": "./logs/runtime.jsonl"
  }
}
```

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
- `/file [path]`
- `/image [path]`
- `/code [n|last]`
- `/quit`, `/exit`, `/q`

Branching:
- `/branch [label]`
- `/unbranch`
- `/branches`
- `/switch <n>`
- `/tree`

Skills and diagnostics:
- `/skills`
- `/reload`
- `/doctor`
- `/skill-on <id>`
- `/skill-off <id>`
- `/skill-unload <id>`
- `/skill-unload-all`
- `/skill-reload`
- `/skill-info <id>`

Memory, workspace, and support:
- `/memory-stats`
- `/context`
- `/workspace-tree`
- `/config`
- `/report [file]`

Keyboard shortcuts:
- `F1` shows the keyboard shortcut reference
- `Ctrl+P` opens the slash-command palette
- `Ctrl+G` focuses the composer
- `F2` toggles tool details
- `F3` toggles thinking mode
- `Ctrl+U` clears the full draft
- `Ctrl+K` deletes from the cursor to the end of the line

Session notes:
- Startup opens a session picker before chat input.
- Sessions are stored under `.alphanus/sessions/` in the current workspace.
- The active session is autosaved after each turn.
- Loaded skill ids are persisted with the session.

## Skills

See [SKILLS_GUIDE.md](./SKILLS_GUIDE.md).

Important behavior:
- Skill discovery scans upward from the workspace for `skills/`, `.claude/skills`, `.agents/skills`, and `.opencode/skills`, then also checks user-local skill roots under the home directory.
- Shadowing priority is `workspace/local` > `bundled` > `user/local` > `external/local`.
- Only bundled and workspace-local roots are trusted executable roots. Home/external roots still show up in `/skills` and `/doctor`, but executable surfaces are blocked.
- `/skill-on` and `/skill-off` control whether a skill is globally enabled.
- Skills become session-loaded when the runtime calls `skill_view(name)` for that skill. Session-loaded skills persist with the session and can be removed with `/skill-unload` or `/skill-unload-all`.
- Core tools from enabled trusted skills remain model-exposed; optional skill runtime surfaces are added when a skill is loaded for the session.

## Memory and Search Notes

Memory:
- Recommended embedding model: `BAAI/bge-small-en-v1.5`.
- `memory.allow_model_download` controls whether uncached embedding weights may be downloaded on first use.
- Existing memories are automatically re-embedded when you switch models.
- Retrieval is semantic-only. The legacy hash backend, lexical fallback behavior, and regex fact lookup path are gone.
- For corrections, prefer `replace_query` or `replace_ids` when calling memory tools.

Search:
- `search-ops` supports `tavily` and `brave`.
- Time-sensitive answers are expected to use fetched source text; otherwise Alphanus declines to speculate.
- Search credentials are environment variables only.
- Default per-turn tool budgets are `web_search=2`, `fetch_url=2`, and `recall_memory=2`.

Workspace and shell safety:
- `delete_path` supports files and directories.
- `run_checks` is limited to approved verification runners (`pytest`, `ruff`, `mypy`, `pyright`, `eslint`, `tsc`, `vitest`, `jest`, `tox`, `nox`) and `uv run <approved-runner>`.
- Shell commands run with confirmation by default and are blocked from shell-control operators and dangerous patterns.

## Tests

```bash
uv run pytest
```

## Live Smoke Checks

Run a real end-to-end smoke pass against your current model server:

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
- The script uses a temporary workspace and temporary memory file.
- Browser scenarios are optional because they trigger desktop-side effects.
