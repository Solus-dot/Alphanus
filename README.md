# Alphanus

Alphanus is a local-first coding assistant with a Textual TUI, an OpenAI-compatible model client (`llama-server` style endpoints), and a modular AgentSkills-style skill runtime.

## Current Architecture

- Streaming agent loop with reasoning/content token rendering
- Tool-calling loop using OpenAI `tool_calls`
- Skill runtime with `SKILL.md` manifests and command-backed tools
- Workspace-safe file/shell execution policies
- Persistent vector memory with semantic recall fallback behavior
- Branchable conversation tree with save/load
- RAM controls for long sessions (bounded TUI log + inactive branch compaction)

## Quick Start

1. Install dependencies:

```bash
uv sync --extra dev
```

2. Start your model server externally (defaults):
- `http://127.0.0.1:8080/v1/chat/completions`
- `http://127.0.0.1:8080/v1/models`

3. Run Alphanus:

```bash
uv run main.py
```

At startup, Alphanus prints endpoint readiness status, including:
- `waiting for endpoint <...>/v1/models handshake...`

## Important Runtime Notes

- `agent.max_tokens` defaults to `null` (no explicit cap sent).
- System prompt includes the current local date (`agent/prompts.py`).
- Shell commands require confirmation by default.
- `--dangerously-skip-permissions` disables shell confirmation (unsafe).

## TUI Commands

- `/help`
- `/think`
- `/branch [label]`, `/unbranch`, `/branches`, `/switch <n>`, `/tree`
- `/skills`, `/skill on <id>`, `/skill off <id>`, `/skill reload`, `/skill info <id>`
- `/memory stats`
- `/workspace tree`
- `/file <path>`
- `/save [file]`, `/load [file]`, `/clear`

## Memory and RAM Tuning

Edit `config/global_config.json`:

```json
{
  "memory": {
    "embedding_backend": "hash"
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
- `memory.embedding_backend: "hash"` is lowest RAM usage.
- `tui.chat_log_max_lines` bounds RichLog memory growth.
- Tree compaction is lossy for inactive branches; disable it if you need full historical payload fidelity when switching back.

## Skills

See [SKILLS_GUIDE.md](./SKILLS_GUIDE.md).

Skills live under `skills/<skill-id>/SKILL.md` and can expose tools via:
- `metadata.tools.definitions` command entries (preferred)
- `tools.py` (`TOOL_SPECS` + `execute`) legacy fallback

## Tests

```bash
uv run pytest
```
