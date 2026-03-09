# Alphanus

Alphanus is a local-first coding assistant that talks to an external `llama-server` endpoint using OpenAI-compatible chat completions.

## Features

- Textual terminal UI with streaming output
- Conversation tree with branch/switch/save/load
- Skill runtime with capability-scoped tool calls
- Workspace-safe file and shell operations
- Persistent vector memory with semantic recall

## Setup

1. Install dependencies:

```bash
uv sync --extra dev
```

2. Start your model server externally (example endpoint defaults):

- `http://127.0.0.1:8080/v1/chat/completions`
- `http://127.0.0.1:8080/v1/models`

3. Run:

```bash
uv run main.py
```

## TUI Commands

- `/help`
- `/think`
- `/branch [label]`, `/unbranch`, `/branches`, `/switch <n>`, `/tree`
- `/skills`, `/skill on <id>`, `/skill off <id>`, `/skill reload`, `/skill info <id>`
- `/memory stats`
- `/workspace tree`
- `/file <path>`
- `/save [file]`, `/load [file]`, `/clear`

## Tests

```bash
uv run pytest
```

## Notes

- Shell commands require confirmation by default; override with `--dangerously-skip-permissions`.
- Write/delete operations are constrained to the configured workspace root.
- Memory is persisted at `memories/memory.pkl` by default.
- For lower RAM usage, keep `memory.embedding_backend` set to `hash` in `config/global_config.json`.
