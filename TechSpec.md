# Alphanus - Technical Specification

**Project:** Personal AI coding assistant
**Platform:** Mac M-series (Apple Silicon only)
**Date:** February 2026

---

## Table of Contents

1. [Project Goals](#1-project-goals)
2. [Architecture Overview](#2-architecture-overview)
3. [LLAMA-SERVER API Contract](#3-llama-server-api-contract)
4. [Core Components](#4-core-components)
5. [Modular Skills System](#5-modular-skills-system)
6. [Skill Catalogue](#6-skill-catalogue)
7. [System Prompt Design](#7-system-prompt-design)
8. [TUI Interface](#8-tui-interface-tuiinterfacepy)
9. [Configuration](#9-configuration)
10. [Architectural Changes and Optimizations](#10-architectural-changes-and-optimizations)
11. [File Structure](#11-file-structure)
12. [Testing Strategy](#12-testing-strategy-tests)
13. [Dependencies](#13-dependencies-pyprojecttoml)
14. [Implementation Checklist](#14-implementation-checklist)
15. [Known Tradeoffs](#15-known-tradeoffs)
16. [Success Criteria](#16-success-criteria)
17. [Normative Contracts and Ops](#17-normative-contracts-and-ops)

---

## 1. Project Goals

### 1.1 What We're Building

A personal AI assistant that:

- Runs entirely on-device via llama-server — local HTTP inference with an OpenAI-compatible API
- Remembers context and facts across sessions using vector-similarity memory
- Writes, reads, and edits code files inside a sandboxed workspace directory
- Executes shell commands (`ls`, `mkdir`, `cd`, `cat`, `grep`, `git`, etc.) with user confirmation before every run
- Has basic utility tools: web search, weather, email reading, home-directory file search
- Is operated via an interactive Textual TUI as the primary interface
- Optionally receives messages from a phone via WhatsApp

### 1.2 What We're NOT Building

- Multiple LLM backends — llama-server only; no Ollama, no OpenAI API, no direct Python bindings
- A web UI or public-facing REST API
- Telegram integration
- Multi-user or session-isolation features
- Enterprise-grade logging, metrics, or tracing

### 1.3 Philosophy

Simple. Fast. Useful. Under 5,000 lines. No over-engineering. Every component does exactly one thing.

---


## 2. Architecture Overview

```text
+-----------------------------------------------------------+
| Entry Points                                              |
|  - Textual TUI (primary)                                  |
|  - WhatsApp Bridge (optional)                             |
+------------------------------+----------------------------+
                               |
                               v
+-----------------------------------------------------------+
| Agent Core                                                 |
|  1) build messages + skill context                         |
|  2) POST /v1/chat/completions (stream=true)               |
|  3) stream reasoning/content to UI                         |
|  4) execute skill-scoped actions if needed                 |
|  5) persist turn + update memory/tree                      |
+--------------------+--------------------+-----------------+
                     |                    |
                     v                    v
              +-------------+      +--------------+
              | Memory (RAG)|      | Workspace    |
              |             |      | Manager      |
              +-------------+      +--------------+
                       \              /
                        \            /
                         v          v
                      +------------------+
                      | Skill Runtime    |
                      | (manifests/hooks)|
                      +------------------+
                               |
                               v
                      +------------------+
                      | Capability       |
                      | Adapters         |
                      | (workspace/shell |
                      |  memory/utility) |
                      +------------------+
```

### 2.1 Key Design Decisions

**llama-server with HTTP API**

The agent communicates with `llama-server` via HTTP at `localhost:8080/v1/chat/completions`. The server is started once and reused for the full session.

- SSE streaming support for responsive TUI rendering
- Separate `reasoning_content` and `content` channels when model supports reasoning
- OpenAI-compatible chat endpoint

All requests use Python stdlib `urllib` (no external HTTP client dependency).

**stream=true for all requests**

All completions use streaming mode. The runtime parses `data:` SSE lines and yields token deltas to the TUI worker.

**Thinking toggle via `chat_template_kwargs`**

Thinking is controlled per request through:

```json
{
  "chat_template_kwargs": { "enable_thinking": true }
}
```

This is exposed as `/think` and reflected in status UI.

**Skill Runtime instead of Tool Registry**

The agent uses modular skills to select behavior and capabilities per turn.

- Skills are discovered from `skills/`
- Skill prompts are composed dynamically
- Optional skill hooks run with safety restrictions
- Workspace/memory/shell/utility actions are capability adapters invoked under skill policy

No centralized tool registry is used.

**Workspace isolation**

All write operations are restricted to the workspace directory. Any path escaping workspace root is rejected pre-execution. Shell actions execute with `cwd` set to workspace root and require user confirmation.

---


## 3. LLAMA-SERVER API Contract

This section documents the HTTP API surface used by this project.

### 3.1 External Model Serving

Alphanus does **not** launch or manage model servers. The user runs their inference server externally (for example, `llama-server`) and exposes an OpenAI-compatible endpoint.

Alphanus only acts as an HTTP client to the configured endpoint. This avoids subprocess pipe-drain deadlock risks entirely.

### 3.2 API Endpoint

The endpoint is user-configured in Alphanus.

Default:

```
POST http://127.0.0.1:8080/v1/chat/completions
```

#### Endpoint Readiness Policy (single policy)

Before the first generation request, Alphanus performs a readiness check by polling `<base_url>/v1/models` every 0.5 seconds for up to 30 seconds.

- If ready: proceed.
- If not ready: return a clear connection error and keep the app running.
- No auto-restart is attempted by Alphanus, since server lifecycle is external.

### 3.3 Request Schema

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.1,
  "top_p": 1.0,
  "top_k": 0,
  "max_tokens": 1024,
  "stream": true,
  "chat_template_kwargs": {"enable_thinking": true}
}
```

Skill selection and capability permissions are resolved inside the agent runtime, not by passing a `tools` array.

### 3.4 Response Schema

#### Streaming Response (Server-Sent Events)

Each event has format `data: <json>\n\n`:

```json
data: {"id":"chatcmpl-...","choices":[{"delta":{"role":"assistant","reasoning_content":"The"}}]}

data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-...","choices":[{"finish_reason":"stop"}]}

data: [DONE]
```

### 3.5 Key Response Fields

- **`reasoning_content`**: The model's internal chain-of-thought, emitted before `content`. Present only on reasoning-capable models. Normalized to `reasoning` when stored in history.
- **`content`**: The actual response text shown to the user.
- **`finish_reason`**: `"stop"` (finished naturally) or `"length"` (hit `max_tokens`).

### 3.6 Sampling Parameters

**Server-level parameters** (set at startup via CLI flags, loaded from `models_config.json`):

- `ctx_size` — context window size (`--ctx-size`)
- `n_gpu_layers` — Metal acceleration layer count (`--n-gpu-layers`)

**Request-level parameters** (passed in each API request):

- `temperature` (float, default: 0.0)
- `top_p` (float, default: 1.0)
- `top_k` (int, default: 0)
- `min_p` (float, default: 0.0)
- `repetition_penalty` (float, default: 1.0)
- `max_tokens` (int, default: 512)

### 3.7 Skill Action Flow

1. **Send request with skill context** (system prompt + selected skill instruction block):

```python
payload = json.dumps({
    "messages": history,
    "stream": True,
    "chat_template_kwargs": {"enable_thinking": thinking},
}).encode()
```

2. **Process stream and separate reasoning from content**:

```python
accumulated_content = ""
accumulated_reasoning = ""

with urllib.request.urlopen(req, timeout=180) as resp:
    for raw in resp:
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data:"):
            continue
        ds = line[5:].strip()
        if ds == "[DONE]":
            break
        chunk = json.loads(ds)
        delta = chunk["choices"][0].get("delta", {})

        r = delta.get("reasoning_content") or ""
        c = delta.get("content") or ""
        if r:
            accumulated_reasoning += r
            yield "reasoning", r
        if c:
            accumulated_content += c
            yield "content", c
```

3. **Execute skill actions when needed**:

- Agent runtime evaluates active skills and context
- Selected skill action is executed through capability adapters
- Result is appended to history as structured assistant/context message
- Loop continues until final assistant answer is produced

4. **Stop condition**: assistant emits final response for the turn.

### 3.8 Error Handling

Standard HTTP status codes:

- `200 OK`: Success
- `400 Bad Request`: Invalid request JSON
- `500 Internal Server Error`: Model inference failed

Network errors and connection refusals are caught and returned as error strings to the TUI. A failed generation marks the active conversation turn as cancelled (not complete) so it shows a `✗` state in the tree rather than silently appearing as a finished response.

### 3.9 Server Management

Server lifecycle is out of scope for Alphanus.

Alphanus must:

1. Read `agent.model_endpoint` from config
2. Perform readiness check (`/v1/models`) before first generation
3. Retry transient request failures at most once per turn
4. Surface endpoint errors without crashing the TUI

---


## 4. Core Components

### 4.1 Agent (`agent/core.py`)

The central class. Manages endpoint communication, skill orchestration, conversation history, and stream lifecycle.

**Responsibilities:**

- Build request payloads with system prompt + selected skill instruction block
- Maintain conversation history as a flat list of message dicts
- Stream responses (`stream=true`) and forward token events to the TUI via `call_from_thread`
- Invoke skill-scoped actions when requested by the runtime policy
- Expose `run(user_input)` as the public interface

**Agent loop logic:**

1. Append user message to history
2. Resolve active skills for the turn (context + priority)
3. Build request payload with messages, sampling params, and thinking toggle
4. POST to `/v1/chat/completions` with `stream=true`
5. Stream reasoning/content tokens to the interface
6. If a skill action is required, execute via skill runtime and append results to history
7. If the action batch is write-only and successful (`create_file`/`edit_file`/`delete_file`), fast-finalize locally to avoid an extra model pass
8. Otherwise continue until assistant emits a final answer, then complete turn

**Execution guardrails:**
- Maximum skill action depth per turn (default: 10)
- Per-action timeout and structured error return
- Deterministic replay in debug mode (active skills + action trace)

---

### 4.2 Conversation Tree (`core/conv_tree.py`)

Conversation history is stored as a **tree of paired turn nodes** rather than a flat list. This enables branching: the user can fork a conversation at any point, explore a tangent, then return to the original thread without losing either path.

#### 4.2.1 Data Structure

Each node is a **Turn** — a single complete exchange pairing one user message with one assistant reply:

```python
@dataclass
class Turn:
    id:                str            # 8-character UUID prefix
    user_content:      Any            # str or list[dict] for multimodal
    assistant_content: Optional[str]  # None until streaming completes
    parent:            Optional[str]  # parent Turn id; None for root's children
    children:          List[str]      # ordered list of child Turn ids
    label:             str            # optional branch name
    branch_root:       bool           # True if this turn started a new branch
```

The full tree is stored as a flat `dict[id → Turn]`. A sentinel root node (id `"root"`) has no content; real turns are its descendants.

`assistant_content` is deliberately `None` during streaming. The turn is added to the tree before the response arrives, so the structure is finalized before the model starts generating. It is filled in-place by `complete_turn()` on success or `cancel_turn()` on interruption or error.

#### 4.2.2 Active Path

The **active path** is derived on demand by walking parent pointers from `current_id` up to the root. This path is what `history_messages()` serializes into the flat `messages` list sent to the server.

```python
def history_messages(self) -> List[dict]:
    msgs = []
    for turn in self.active_path:
        if turn.id == "root":
            continue
        msgs.append({"role": "user", "content": turn.user_content})
        if turn.assistant_content:
            msgs.append({"role": "assistant", "content": turn.assistant_content})
    return msgs
```

#### 4.2.3 Skill Action Exchanges

Skill action exchanges are not modeled as separate `Turn` nodes. They are stored as a sub-list on the same turn that triggered them.

```python
@dataclass
class Turn:
    ...
    skill_exchanges: List[dict]  # action requests/results linked to this turn
```

When serializing `history_messages()`, each turn expands in order:

1. `{"role": "user", "content": turn.user_content}`
2. entries in `turn.skill_exchanges`
3. `{"role": "assistant", "content": turn.assistant_content}`

This keeps one logical user->assistant exchange per node while preserving intermediate skill interactions for model context.

#### 4.2.4 Branching

Branching is armed with a pending flag rather than being triggered at add-time:

```python
def arm_branch(self, label: str = ""):
    self._pending_branch = True
    self._pending_branch_label = label

def add_turn(self, user_content: Any) -> Turn:
    is_branch = self._pending_branch
    label     = self._pending_branch_label
    self._pending_branch = False          # flag consumed on send
    self._pending_branch_label = ""
    ...
    turn = Turn(..., branch_root=is_branch, label=label)
```

The flag is consumed when the user sends their next message, not when `/branch` is typed. This means the tree shape is locked in before streaming begins, which simplifies the cancel/error path.

To move between branches:

- `unbranch()` — walks up the parent chain to the nearest `branch_root` ancestor, then sets `current_id` to that node's parent (the fork point)
- `switch_child(idx)` — moves `current_id` to the nth child of the current node
- After either operation, the TUI clears and replays the chat log from the new active path

#### 4.2.5 Turn Lifecycle

```
arm_branch()  [optional]
      │
      ▼
add_turn(user_content)        → assistant_content = None, turn added to tree
      │
      ▼
_start_stream(turn)           → @work(thread=True) worker begins streaming
      │
      ├── [stream runs]       → tokens posted to main thread via call_from_thread
      │
      ├── complete_turn(id, reply)    → assistant_content filled in, turn ✓
      │
      ├── cancel_turn(id, partial)    → partial + "[interrupted]" stored, turn ✗
      │                                 (triggered by ESC confirm or network error)
      │
      └── errored turn               → cancel_turn() called with empty string
```

#### 4.2.6 Persistence

The full tree serializes to JSON via `to_dict()` / `from_dict()`. `save(path)` writes atomically (temp file renamed). `load(path)` restores the full node dict and `current_id`. The session resumes from exactly where it was saved.

#### 4.2.7 Tree Visualization

`render_tree()` returns a list of `(text, tag, is_active)` rows for the TUI to render. Markers:

- `●` — current node
- `○` — on the active path
- `·` — exists but not on the active path
- `✓` — complete turn
- `…` — streaming in progress
- `✗` — interrupted or errored

---

### 4.3 Memory System (`core/memory.py`)

Provides persistent long-term memory and context retrieval using offline vector similarity search.

**Class:** `VectorMemory`

**Responsibilities:**

- **Offline-first vectorization:** Uses `sentence-transformers` (`all-MiniLM-L6-v2`) initialized with `local_files_only=True`. Downloads from Hugging Face only on first run if the local cache is missing.
- **Data structure:** Each memory is a dict containing:
  - `id` — integer index
  - `text` — raw string content
  - `vector` — NumPy embedding array
  - `metadata` — flexible attributes (importance, source, tags)
  - `type` — categorical string (`conversation`, `fact`, `preference`, `goal`)
  - `timestamp` — Unix epoch of creation
  - `access_count` and `last_accessed` — tracks retrieval frequency for future pruning
- **Semantic search:** Filters by optional `memory_type`, computes cosine similarity against the query vector, drops results below `min_score` threshold (default `0.3`)
- **Auto-saving:** Triggers an atomic `.save()` immediately after any add, delete, or search to prevent data loss. Uses a `.tmp` → `.pkl` rename to prevent corruption
- **Resilient loading:** On `EOFError` or unpickling failure, backs up the broken file to `.corrupted` and starts fresh rather than crashing the agent
- **Human-readable export:** `export_txt()` dumps all memories to a formatted text log

**Configuration:**

- `model_name`: `all-MiniLM-L6-v2`
- `storage_path`: `Alphanus/memories/memory.pkl`
- `dimension`: derived automatically from the encoder (384 for MiniLM)

---

### 4.4 Workspace Manager (`core/workspace.py`)

Manages the workspace directory where all agent-written files live.

**Responsibilities:**

- **Dual-zone path validation:**
  - *Write operations:* Strict. Destination must resolve inside `workspace_root`. Any attempt to write outside raises `PermissionError`.
  - *Read operations:* Permissive. Source must resolve inside `home_directory`. System paths (`/etc`, `/var`) are blocked.
- **Path resolution:**
  - Relative paths are resolved relative to `workspace_root`
  - Absolute and `~`-prefixed paths are accepted for read operations only, provided they stay within the user's home directory
- **Sensitive file masking:** Explicitly blocks read access to `.ssh/*`, `.env`, `.bash_history`, `*.pem` etc. even within the home directory
- **Visual tree:** `workspace_tree` renders only `workspace_root`

**Configuration:**

- `workspace_root`: `~/Desktop/Alphanus-Workspace`
- `home_root`: `~` (user's actual home directory)
- `blocked_patterns`: `['.ssh', '.aws', '.gnupg', 'id_rsa', '.env', '.DS_Store']`

---

### 4.5 Skill Runtime (`core/skills.py`)

A lightweight runtime that discovers, activates, and applies modular skills.

**Responsibilities:**

- Load skill manifests from `skills/<skill_id>/skill.toml`
- Parse and cache prompt snippets from each skill (`prompt.md`)
- Discover tool definitions from each skill's `tools.py`
- Score skills against turn context (keywords, file types, branch metadata)
- Build a deterministic skill instruction block for each request
- Execute optional hook callbacks in a restricted sandbox
- Fail closed: skill errors are logged and skipped; they never crash the agent

The runtime replaces the old centralized tool-registry abstraction. Each skill can ship its own tool schemas and execution function in `tools.py`, and those tools are exposed only when the skill is active and capability-allowed.

---

### 4.6 Context Manager (`agent/context.py`)

Prevents the agent from crashing when conversation history exceeds the model's context window.

**Responsibilities:**

- Estimate token usage before each generation step using the loaded tokenizer
- If `estimated_prompt_tokens + max_tokens > context_limit - safety_margin`, prune history:
  1. Always keep `history[0]` (system prompt)
  2. Always keep the last `keep_last_n` messages
  3. Discard messages between index 1 and `-(keep_last_n + 1)`
  4. Insert `{"role": "system", "content": "[...Earlier conversation history pruned for length...]"}` at index 1

**Why not summarization?** An extra LLM pass adds latency and non-determinism. A sliding window with a hard system-prompt anchor is instant and sufficient for 95% of tasks.

**Configuration:**

- `context_limit`: integer (default: 8192 or model max)
- `keep_last_n`: integer (default: 10)
- `safety_margin`: integer (default: 500)

---


## 5. Modular Skills System

Skills extend behavior in a controlled way without hardcoding domain logic into the base agent.

### 5.1 Skill Directory Structure

```text
skills/
  <skill_id>/
    skill.toml
    prompt.md
    tools.py                 # optional but recommended for modular tool definitions
    examples.md              # optional
    schemas/
      input.schema.json      # optional
    hooks.py                 # optional
    tests/                   # optional
```

### 5.2 `skill.toml` Schema (Minimum)

```toml
id = "python-refactor"
name = "Python Refactor"
version = "1.0.0"
description = "Refactoring guidance and safe-change policy"
enabled = true
priority = 50

[triggers]
keywords = ["refactor", "cleanup", "rename"]
file_ext = [".py"]
capabilities = ["workspace_edit", "run_shell_command"]
```

#### Canonical Capability Adapter Names

| Capability | Adapter Contract |
| --- | --- |
| `workspace_read` | Read file/list directory under home/workspace policy |
| `workspace_write` | Create or overwrite workspace file |
| `workspace_edit` | Replace or patch workspace file contents |
| `workspace_delete` | Delete workspace file |
| `workspace_tree` | Render workspace tree summary |
| `run_shell_command` | Execute confirmed shell command in workspace cwd |
| `memory_store` | Persist memory item |
| `memory_recall` | Semantic retrieval over memory store |
| `memory_list` | List recent memories |
| `memory_forget` | Delete memory by id |
| `memory_stats` | Return memory statistics |
| `memory_export` | Export memories to text file |
| `utility_weather` | Fetch weather by city |
| `utility_email_read` | Read recent email metadata |
| `utility_file_search` | Search files under user/home scope |
| `utility_open_url` | Open URL in default browser |
| `utility_play_youtube` | Play YouTube result for topic |

### 5.3 Runtime Contract

- `SkillManifest`: parsed metadata
- `SkillContext`:
  - user input
  - active branch info
  - attachment metadata
  - workspace root
  - memory hits (RAG)
- Optional hooks in `hooks.py`:
  - `pre_prompt(context) -> str | None`
  - `pre_action(context, action_name, args) -> (allow, reason)`
  - `post_response(context, text) -> None`
- Optional tool module in `tools.py`:
  - `TOOL_SPECS: dict[str, {capability, description, parameters}]`
  - `execute(tool_name, args, env) -> normalized envelope`

### 5.4 Selection and Prompt Injection

1. Discover enabled skills from `skills/`
2. Default personal mode: activate all enabled skills up to `skills.max_active_skills`
3. Optional scored mode: score by trigger match and priority, then pick top N
4. Build deterministic skill instruction block
5. Discover tools from selected skills' `tools.py`
6. Apply prompt budget caps before injection:
   - `skill_prompt_budget_tokens` default: 15% of context window
   - hard cap: max 3 skill prompts per turn
   - overflow policy: keep highest-score skills; truncate lowest-score prompt tails first
7. Inject bounded skill block into system/context message before completion call
8. Expose selected-skill tools; capability filtering is strict only when `skills.strict_capability_policy=true`

### 5.5 Safety Rules

- Skill hooks cannot bypass workspace sandbox checks
- Skill hooks cannot auto-confirm shell commands
- Hook failure is non-fatal: skill is skipped and warning logged
- All active skill IDs included in debug trace for reproducibility

### 5.6 TUI / CLI UX Additions

Add command set:

- `/skills` list installed skills and status
- `/skill on <id>` enable skill
- `/skill off <id>` disable skill
- `/skill reload` re-read manifests
- `/skill info <id>` show trigger and capability summary

Status line can show a compact `skills: n` indicator and optional active IDs in sidebar detail view.

### 5.7 File Structure Extension

The Section 9 project tree should be extended with:

```text
Alphanus/
├── skills/
│   ├── python-refactor/
│   │   ├── skill.toml
│   │   ├── prompt.md
│   │   ├── tools.py
│   │   └── hooks.py
│   └── git-assistant/
│       ├── skill.toml
│       ├── prompt.md
│       └── tools.py
├── core/
│   └── skills.py
└── tests/
    └── test_skills.py
```

### 5.8 Interaction with RAG Memory

Skills can consume memory retrieval results through `SkillContext`, but memory indexing/retrieval remains centralized in `core/memory.py`.

This keeps:
- one canonical memory store
- consistent retrieval policy
- no per-skill duplicate vector indexes

## 6. Skill Catalogue

The platform is organized around modular skills instead of a centralized tool registry. Each skill encapsulates prompt guidance, optional hooks, and capability permissions.
Executable tool schemas and handlers live in each skill's `tools.py`.

### 6.1 Workspace Skill (`skills/workspace-ops`)

| Skill Action | Description | Required Parameters | Path Permissions |
| --- | --- | --- | --- |
| `create_file` | Create or overwrite a file with content | `filepath` (relative), `content` | **Workspace only** |
| `edit_file` | Replace contents of an existing file | `filepath` (relative), `content` | **Workspace only** |
| `read_file` | Read full contents of a file | `filepath` (relative or absolute) | **Home read / Workspace read** |
| `list_files` | List files in a directory | `path` (optional, default workspace root) | **Home read / Workspace read** |
| `delete_file` | Delete a file | `filepath` (relative) | **Workspace only** |
| `workspace_tree` | Render a tree view of the workspace | none | **Workspace only** |

### 6.2 Shell Skill (`skills/shell-ops`)

| Skill Action | Description | Required Parameters |
| --- | --- | --- |
| `shell_command` | Execute a shell command in workspace after user confirmation | `command` |

Policy:
- Confirmation required on every command
- `cwd` forced to workspace root
- 30-second execution timeout
- Dangerous patterns (`sudo`, destructive non-workspace deletes) are blocked

### 6.3 Memory Skill (`skills/memory-rag`)

| Skill Action | Description | Required Parameters |
| --- | --- | --- |
| `store_memory` | Persist a fact, preference, identity detail, or goal | `text`, `memory_type` (optional), `importance` (optional) |
| `recall_memory` | Semantic search over stored memories | `query`, `top_k` (optional, default 5) |
| `list_memories` | List most recently stored memories | `count` (optional, default 5) |
| `forget_memory` | Delete memory by integer ID | `memory_id` |
| `get_memory_stats` | Return memory counts and recency stats | none |
| `export_memories` | Export memories to text file | `filepath` (optional) |

### 6.4 Utility Skill (`skills/utilities`)

| Skill Action | Description | Required Parameters |
| --- | --- | --- |
| `get_weather` | Fetch current weather for a city | `city` |
| `read_email` | Read N latest email subjects/senders via IMAP | `count` (optional) |
| `search_home_files` | Recursive filename search under a directory | `query`, `directory` (optional) |
| `open_url` | Open a URL in the default browser | `url` |
| `play_youtube` | Play first YouTube result for a topic | `topic` |

---


## 7. System Prompt Design

The system prompt is assembled at startup and injected as the first message in every request. It has four sections:

**Identity and workspace context** — names the agent, states workspace path, and describes purpose.

**Core behavioural rules** — prioritize memory lookup for personal/contextual facts; write code to files when asked; use workspace-safe operations; never fabricate file contents; respect shell confirmation boundaries.

**Skill guidance** — active skills contribute scoped instruction blocks at runtime. The base prompt remains compact; skill prompts are composed per turn by the skill runtime.

**Safety invariants** — workspace containment, confirmation for shell actions, and restricted paths cannot be bypassed by skills.

The system prompt is static for the session as `history[0]`. Skill blocks are dynamic turn-level additions and are not persisted as static system text.

---


## 8. TUI Interface (`tui/interface.py`)

The primary interface is built with **Textual**. This section reflects the implemented behavior in `playground/llama_tui.py`.

### 8.1 Layout

Main screen structure:

1. Main area (`#main-area`, horizontal)
- Chat pane (`#chat-scroll`) with committed log (`#chat-log`) and streaming preview row (`#partial`)
- Conversation sidebar (`#sidebar`) shown only when terminal width >= 120

2. Footer (`#footer`)
- separator (`#footer-sep`)
- status row 1 (`#status1`): thinking state, branch state, pending attachments
- status row 2 (`#status2`): idle shortcuts or generation spinner/interruption hint
- input row with custom `ChatInput`

### 8.2 Streaming Architecture

- Worker uses Textual `@work(thread=True, exclusive=True)`
- SSE lines parsed from `data:` events
- Delta channels handled separately:
  - `reasoning_content` -> dimmed reasoning block
  - `content` -> assistant answer block
- UI updates are posted using `call_from_thread`
- Scroll-to-end is throttled to reduce redraw jitter

### 8.3 ESC Interrupt (Two-Press Confirm)

- First `Esc` during generation arms interrupt warning
- Second `Esc` within 3 seconds cancels stream (`stop_event.set()`)
- If timeout elapses, interrupt arm state resets
- `Esc` while idle clears input line

Cancelled turns are persisted with `[interrupted]` marker and shown as failed in tree (`x` state).

### 8.4 Built-in Slash Commands (Implemented)

| Command | Description |
| --- | --- |
| `/help` | Show commands and keyboard hints |
| `/think` | Toggle `enable_thinking` |
| `/branch [label]` | Arm next message as branch fork |
| `/unbranch` | Return to nearest branch parent |
| `/branches` | List child turns of current node |
| `/switch <n>` | Switch active path to child index `n` |
| `/tree` | Render full tree |
| `/save [file]` | Save tree JSON (default: `llamachat_tree.json`) |
| `/load [file]` | Load tree JSON |
| `/file <path>` | Attach image or text file to next message |
| `/image <path>` | Alias of `/file` behavior |
| `/clear` | Reset conversation tree and UI log |
| `/quit`, `/exit`, `/q` | Exit app |

### 8.5 Keyboard Shortcuts

| Key | Action |
| --- | --- |
| `Enter` | Send message |
| `Esc` | Clear input when idle; two-press interrupt while streaming |
| `PgUp / PgDn` | Scroll chat |
| `Ctrl-U` | Clear full input |
| `Ctrl-K` | Kill input from cursor to end |
| `Ctrl-C / Ctrl-D` | Quit |

### 8.6 Planned Skill UX (Not Yet Implemented)

| Command | Description |
| --- | --- |
| `/skills` | List installed skills and enabled/disabled state |
| `/skill on <id>` | Enable a skill |
| `/skill off <id>` | Disable a skill |
| `/skill reload` | Reload manifests from disk |
| `/skill info <id>` | Show manifest, triggers, and permissions |


## 9. Configuration

**Global config path:** `Alphanus/config/global_config.json`
**Optional model profiles path:** `Alphanus/config/models_config.json`

`global_config.json` is created on first run with defaults.

### Configuration Schema

**`global_config.json`**

```
agent
  model_endpoint      - Chat completions endpoint URL
                        (default: http://127.0.0.1:8080/v1/chat/completions)
  models_endpoint     - Health endpoint URL
                        (default: http://127.0.0.1:8080/v1/models)
  request_timeout_s   - Per-request timeout in seconds (default: 180)
  readiness_timeout_s - Endpoint readiness timeout seconds (default: 30)
  readiness_poll_s    - Readiness poll interval seconds (default: 0.5)
  enable_thinking     - Default thinking state at startup (default: true)
  max_tokens          - Optional max output tokens (null disables cap)
  max_action_depth    - Max tool/action depth per turn (default: 10)

context
  context_limit       - Approx context window size used for pruning
  keep_last_n         - Minimum recent messages retained
  safety_margin       - Reserved token margin before hard limit

workspace
  path                - Absolute or ~ path to workspace
                        (default: ~/Desktop/Alphanus-Workspace)

memory
  path                - Absolute or ~ path to memory pickle
                        (default: Alphanus/memories/memory.pkl)

capabilities
  shell_require_confirmation - bool, must always be true (default: true)
  email_enabled              - bool (default: false)
  email_imap_server          - IMAP host (default: imap.gmail.com)

skills
  selection_mode             - "all_enabled" (default) or scored mode
  max_active_skills          - Active skill cap (0 => no cap)
  strict_capability_policy   - Enforce capability matching fail-closed (default: false)

whatsapp
  enabled             - bool (default: false)
```

**`models_config.json` (optional)**

Used only for named model/server profiles in the UI (no local model launch):

```
[profile_name]
  endpoint            - Chat completions endpoint URL
  models_endpoint     - Models endpoint URL
  temperature         - Default sampling temperature
  top_p               - Default nucleus sampling threshold
  top_k               - Default top-k sampling
  max_tokens          - Default maximum tokens per generation
```

### Environment Variables (`.env`)

Sensitive credentials are never stored in JSON config. They go in a `.env` file at the project root:

```
EMAIL_USER=your_email@example.com
EMAIL_PASSWORD=your_app_password
```


## 10. Architectural Changes and Optimizations

These changes keep the original design (RAG memory, workspace isolation, WhatsApp optional path) and make implementation more modular and testable.

### 10.1 Module Split

Refactor into focused modules:

Canonical conversation-tree import path: `from core.conv_tree import ConvTree, Turn` (used by `tui/interface.py` and agent modules).

- `agent/core.py` (orchestration loop)
- `core/conv_tree.py`
- `core/streaming.py`
- `core/attachments.py`
- `core/memory.py` (RAG retrieval/index lifecycle)
- `core/workspace.py` (path safety + cwd policy)
- `core/skills.py` (new skill runtime)
- `tui/interface.py` (Textual rendering + command router)

### 10.2 Performance

- Keep existing throttled UI scrolling
- Add token coalescing buffer (flush every 20-40 ms) before render
- Add configurable conversation budget + summarization of stale branches
- Cache sidebar render rows and only diff-update on tree mutations

### 10.3 Reliability

- Health-check llama-server before first request and on transport errors
- No process restart logic in Alphanus; only per-turn request retry with clear failure messaging
- Atomic save for conversation tree and memory indexes
- Add schema versioning for saved tree JSON

### 10.4 Workspace and Safety

- Keep hard path normalization and workspace-root containment checks
- Keep shell confirmation gate by default
- Add explicit allowlist/denylist per skill and per capability action

### 10.5 WhatsApp Integration Path

- Keep WhatsApp as optional entrypoint feeding same agent core
- Normalize incoming WhatsApp message events into the same turn schema
- Preserve branch semantics with fixed WhatsApp branch label format: `wa:<chat_id>:<YYYYMMDD>`

---


## 11. File Structure

```text
Alphanus/
│
├── agent/
│   ├── __init__.py
│   ├── core.py                 # Agent class — endpoint client, stream loop, skill orchestration
│   ├── context.py              # Context window pruning
│   └── prompts.py              # build_system_prompt()
│
├── core/
│   ├── __init__.py
│   ├── conv_tree.py            # ConvTree + Turn
│   ├── memory.py               # RAG memory store
│   ├── workspace.py            # WorkspaceManager
│   └── skills.py               # SkillRuntime
│
├── skills/
│   ├── workspace-ops/
│   │   ├── skill.toml
│   │   ├── prompt.md
│   │   └── tools.py
│   ├── shell-ops/
│   │   ├── skill.toml
│   │   ├── prompt.md
│   │   └── tools.py
│   ├── memory-rag/
│   │   ├── skill.toml
│   │   ├── prompt.md
│   │   └── tools.py
│   └── utilities/
│       ├── skill.toml
│       ├── prompt.md
│       └── tools.py
│
├── tui/
│   ├── __init__.py
│   └── interface.py            # Textual app, widgets, slash commands
│
├── whatsapp/                   # Optional
│   ├── __init__.py
│   ├── bridge.py               # Receive WhatsApp message -> agent.run() -> send reply
│   └── sessions.py             # ChatSession by chat id
│
├── tests/
│   ├── conftest.py
│   ├── test_workspace.py
│   ├── test_memory.py
│   ├── test_skills.py
│   ├── test_conv_tree.py
│   └── test_agent_loop.py
│
├── config/
│   ├── global_config.json
│   └── models_config.json
│
├── memories/
│   └── memories.pkl
│
├── .env
├── main.py
├── pyproject.toml
└── README.md
```

---


## 12. Testing Strategy (`tests/`)

Testing focuses on deterministic, high-risk components. The stack uses `pytest` and `pytest-mock` only.

### 12.1 Sandbox Security

The `WorkspaceManager` is the most critical security component. Tests must assert `PermissionError` is raised for:

- **Path traversal:** Using `../` to escape the workspace root
- **Absolute write escapes:** Attempting to write to `/tmp/`, `/etc/`, or `~/Documents/`
- **Restricted reads:** Attempting to read `~/.ssh/id_rsa` or `~/.env`

### 12.2 Conversation Tree

`ConvTree` must be tested for correctness across its full lifecycle:

- **Add and complete:** `add_turn()` → `complete_turn()` → turn shows `✓`
- **Add and cancel:** `add_turn()` → `cancel_turn()` → turn shows `✗`, partial text preserved
- **Branch and unbranch:** After branching, `active_path` includes only the correct lineage; `unbranch()` moves `current_id` to the fork point
- **Switch child:** `switch_child(n)` updates `current_id`; subsequent `active_path` reflects the new branch
- **`history_messages()` correctness:** Verify the serialized flat message list matches the active path, including skill exchanges in the correct order
- **Serialization round-trip:** `save()` → `load()` must restore the full tree with identical nodes and `current_id`
- **Cancelled turn in history:** A turn with `[interrupted]` in `assistant_content` must appear in `history_messages()` with its partial reply included

### 12.3 Skill Logic

Skills and capability adapters are tested in isolation:

- **Workspace skill actions:** File CRUD with `pytest`'s `tmp_path` fixture
- **Memory system:** Temporary `.pkl` file; verify similarity search thresholding, storage, deletion logic, and the corrupt-file recovery path
- **Utility skill actions:** Mock all external network calls with `pytest-mock`

**Memory scale test:** Linear O(n) scan must complete in under 500ms on an M2 Mac with 5,000 memories loaded.

### 12.4 Agent Loop (Mocked Server)

Running the actual llama-server during unit tests is too slow and hardware-dependent. Use `pytest-mock` to patch the `urllib.request.urlopen` call and return predefined SSE sequences:

1. A stream that triggers one or more skill actions before final completion
2. A stream ending with `finish_reason: "stop"` with final content

Assert that `agent.run()` correctly:

1. Sends the POST request with `stream=true`
2. Parses model output and skill-action trigger points
3. Executes the selected skill action through capability adapters
4. Appends skill action results to history with normalized `reasoning`
5. Loops back to send another request
6. Streams the final response and exits when `finish_reason` is `"stop"`

Also test the error path: a `urllib.error.URLError` mid-stream must result in `cancel_turn()` being called (not `complete_turn()`), and the turn must show `✗` in the tree.

---


## 13. Dependencies (`pyproject.toml`)

```
textual >= 0.60                  TUI framework (widgets, layout, async workers)
sentence-transformers >= 5.2.0   Memory vector encoding (CPU)
numpy >= 2.4.2                   Vector math for cosine similarity
python-dotenv >= 1.2             Loads .env secrets
pytest >= 9.0.2                  Testing
pytest-mock >= 3.15.1            Mocking HTTP responses
```

`llama-cpp-python` is optional and only needed if the same machine is also used to host `llama-server` outside Alphanus.

No `requests`, no `httpx`, no `openai`, no `ollama`, no `mcp`, no `fastapi`. All HTTP is handled via stdlib `urllib`. The `sentence-transformers` model downloads once on first use and caches locally.

---


## 14. Implementation Checklist

### Week 1 — Foundation

- [ ] Create project directory structure and `pyproject.toml`
- [ ] Central config loader with endpoint profile selection in `main.py`
- [ ] `main.py` — `--debug` CLI argument for verbose logging
- [ ] `core/skills.py` — SkillRuntime (discover, score, compose, execute actions)
- [ ] `core/memory.py` — VectorMemory
- [ ] `core/conv_tree.py` — ConvTree + Turn, full branch/switch/save/load/history_messages
- [ ] `agent/prompts.py` — `build_system_prompt()`
- [ ] `agent/core.py` — Agent class: endpoint client, streaming loop, skill action execution
- [ ] `main.py` — load config, verify endpoint readiness, launch TUI

### Week 1–2 — Workspace and Skills

- [ ] `core/workspace.py` — WorkspaceManager with dual-zone path validation
- [ ] `skills/workspace-ops/skill.toml` + `prompt.md` — workspace skill actions
- [ ] `skills/shell-ops/skill.toml` + `prompt.md` — shell action policy
- [ ] `skills/memory-rag/skill.toml` + `prompt.md` — memory skill actions
- [ ] `skills/utilities/skill.toml` + `prompt.md` — utility skill actions
- [ ] Verify: agent writes a file, reads it back, runs `ls -la`

### Week 2 — TUI

- [ ] `tui/interface.py` — full Textual `App` subclass: `RichLog` chat pane, `Input` widget, all slash commands, `@work` stream worker, ESC two-press interrupt
- [ ] Verify: `/think` state reflects correctly; `/tree` renders correctly; `/branch` + `/unbranch` round-trip works
- [ ] Verify: Ctrl+C/D quit cleanly; startup banner shows on mount

### Week 2–3 — Testing and Polish

- [ ] `tests/test_conv_tree.py` — full branch lifecycle, skill_exchanges serialization, cancel path
- [ ] `tests/test_workspace.py` — sandbox security boundary tests
- [ ] `tests/test_memory.py` — similarity search, scale test, corrupt recovery
- [ ] `tests/test_skills.py` — skill execution and error handling
- [ ] `tests/test_agent_loop.py` — mocked SSE with skill-action paths and stop paths
- [ ] End-to-end: "write me a Flask app" → file appears in workspace
- [ ] End-to-end: "what do I prefer?" → memory recall works
- [ ] End-to-end: "run ls -la" → confirmation prompt appears, command runs
- [ ] Edge case: server connection error during stream → turn marked `✗`, error written to chat log
- [ ] Edge case: max skill-action depth guard fires correctly
- [ ] Tune skill prompts if action-selection compliance is poor on chosen model
- [ ] Write README with setup instructions

### Week 4 (optional) — WhatsApp

- [ ] `whatsapp/bridge.py` — receive message, call `agent.run()`, send reply
- [ ] `whatsapp/sessions.py` — ChatSession per chat_id, 24-hour TTL
- [ ] Session file format: `whatsapp_sessions/{chat_id}.json`
- [ ] Cleanup task for sessions older than 24 hours
- [ ] Test: send message from phone, receive response

---


## 15. Known Tradeoffs

| Tradeoff | Impact | Mitigation |
| --- | --- | --- |
| stdlib `urllib` only | Slightly more verbose HTTP code than `requests` | Eliminates one dependency; streaming via `urlopen` context manager is straightforward |
| HTTP overhead vs direct API | ~10–50 ms latency per request | Acceptable for local server; eliminates Python model lifecycle complexity |
| External model server dependency | Alphanus cannot recover if endpoint is down | Readiness check + clear reconnect guidance + one per-turn retry |
| Endpoint/model warm-up outside Alphanus | First requests may be slow | Document warm-up expectation; keep request retry lightweight |
| Textual as a dependency | Adds ~2 MB; requires pip install | Eliminates all manual terminal rendering, encoding bugs, and resize handling |
| Textual `@work` thread vs asyncio | Worker threads block asyncio-unaware code | stdlib `urllib` streaming runs fine in threads; no async HTTP client needed |
| sentence-transformers on CPU | Memory search adds ~100ms per query | Acceptable for personal use; model is small (90 MB) |
| Linear memory scan O(n) | Degrades beyond ~10K memories | Document scale limits; suggest FAISS or pgvector migration at that threshold |
| Conversation tree in memory | Tree grows unbounded across very long sessions | `save()`/`load()` lets the user checkpoint and start fresh without losing history |
| No WhatsApp context persistence | Multi-turn conversations lose prior turns | Store last N history messages in a session file keyed to the WhatsApp chat ID |
| Workspace-only shell | Cannot run commands outside workspace | By design — prevents accidental system modifications |

---


## 16. Success Criteria

The project is complete when:

1. `uv run main.py` loads config, validates the configured model endpoint (default `http://127.0.0.1:8080/v1/chat/completions`), and shows the TUI in under 10 seconds when endpoint is reachable
2. The agent writes a Python file to the workspace when asked
3. The agent recalls stored preferences from a previous session via memory search
4. All skills and capability actions work as described in this spec
5. Shell commands prompt for confirmation and run correctly inside the workspace
6. `/memory stats` and `/workspace tree` work from the TUI
7. `/branch`, `/unbranch`, `/switch`, and `/tree` correctly navigate the conversation tree
8. The thinking toggle (`/think`) changes the `enable_thinking` parameter on subsequent requests and the interface reflects the current state
9. Interrupting generation with ESC (two-press confirm) produces a `✗` turn in the tree with the partial reply preserved
10. The entire codebase is under 5,000 lines of Python
11. Stopping and restarting the process does not lose any memories
12. Exiting the TUI does not modify external model-server process state
---

## 17. Normative Contracts and Ops

This section is normative. If any earlier section conflicts with these rules, this section takes precedence.

### 17.1 Skill and Tool-Call Execution Contract

Alphanus uses a hybrid model:

1. Skills select policy/instructions/capabilities for the turn.
2. The model invokes executable actions through native OpenAI-compatible `tool_calls`.
3. Alphanus executes only calls declared by active skills' `tools.py`.
4. Capability matching is enforced fail-closed when `skills.strict_capability_policy=true` (default personal mode is relaxed).

Required loop behavior:

1. Send request with `tools` derived from selected skills' `tools.py` declarations.
2. Parse streaming chunks and assemble fragmented `tool_calls` by `index`.
3. When `finish_reason == "tool_calls"`, execute calls in order.
4. Append assistant `tool_calls` message + `role:"tool"` results to history.
5. Continue until `finish_reason == "stop"`.

Safety invariant: if skills permit an action but no adapter exists, fail closed with structured error.

### 17.2 Versioned Schemas (Required)

All persisted or exchanged internal documents must carry `schema_version`:

- Conversation tree JSON: `schema_version` (e.g., `1.0.0`)
- Skill manifest TOML: `schema_version` (e.g., `1.0.0`)
- Global config JSON: `schema_version` (e.g., `1.0.0`)
- WhatsApp session JSON: `schema_version` (e.g., `1.0.0`)

Upgrade policy:

- Backward-compatible change: minor bump.
- Breaking change: major bump + migration function.
- Unknown major version: reject with explicit error message.

### 17.3 Endpoint, Auth, and TLS Policy

Config keys:

- `agent.model_endpoint`
- `agent.models_endpoint`
- `agent.auth_header` (optional, e.g., `Authorization: Bearer ...`)
- `agent.tls_verify` (default: true)
- `agent.ca_bundle_path` (optional)

Rules:

- Never log auth header values.
- If `tls_verify=false`, print startup warning.
- Endpoint and models URL must share host unless `allow_cross_host_endpoints=true`.

### 17.4 Retry and Timeout Matrix (Single Source of Truth)

- Connect timeout: 10s
- Read timeout: 180s
- Readiness timeout: 30s
- Readiness poll interval: 0.5s
- Per-turn retries: max 1
- Backoff: fixed 500ms

Retryable failures:

- Network reset/timeout
- HTTP 429
- HTTP 500/502/503/504

Non-retryable failures:

- HTTP 400/401/403/404/422
- JSON schema/validation failures

### 17.5 Capability Adapter I/O Contracts

All adapters must return a normalized envelope:

```json
{
  "ok": true,
  "data": {},
  "error": null,
  "meta": {"duration_ms": 12}
}
```

On failure:

```json
{
  "ok": false,
  "data": null,
  "error": {"code": "E_POLICY", "message": "..."},
  "meta": {"duration_ms": 4}
}
```

Minimum error codes:

- `E_POLICY`
- `E_VALIDATION`
- `E_TIMEOUT`
- `E_IO`
- `E_NOT_FOUND`
- `E_UNSUPPORTED`

### 17.6 Security and Sanitization Requirements

Required controls:

- Workspace write operations must resolve and enforce root containment after symlink resolution.
- Shell commands must run with `shell_require_confirmation=true` and workspace `cwd`.
- Blocklist must include dangerous shell patterns and secrets paths.
- No adapter may expose raw traceback to model output in non-debug mode.

Required tests:

- Path traversal and symlink escape tests.
- Command injection/shell metacharacter policy tests.
- Secret-file access denial tests.

### 17.7 WhatsApp Message and State Model

Message identity:

- `provider_message_id` (dedupe key)
- `chat_id`
- `timestamp_utc`

Processing guarantees:

- Idempotent processing by `provider_message_id`.
- Per-chat in-order processing.
- Ignore duplicates with same message id.

Branch label format (fixed):

- `wa:<chat_id>:<YYYYMMDD>`

Session schema keys:

- `schema_version`
- `chat_id`
- `last_message_id`
- `history`
- `updated_at_utc`

### 17.8 Migration and Compatibility Policy

Each breaking schema change must include:

- `migrations/<from>_to_<to>.py`
- unit tests for migration correctness
- rollback plan for failed migration

Startup behavior:

1. Detect current schema version.
2. Run sequential migrations to target.
3. Atomic write migrated file.
4. Keep one backup copy with previous version suffix.
