# Alphanus Technical Specification

Project: Personal local-first coding assistant
Updated: March 10, 2026

---

## 1. Scope

Alphanus is a terminal-first assistant that connects to an external OpenAI-compatible model endpoint (typically `llama-server`) and executes local tools through a skill runtime.

In scope:
- Textual TUI
- Branchable conversation tree
- Skill-based tool exposure and execution
- Workspace-safe file/shell operations
- Persistent memory (vector + lexical fallback)

Out of scope:
- Built-in model hosting
- Web UI
- WhatsApp integration
- Multi-user tenancy

---

## 2. High-Level Architecture

```text
Textual TUI (tui/interface.py)
  -> Agent Loop (agent/core.py)
      -> Context Pruner (agent/context.py)
      -> Prompt Builder (agent/prompts.py)
      -> Skill Runtime (core/skills.py)
          -> Skill Parser (core/skill_parser.py)
          -> Command-backed scripts (skills/*/scripts/*.py)
          -> Optional legacy tools.py adapters
      -> Workspace Manager (core/workspace.py)
      -> Memory Store (core/memory.py)
```

Core runtime path:
1. Build system prompt + selected skill guidance.
2. Send streaming request to `/v1/chat/completions`.
3. Parse deltas (`reasoning_content`, `content`, `tool_calls`).
4. Execute tool calls through `SkillRuntime`.
5. Append tool results to loop history.
6. Finalize assistant response and persist turn into conversation tree.

---

## 3. Model Endpoint Contract

Configured in `config/global_config.json`:
- `agent.model_endpoint`
- `agent.models_endpoint`

Defaults:
- `http://127.0.0.1:8080/v1/chat/completions`
- `http://127.0.0.1:8080/v1/models`

Readiness behavior:
- Startup prints: `waiting for endpoint <models_endpoint> handshake...`
- Polls readiness before first turn.
- If unavailable, app still opens with warning and retries occur at turn-time.

Request behavior:
- OpenAI-compatible chat payload
- `stream: true`
- `tools` included when active skills expose callable tools
- `max_tokens` omitted when config value is `null`

---

## 4. Prompting

System prompt source: `agent/prompts.py`

Current behavior:
- Includes the current local date each run.
- Includes resolved workspace path.
- Encodes safety constraints and style expectations.

---

## 5. Skill System

### 5.1 Format and Loading

Canonical skill manifest: `skills/<skill-id>/SKILL.md`

Frontmatter parser:
- Implemented in `core/skill_parser.py`
- Uses `PyYAML` (`yaml.safe_load`)
- No custom fallback parser path

Required fields:
- `name` (must match folder name)
- `description`

Optional fields used by runtime:
- `metadata.version` (semver)
- `metadata.tags`, `metadata.categories`
- `metadata.triggers.keywords`, `metadata.triggers.file_ext`
- `allowed-tools`, `required-tools`
- `metadata.tools.definitions` (preferred command definitions)
- `tools.definitions` (fallback location)
- `metadata.tools.disable-model-invocation`

### 5.2 Execution Modes

1. Preferred: command-backed definitions in SKILL frontmatter
- Each tool definition includes `command`, `parameters`, and optional `confirm-arg`.
- Commands run in skill directory.
- Tool args are available via stdin and env vars.
- Final stdout line must be JSON.

2. Legacy fallback: `tools.py`
- `TOOL_SPECS`
- `execute(tool_name, args, env)`

Both can coexist; duplicates are ignored.

### 5.3 Hooks

Optional `hooks.py`:
- `pre_prompt`
- `pre_action`
- `post_response`

Hook failures are non-fatal.

### 5.4 Skill Selection

Configured by:
- `skills.selection_mode`
- `skills.max_active_skills`
- `skills.strict_capability_policy`

Current default behavior is practical/personal mode (`all_enabled`, capped).

---

## 6. Tool Result and History Handling

Agent loop uses OpenAI tool-call semantics:
- model emits `tool_calls`
- runtime executes calls
- tool responses are appended as `role: tool`

Relevant controls:
- `agent.max_action_depth`
- `agent.max_tool_result_chars`
- `agent.max_reasoning_chars`
- `agent.compact_tool_results_in_history`
- `agent.compact_tool_result_tools`

Malformed tool args support:
- `_recover_tool_args` attempts to recover arguments from `_raw` payloads when model emits broken JSON.

---

## 7. Conversation Tree

Implementation: `core/conv_tree.py`

Features:
- Tree of turns (branch/switch/unbranch)
- Save/load JSON snapshots
- Skill exchanges stored per turn for replay/history

Memory-oriented updates:
- Inactive branch compaction policy is configurable and enabled by default.
- Large inactive assistant/tool payloads are truncated with a compact marker.
- Active path remains unmodified.

Tradeoff:
- Inactive-branch compaction is lossy. If you return to an old compacted branch, large payloads remain compacted.

---

## 8. TUI

Implementation: `tui/interface.py`

Streaming behavior:
- Separate reasoning and content rendering
- Live tool-call previews for file tools
- ESC interrupt flow

Refactors:
- Markdown helpers extracted to `tui/markdown_utils.py`
- Live preview state/logic extracted to `tui/live_tool_preview.py`

RAM-oriented control:
- `RichLog` now uses bounded retention via `tui.chat_log_max_lines`.

---

## 9. Memory System

Implementation: `core/memory.py`

Backends:
- `hash` (default, low RAM)
- `transformer`
- `auto`

Current default config favors RAM efficiency (`hash`, lazy encoder load).

Recent reliability change:
- Memory recall flow includes lexical fallback behavior (skill-script side) when semantic hit quality is poor.

---

## 10. Workspace and Shell Safety

Workspace manager (`core/workspace.py`) enforces path policy.

Shell behavior:
- `shell_command` requires confirmation by default.
- `--dangerously-skip-permissions` disables confirmations (unsafe mode).

---

## 11. Configuration Snapshot

Main config file: `config/global_config.json`

Key groups:
- `agent`
- `context`
- `workspace`
- `memory`
- `capabilities`
- `skills`
- `tui`

New/important `tui` keys:

```json
"tui": {
  "chat_log_max_lines": 5000,
  "tree_compaction": {
    "enabled": true,
    "inactive_assistant_char_limit": 12000,
    "inactive_tool_argument_char_limit": 5000,
    "inactive_tool_content_char_limit": 8000
  }
}
```

---

## 12. Repository Structure (Current)

```text
agent/
  core.py
  context.py
  prompts.py
core/
  conv_tree.py
  memory.py
  workspace.py
  skills.py
  skill_parser.py
skills/
  workspace-ops/
  shell-ops/
  memory-rag/
  utilities/
tui/
  interface.py
  markdown_utils.py
  live_tool_preview.py
tests/
  test_agent_loop.py
  test_conv_tree.py
  test_context_config.py
  test_memory.py
  test_skills.py
```

---

## 13. Dependencies

From `pyproject.toml`:
- `textual`
- `sentence-transformers`
- `numpy`
- `python-dotenv`
- `PyYAML`
- dev: `pytest`, `pytest-mock`

---

## 14. Testing Status

Current suite covers:
- agent loop and tool-call paths
- context pruning invariants
- conversation tree behavior
- memory behavior and recall fallback
- skills parsing/runtime behavior

Recent additions include:
- memory-control config assertions
- inactive branch compaction behavior checks

---

## 15. Known Tradeoffs

1. Inactive branch compaction saves RAM but loses full payload fidelity on old branches.
2. Command execution relies on scripts returning valid final-line JSON.
3. `shell=True` command mode is intentionally constrained by policy and confirmation, not a full shell sandbox.
4. Large long-running sessions still grow on disk (saved trees/memory), even with bounded in-process UI log size.

---

## 16. Operational Checklist

1. Start endpoint.
2. Launch `uv run main.py`.
3. Verify handshake message and readiness status.
4. Verify tool calls and branching in TUI.
5. Tune `tui` memory settings if session size is large.

