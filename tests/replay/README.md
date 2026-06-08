# Replay Regression Fixtures

Replay tests cover product behavior that previously tended to regress despite
unit tests passing. Keep fixtures small and focused on a single user-visible
outcome.

Each fixture may define:

- `user_input`: the prompt passed to `Agent.run_turn`.
- `skills`: bundled skill ids to expose for the turn.
- `classification`: optional routing flags used by the scripted classifier.
- `initial_files`: workspace files to create before the turn.
- `model_passes`: deterministic model pass outputs. Use `tool_calls` for tool
  phases and `content` with `finish_reason: "stop"` for final answers.
- `cancel_on_event_type`: set a stop event when the named agent event is
  emitted.
- `expect`: final status, content substrings, tool sequence, workspace file
  contents, tool result checks, event types, and optional skill-exchange tool
  messages.

Prefer replay fixtures for regressions involving turn flow, tool execution,
policy blocking, cancellation, and final answers. Keep provider/SSE parsing,
serialization, config normalization, and pure safety-policy contracts in their
dedicated unit tests.
