---
name: shell-ops
description: Run workspace shell commands with confirmation.
allowed-tools: shell_command
metadata:
  version: "1.0.0"
  categories:
    - devops
  tags:
    - run
    - shell
    - command
    - terminal
    - ls
---
Use `shell_command` only when shell output is the best way to complete the task.

Rules:
- Every command requires explicit user confirmation.
- Commands run in workspace `cwd` only.
- Do not use `cd` to change directories before a later command. Each `shell_command` call runs independently and does not persist shell state.
- If a command must target another directory, use a single command with an absolute path or the tool's own path flag instead of changing directories first.
- For `uv`, prefer `uv --directory /absolute/path ...` or `uv ... --project /absolute/path ...` when operating outside the workspace root.
- Do not use unsafe or destructive shell commands.
