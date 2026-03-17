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
- Do not use unsafe or destructive shell commands.
