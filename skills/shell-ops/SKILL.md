---
name: shell-ops
description: Run workspace shell commands with confirmation.
version: 1.0.0
categories:
  - devops
tags:
  - run
  - shell
  - command
  - terminal
  - ls
  - git
tools:
  allowed-tools:
    - shell_command
  definitions:
    - name: shell_command
      capability: run_shell_command
      description: Run a shell command in workspace with explicit confirmation.
      command: python3 scripts/ops.py shell_command
      timeout-s: 30
      confirm-arg: command
      parameters:
        type: object
        properties:
          command:
            type: string
        required:
          - command
---
Use `shell_command` only when shell output is the best way to complete the task.

Rules:
- Every command requires explicit user confirmation.
- Commands run in workspace `cwd` only.
- Do not use unsafe or destructive shell commands.
