---
name: shell-ops
description: Run terminal/shell commands. The tool itself prompts for approval when required.
allowed-tools: shell_command
metadata:
  version: "1.0.0"
  tags:
    - run
    - shell
    - command
    - terminal
    - ls
---
Use `shell_command` only when shell output is the best way to complete the task.

Rules:
- The `shell_command` tool itself asks the user for confirmation when confirmation is enabled.
- Do not ask for duplicate confirmation in normal assistant text before calling `shell_command`.
- Commands run in workspace `cwd` only.
- Do not use `cd` to change directories before a later command. Each `shell_command` call runs independently and does not persist shell state.
- Do not chain commands with shell metacharacters such as `&&`, `||`, `;`, `|`, newlines, or carriage returns; Alphanus rejects them before execution.
- If multiple commands are needed, run them as separate `shell_command` calls.
- Commands are executed as a single argv process, not through an interactive shell. Shell aliases, functions, shell wrappers such as `bash -lc`, pipes, redirection, and persistent environment changes are not available.
- A timed-out command means the process did not finish before the execution timeout. It does not mean the process was merely quiet.
- If a command must target another directory, use a single command with an absolute path or the tool's own path flag instead of changing directories first.
- For `uv`, prefer `uv --directory /absolute/path ...` or `uv ... --project /absolute/path ...` when operating outside the workspace root.
- Do not use unsafe or destructive shell commands.
