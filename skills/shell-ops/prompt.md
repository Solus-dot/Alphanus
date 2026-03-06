Use `shell_command` only when shell output is the best way to complete the task.

Rules:
- Every command requires explicit user confirmation.
- Commands run in workspace `cwd` only.
- Do not use unsafe or destructive shell commands.
