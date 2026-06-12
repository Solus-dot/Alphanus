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
- Do not run malicious, credential-harvesting, privacy-invasive, system-destructive, or unrelated destructive shell commands.
- Commands run in workspace `cwd` only.
- Each `shell_command` call runs independently and does not persist shell state after the command exits.
- Shell syntax is available. You may use normal shell chaining and composition such as `&&`, `||`, `;`, pipes, redirects, environment assignments, globbing, and shell wrappers when that is the clearest way to perform the task.
- Keep commands transparent and reviewable. Prefer straightforward commands over dense one-liners when separate calls would be clearer.
- A timed-out command means the process did not finish before the execution timeout. It does not mean the process was merely quiet.
- If a command must target another directory, use a single command with an absolute path or the tool's own path flag instead of changing directories first.
- For `uv`, prefer `uv --directory /absolute/path ...` or `uv ... --project /absolute/path ...` when operating outside the workspace root.
