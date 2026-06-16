from __future__ import annotations

from skills.runtime import ToolExecutionEnv

_DEFAULT_TIMEOUT_S = 600
_MAX_TIMEOUT_S = 7200

TOOL_SPECS = {
    "shell_command": {
        "capability": "run_shell_command",
        "mutates": True,
        "actions": ["run", "check", "read", "list"],
        "description": (
            "Run a shell command in the workspace using the user's shell. Shell syntax such as &&, ;, pipes, redirects, "
            "environment assignments, and globbing is allowed. The tool itself asks for confirmation when required, "
            "so do not ask separately in assistant text. Use timeout_s for long builds, installs, updates, or tests. "
            "Do not run malicious or unrelated destructive commands."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_s": {
                    "type": "integer",
                    "description": "Maximum runtime in seconds. Defaults to 600; capped at 7200.",
                },
            },
            "required": ["command"],
        },
    }
}


def _timeout_s(args: dict[str, object]) -> int:
    raw = args.get("timeout_s")
    if raw is None:
        return _DEFAULT_TIMEOUT_S
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError("timeout_s must be an integer") from None
    if value <= 0:
        raise ValueError("timeout_s must be greater than 0")
    return min(_MAX_TIMEOUT_S, value)


def execute(tool_name: str, args: dict[str, object], env: ToolExecutionEnv):
    if tool_name != "shell_command":
        raise ValueError(f"Unsupported tool: {tool_name}")

    command = str(args["command"]).strip()
    timeout_s = _timeout_s(args)
    caps = env.config.get("capabilities", {})
    dangerously_skip_permissions = bool(caps.get("dangerously_skip_permissions", False))
    shell_require_confirmation = bool(caps.get("shell_require_confirmation", True))

    if shell_require_confirmation and not dangerously_skip_permissions:
        if not env.confirm_shell:
            raise PermissionError("Shell confirmation callback is required")
        if not env.confirm_shell(command):
            raise PermissionError("Shell command rejected by user")

    return env.workspace.run_shell_command(command, timeout_s=timeout_s)
