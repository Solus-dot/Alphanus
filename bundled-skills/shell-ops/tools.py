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
            "Run a shell command using the user's shell. Commands run in the project by default; pass cwd only when "
            "the user explicitly named another directory. Shell syntax such as &&, ;, pipes, redirects, environment "
            "assignments, and globbing is allowed. The tool itself asks for confirmation when required, so do not ask "
            "separately in assistant text. On macOS and Linux, invoke Python scripts with python3 rather than python. "
            "Use timeout_s for long builds, installs, updates, or tests. Do not run "
            "malicious or unrelated destructive commands."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory. Use an explicit absolute path when the user asked to run the command outside the project.",
                },
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
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        raise ValueError("timeout_s must be an integer")
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
    cwd = str(args["cwd"]).strip() if args.get("cwd") is not None else None
    timeout_s = _timeout_s(args)
    approved = False
    external_paths = (
        ()
        if env.project.permission_mode == "danger-full-access"
        else env.project.shell_command_external_paths(command)
    )
    needs_approval = (
        env.project.shell_command_requires_approval(command)
        or env.project.shell_cwd_requires_approval(cwd)
        or bool(external_paths)
    )
    if needs_approval and env.project.permission_mode != "danger-full-access":
        if not env.request_approval:
            raise PermissionError("Approval callback is required")
        approved = bool(
            env.request_approval(
                {
                    "kind": "shell_command",
                    "command": command,
                    "cwd": str(cwd or env.project.project_root),
                    "paths": [str(path) for path in external_paths],
                    "reason": "Command crosses the project-write approval boundary.",
                }
            )
        )
        if not approved:
            raise PermissionError("Shell command rejected by user")

    return env.project.run_shell_command(
        command,
        timeout_s=timeout_s,
        cwd=cwd,
        allowed_cwd_roots=[cwd] if cwd else None,
        approved=approved,
    )
