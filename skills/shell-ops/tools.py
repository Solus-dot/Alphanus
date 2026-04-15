from __future__ import annotations

from core.skills import ToolExecutionEnv

TOOL_SPECS = {
    "shell_command": {
        "capability": "run_shell_command",
        "description": "Run a shell command in workspace. The tool itself asks for confirmation when required, so do not ask separately in assistant text.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
}


def execute(tool_name: str, args: dict[str, object], env: ToolExecutionEnv):
    if tool_name != "shell_command":
        raise ValueError(f"Unsupported tool: {tool_name}")

    command = str(args["command"]).strip()
    caps = env.config.get("capabilities", {})
    dangerously_skip_permissions = bool(caps.get("dangerously_skip_permissions", False))
    shell_require_confirmation = bool(caps.get("shell_require_confirmation", True))

    if shell_require_confirmation and not dangerously_skip_permissions:
        if not env.confirm_shell:
            raise PermissionError("Shell confirmation callback is required")
        if not env.confirm_shell(command):
            raise PermissionError("Shell command rejected by user")

    return env.workspace.run_shell_command(command)
