from __future__ import annotations

from typing import Any, Dict


TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "shell_command": {
        "capability": "run_shell_command",
        "description": "Run a shell command in workspace with explicit confirmation.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
}


def execute(tool_name: str, args: Dict[str, Any], env) -> Dict[str, Any]:
    if tool_name != "shell_command":
        return {
            "ok": False,
            "data": None,
            "error": {"code": "E_UNSUPPORTED", "message": f"Unsupported tool: {tool_name}"},
            "meta": {},
        }

    command = str(args["command"])
    if not env.confirm_shell:
        return {
            "ok": False,
            "data": None,
            "error": {"code": "E_POLICY", "message": "Shell confirmation callback is required"},
            "meta": {},
        }
    if not env.confirm_shell(command):
        return {
            "ok": False,
            "data": None,
            "error": {"code": "E_POLICY", "message": "Shell command rejected by user"},
            "meta": {},
        }
    return env.workspace.run_shell_command(command)
