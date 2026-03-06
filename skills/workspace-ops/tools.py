from __future__ import annotations

from typing import Any, Dict


TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "create_file": {
        "capability": "workspace_write",
        "description": "Create or overwrite a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["filepath", "content"],
        },
    },
    "edit_file": {
        "capability": "workspace_edit",
        "description": "Replace content of an existing workspace file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["filepath", "content"],
        },
    },
    "read_file": {
        "capability": "workspace_read",
        "description": "Read a file under home/workspace policy.",
        "parameters": {
            "type": "object",
            "properties": {"filepath": {"type": "string"}},
            "required": ["filepath"],
        },
    },
    "list_files": {
        "capability": "workspace_read",
        "description": "List files in a directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": [],
        },
    },
    "delete_file": {
        "capability": "workspace_delete",
        "description": "Delete a workspace file.",
        "parameters": {
            "type": "object",
            "properties": {"filepath": {"type": "string"}},
            "required": ["filepath"],
        },
    },
    "workspace_tree": {
        "capability": "workspace_tree",
        "description": "Render the workspace tree.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def execute(tool_name: str, args: Dict[str, Any], env) -> Dict[str, Any]:
    if tool_name == "create_file":
        path = env.workspace.create_file(args["filepath"], args["content"])
        return {"ok": True, "data": {"filepath": path}, "error": None, "meta": {}}

    if tool_name == "edit_file":
        path = env.workspace.edit_file(args["filepath"], args["content"])
        return {"ok": True, "data": {"filepath": path}, "error": None, "meta": {}}

    if tool_name == "read_file":
        text = env.workspace.read_file(args["filepath"])
        return {"ok": True, "data": {"content": text}, "error": None, "meta": {}}

    if tool_name == "list_files":
        names = env.workspace.list_files(args.get("path", "."))
        return {"ok": True, "data": {"files": names}, "error": None, "meta": {}}

    if tool_name == "delete_file":
        path = env.workspace.delete_file(args["filepath"])
        return {"ok": True, "data": {"filepath": path}, "error": None, "meta": {}}

    if tool_name == "workspace_tree":
        return {"ok": True, "data": {"tree": env.workspace.workspace_tree()}, "error": None, "meta": {}}

    return {
        "ok": False,
        "data": None,
        "error": {"code": "E_UNSUPPORTED", "message": f"Unsupported tool: {tool_name}"},
        "meta": {},
    }
