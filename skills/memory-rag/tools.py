from __future__ import annotations

from typing import Any, Dict


TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "store_memory": {
        "capability": "memory_store",
        "description": "Persist a memory item.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "memory_type": {"type": "string"},
                "importance": {"type": "number"},
            },
            "required": ["text"],
        },
    },
    "recall_memory": {
        "capability": "memory_recall",
        "description": "Semantic search over memories.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
                "memory_type": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    "list_memories": {
        "capability": "memory_list",
        "description": "List recent memories.",
        "parameters": {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": [],
        },
    },
    "forget_memory": {
        "capability": "memory_forget",
        "description": "Delete memory by id.",
        "parameters": {
            "type": "object",
            "properties": {"memory_id": {"type": "integer"}},
            "required": ["memory_id"],
        },
    },
    "get_memory_stats": {
        "capability": "memory_stats",
        "description": "Get memory statistics.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "export_memories": {
        "capability": "memory_export",
        "description": "Export memories to a text file.",
        "parameters": {
            "type": "object",
            "properties": {"filepath": {"type": "string"}},
            "required": [],
        },
    },
}


def execute(tool_name: str, args: Dict[str, Any], env) -> Dict[str, Any]:
    if tool_name == "store_memory":
        item = env.memory.add_memory(
            text=args["text"],
            memory_type=args.get("memory_type", "conversation"),
            importance=args.get("importance"),
        )
        return {"ok": True, "data": item, "error": None, "meta": {}}

    if tool_name == "recall_memory":
        hits = env.memory.search(
            query=args["query"],
            top_k=int(args.get("top_k", 5)),
            memory_type=args.get("memory_type"),
        )
        return {"ok": True, "data": {"hits": hits}, "error": None, "meta": {}}

    if tool_name == "list_memories":
        rec = env.memory.list_recent(count=int(args.get("count", 5)))
        return {"ok": True, "data": {"memories": rec}, "error": None, "meta": {}}

    if tool_name == "forget_memory":
        deleted = env.memory.forget(int(args["memory_id"]))
        if not deleted:
            return {
                "ok": False,
                "data": None,
                "error": {"code": "E_NOT_FOUND", "message": "Memory id not found"},
                "meta": {},
            }
        return {"ok": True, "data": {"deleted": True}, "error": None, "meta": {}}

    if tool_name == "get_memory_stats":
        return {"ok": True, "data": env.memory.stats(), "error": None, "meta": {}}

    if tool_name == "export_memories":
        path = args.get("filepath") or str(env.workspace.workspace_root / "memories_export.txt")
        out = env.memory.export_txt(path)
        return {"ok": True, "data": {"filepath": out}, "error": None, "meta": {}}

    return {
        "ok": False,
        "data": None,
        "error": {"code": "E_UNSUPPORTED", "message": f"Unsupported tool: {tool_name}"},
        "meta": {},
    }
