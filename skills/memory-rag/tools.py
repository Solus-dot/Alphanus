from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from core.skills import ToolExecutionEnv

TOOL_SPECS = {
    "store_memory": {
        "capability": "memory_store",
        "description": "Persist a memory item.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "memory_type": {"type": "string"},
                "importance": {"type": "number"},
                "replace_existing": {"type": "boolean"},
                "replace_query": {"type": "string"},
                "replace_top_k": {"type": "integer"},
                "replace_min_score": {"type": "number"},
                "replace_ids": {"type": "array", "items": {"type": "integer"}},
                "metadata": {"type": "object"},
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
                "min_score": {"type": "number"},
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
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
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


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.strip().split()).lower()


def _rank_hits(hits: list[Dict[str, Any]], top_k: int) -> list[Dict[str, Any]]:
    ranked: dict[int, Dict[str, Any]] = {}
    for hit in hits:
        try:
            memory_id = int(hit.get("id", 0))
        except Exception:
            continue
        if memory_id <= 0:
            continue
        score = float(hit.get("score", 0.0))
        timestamp = float(hit.get("timestamp", 0.0))
        current = ranked.get(memory_id)
        if current is None:
            ranked[memory_id] = hit
            continue
        current_score = float(current.get("score", 0.0))
        current_timestamp = float(current.get("timestamp", 0.0))
        if (score, timestamp) > (current_score, current_timestamp):
            ranked[memory_id] = hit
    ordered = sorted(
        ranked.values(),
        key=lambda item: (float(item.get("score", 0.0)), float(item.get("timestamp", 0.0))),
        reverse=True,
    )
    return ordered[: max(1, top_k)]


def _find_exact_text_ids(memory, text: str, memory_type: str | None) -> list[int]:
    target = _normalize_whitespace(text)
    if not target:
        return []
    out: list[int] = []
    for item in list(memory.memories):
        if memory_type and item.type != memory_type:
            continue
        if _normalize_whitespace(str(item.text)) == target:
            out.append(int(item.id))
    return out


def _ids_from_replace_query(memory, args: Dict[str, Any], text: str) -> list[int]:
    query = str(args.get("replace_query") or "").strip()
    if not query:
        return []
    top_k = int(args.get("replace_top_k", 5))
    min_score = float(args.get("replace_min_score", 0.72))
    hits = memory.search(
        query=query,
        top_k=max(1, top_k),
        memory_type=args.get("memory_type"),
        min_score=min_score,
    )
    text_norm = _normalize_whitespace(text)
    out: list[int] = []
    for hit in hits:
        memory_id = int(hit.get("id", 0))
        hit_text = str(hit.get("text", ""))
        if memory_id > 0 and _normalize_whitespace(hit_text) != text_norm:
            out.append(memory_id)
    return out


def _store_memory(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    memory = env.memory
    text = " ".join(str(args["text"]).split()).strip()
    if not text:
        raise ValueError("Memory text must not be empty")
    memory_type = str(args.get("memory_type") or "conversation").strip() or "conversation"
    metadata: Dict[str, Any] = dict(args.get("metadata") or {})
    forgotten_ids: list[int] = []
    replace_existing = bool(args.get("replace_existing", True))

    ids_to_forget = set()
    if replace_existing:
        ids_to_forget.update(_find_exact_text_ids(memory, text, memory_type))
        ids_to_forget.update(_ids_from_replace_query(memory, args, text))
    for memory_id in args.get("replace_ids") or []:
        try:
            ids_to_forget.add(int(memory_id))
        except Exception:
            continue

    for memory_id in sorted(ids_to_forget):
        if memory.forget(memory_id):
            forgotten_ids.append(memory_id)

    item = memory.add_memory(
        text=text,
        memory_type=memory_type,
        metadata=metadata or None,
        importance=args.get("importance"),
    )
    memory.flush()

    meta: Dict[str, Any] = {}
    if forgotten_ids:
        meta["forgotten_ids"] = forgotten_ids
        if args.get("replace_query"):
            meta["auto_resolution"] = "replace_query"
        else:
            meta["auto_resolution"] = "exact_text"

    return {"ok": True, "data": item, "error": None, "meta": meta}


def _recall_memory(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    top_k = int(args.get("top_k", 5))
    min_score = args.get("min_score")
    threshold = 0.18 if min_score is None else float(min_score)
    memory_type = str(args.get("memory_type") or "").strip() or None
    query = " ".join(str(args["query"]).split()).strip()
    if not query:
        raise ValueError("Recall query must not be empty")

    hits = _rank_hits(
        env.memory.search(
            query=query,
            top_k=top_k,
            memory_type=memory_type,
            min_score=threshold,
        ),
        top_k,
    )
    return {"hits": hits}


def _list_memories(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    return {"memories": env.memory.list_recent(count=int(args.get("count", 5)))}


def _forget_memory(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    deleted = env.memory.forget(int(args["memory_id"]))
    if not deleted:
        raise FileNotFoundError("Memory id not found")
    env.memory.flush()
    return {"deleted": True}


def _get_memory_stats(_args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    return env.memory.stats()


def _export_memories(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    workspace_root = Path(env.workspace.workspace_root)
    requested = str(args.get("filepath") or "").strip()
    if requested:
        path = str(env.workspace._resolve_write_path(requested))  # noqa: SLF001
    else:
        path = str(workspace_root / "memories_export.txt")
    out = env.memory.export_txt(path)
    return {"filepath": out}


def execute(tool_name: str, args: Dict[str, Any], env: ToolExecutionEnv):
    if tool_name == "store_memory":
        return _store_memory(args, env)
    if tool_name == "recall_memory":
        return _recall_memory(args, env)
    if tool_name == "list_memories":
        return _list_memories(args, env)
    if tool_name == "forget_memory":
        return _forget_memory(args, env)
    if tool_name == "get_memory_stats":
        return _get_memory_stats(args, env)
    if tool_name == "export_memories":
        return _export_memories(args, env)
    raise ValueError(f"Unsupported tool: {tool_name}")
