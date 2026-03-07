from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


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


_FACT_IS_PATTERN = re.compile(
    r"\b(?:my|user(?:'s)?)\s+([a-z][a-z0-9 _-]{0,40})\s+is\s+([^.!?\n]{1,120})",
    re.IGNORECASE,
)
_FACT_I_AM_PATTERN = re.compile(r"\bi am\s+([^.!?\n]{1,80})", re.IGNORECASE)


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.strip().split()).lower()


def _normalize_fact_key(raw: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 _-]", "", raw.lower())
    cleaned = "_".join(cleaned.strip().split())
    return cleaned


def _extract_fact(text: str) -> Optional[Tuple[str, str]]:
    m = _FACT_IS_PATTERN.search(text)
    if m:
        attr = _normalize_fact_key(m.group(1))
        value = " ".join(m.group(2).strip().split())
        if attr and value:
            return (f"user.{attr}", value)
    m = _FACT_I_AM_PATTERN.search(text)
    if m:
        value = " ".join(m.group(1).strip().split())
        if value:
            return ("user.identity", value)
    return None


def _fact_from_item(item) -> Optional[Tuple[str, str]]:
    md = item.metadata or {}
    key = md.get("fact_key")
    value = md.get("fact_value")
    if isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip():
        return (key.strip(), value.strip())
    return _extract_fact(str(item.text))


def _find_conflicting_fact_ids(env, fact_key: str, fact_value: str) -> list[int]:
    forgotten_ids: list[int] = []
    target = _normalize_whitespace(fact_value)
    # Use a snapshot because we may mutate memory while iterating.
    for item in list(env.memory.memories):
        fact = _fact_from_item(item)
        if not fact:
            continue
        key, value = fact
        if key != fact_key:
            continue
        if _normalize_whitespace(value) != target:
            forgotten_ids.append(int(item.id))
    return forgotten_ids


def _ids_from_replace_query(env, args: Dict[str, Any], text: str) -> list[int]:
    query = str(args.get("replace_query") or "").strip()
    if not query:
        return []
    top_k = int(args.get("replace_top_k", 5))
    min_score = float(args.get("replace_min_score", 0.72))
    hits = env.memory.search(
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
        if memory_id <= 0:
            continue
        if _normalize_whitespace(hit_text) == text_norm:
            continue
        out.append(memory_id)
    return out


def execute(tool_name: str, args: Dict[str, Any], env) -> Dict[str, Any]:
    if tool_name == "store_memory":
        text = str(args["text"])
        memory_type = args.get("memory_type", "conversation")
        metadata: Dict[str, Any] = dict(args.get("metadata") or {})
        forgotten_ids: list[int] = []
        replace_existing = bool(args.get("replace_existing", True))

        fact = _extract_fact(text)
        if fact:
            metadata["fact_key"] = fact[0]
            metadata["fact_value"] = fact[1]

        ids_to_forget = set()
        if replace_existing and fact:
            ids_to_forget.update(_find_conflicting_fact_ids(env, fact[0], fact[1]))
        if replace_existing:
            ids_to_forget.update(_ids_from_replace_query(env, args, text))
        for memory_id in args.get("replace_ids") or []:
            try:
                ids_to_forget.add(int(memory_id))
            except Exception:
                continue

        for memory_id in sorted(ids_to_forget):
            if env.memory.forget(memory_id):
                forgotten_ids.append(memory_id)

        item = env.memory.add_memory(
            text=text,
            memory_type=memory_type,
            metadata=metadata or None,
            importance=args.get("importance"),
        )
        meta: Dict[str, Any] = {}
        if forgotten_ids:
            meta["forgotten_ids"] = forgotten_ids
            if fact:
                meta["auto_resolution"] = fact[0]
            elif args.get("replace_query"):
                meta["auto_resolution"] = "replace_query"
        return {"ok": True, "data": item, "error": None, "meta": meta}

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
