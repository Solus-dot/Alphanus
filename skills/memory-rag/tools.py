from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

_FACT_IS_PATTERN = re.compile(
    r"\b(?:my|user(?:'s)?)\s+([a-z][a-z0-9 _-]{0,40})\s+is\s+([^.!?\n]{1,120})",
    re.IGNORECASE,
)
_FACT_I_AM_PATTERN = re.compile(r"\bi am\s+([^.!?\n]{1,80})", re.IGNORECASE)


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.strip().split()).lower()


def _normalize_fact_key(raw: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 _-]", "", raw.lower())
    return "_".join(cleaned.strip().split())


def _extract_fact(text: str) -> Optional[Tuple[str, str]]:
    match = _FACT_IS_PATTERN.search(text)
    if match:
        attr = _normalize_fact_key(match.group(1))
        value = " ".join(match.group(2).strip().split())
        if attr and value:
            return (f"user.{attr}", value)

    match = _FACT_I_AM_PATTERN.search(text)
    if match:
        value = " ".join(match.group(1).strip().split())
        if value:
            return ("user.identity", value)
    return None


def _fact_from_item(item) -> Optional[Tuple[str, str]]:
    metadata = item.metadata or {}
    key = metadata.get("fact_key")
    value = metadata.get("fact_value")
    if isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip():
        return (key.strip(), value.strip())
    return None


def _find_conflicting_fact_ids(memory, fact_key: str, fact_value: str) -> list[int]:
    target = _normalize_whitespace(fact_value)
    forgotten_ids: list[int] = []
    for item in list(memory.memories):
        fact = _fact_from_item(item)
        if not fact:
            continue
        key, value = fact
        if key == fact_key and _normalize_whitespace(value) != target:
            forgotten_ids.append(int(item.id))
    return forgotten_ids


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


def _lexical_fallback(memory, query: str, top_k: int, memory_type):
    query_tokens = {tok for tok in re.findall(r"[a-z0-9]+", query.lower()) if len(tok) > 1}
    if not query_tokens:
        return []

    scored = []
    for item in list(memory.memories):
        if memory_type and item.type != memory_type:
            continue
        text_tokens = set(re.findall(r"[a-z0-9]+", str(item.text).lower()))
        overlap = len(query_tokens & text_tokens)
        if overlap <= 0:
            continue
        score = overlap / max(1, len(query_tokens))
        scored.append((score, item))

    scored.sort(key=lambda pair: (pair[0], pair[1].timestamp), reverse=True)
    out = []
    for score, item in scored[: max(1, top_k)]:
        record = memory._to_public(item)  # noqa: SLF001
        record["score"] = round(float(score), 4)
        out.append(record)
    return out


def _store_memory(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    memory = env.memory
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
        ids_to_forget.update(_find_conflicting_fact_ids(memory, fact[0], fact[1]))
    if replace_existing:
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
        if fact:
            meta["auto_resolution"] = fact[0]
        elif args.get("replace_query"):
            meta["auto_resolution"] = "replace_query"

    return {"ok": True, "data": item, "error": None, "meta": meta}


def _recall_memory(args: Dict[str, Any], env: ToolExecutionEnv) -> Dict[str, Any]:
    top_k = int(args.get("top_k", 5))
    min_score = args.get("min_score")
    threshold = 0.18 if min_score is None else float(min_score)
    memory_type = args.get("memory_type")
    hits = env.memory.search(
        query=args["query"],
        top_k=top_k,
        memory_type=memory_type,
        min_score=threshold,
    )
    if not hits:
        hits = _lexical_fallback(env.memory, str(args["query"]), top_k=top_k, memory_type=memory_type)
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
    path = str(args.get("filepath") or (workspace_root / "memories_export.txt"))
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
