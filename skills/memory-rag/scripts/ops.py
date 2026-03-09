#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.memory import VectorMemory  # noqa: E402


def _ok(data: Any, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"ok": True, "data": data, "error": None, "meta": dict(meta or {})}


def _err(code: str, message: str) -> Dict[str, Any]:
    return {"ok": False, "data": None, "error": {"code": code, "message": message}, "meta": {}}


def _read_args() -> Dict[str, Any]:
    raw = os.getenv("ALPHANUS_TOOL_ARGS_JSON", "").strip()
    if not raw:
        raw = sys.stdin.read().strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Tool args must be a JSON object")
    return parsed


def _memory() -> VectorMemory:
    storage_path = os.getenv("ALPHANUS_MEMORY_PATH", "").strip()
    if not storage_path:
        raise ValueError("ALPHANUS_MEMORY_PATH is required")
    model_name = os.getenv("ALPHANUS_MEMORY_MODEL", "all-MiniLM-L6-v2")
    backend = os.getenv("ALPHANUS_MEMORY_BACKEND", "hash")
    eager = os.getenv("ALPHANUS_MEMORY_EAGER_LOAD", "0") == "1"
    return VectorMemory(
        storage_path=storage_path,
        model_name=model_name,
        embedding_backend=backend,
        eager_load_encoder=eager,
    )


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


def _find_conflicting_fact_ids(memory: VectorMemory, fact_key: str, fact_value: str) -> list[int]:
    forgotten_ids: list[int] = []
    target = _normalize_whitespace(fact_value)
    for item in list(memory.memories):
        fact = _fact_from_item(item)
        if not fact:
            continue
        key, value = fact
        if key != fact_key:
            continue
        if _normalize_whitespace(value) != target:
            forgotten_ids.append(int(item.id))
    return forgotten_ids


def _ids_from_replace_query(memory: VectorMemory, args: Dict[str, Any], text: str) -> list[int]:
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
        if memory_id <= 0:
            continue
        if _normalize_whitespace(hit_text) == text_norm:
            continue
        out.append(memory_id)
    return out


def _run(tool_name: str, args: Dict[str, Any], memory: VectorMemory) -> Dict[str, Any]:
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
        meta: Dict[str, Any] = {}
        if forgotten_ids:
            meta["forgotten_ids"] = forgotten_ids
            if fact:
                meta["auto_resolution"] = fact[0]
            elif args.get("replace_query"):
                meta["auto_resolution"] = "replace_query"
        return _ok(item, meta)

    if tool_name == "recall_memory":
        hits = memory.search(
            query=args["query"],
            top_k=int(args.get("top_k", 5)),
            memory_type=args.get("memory_type"),
        )
        return _ok({"hits": hits})

    if tool_name == "list_memories":
        rec = memory.list_recent(count=int(args.get("count", 5)))
        return _ok({"memories": rec})

    if tool_name == "forget_memory":
        deleted = memory.forget(int(args["memory_id"]))
        if not deleted:
            return _err("E_NOT_FOUND", "Memory id not found")
        return _ok({"deleted": True})

    if tool_name == "get_memory_stats":
        return _ok(memory.stats())

    if tool_name == "export_memories":
        workspace_root = os.getenv("ALPHANUS_WORKSPACE_ROOT", "").strip()
        if not workspace_root:
            raise ValueError("ALPHANUS_WORKSPACE_ROOT is required")
        path = args.get("filepath") or str(Path(workspace_root) / "memories_export.txt")
        out = memory.export_txt(str(path))
        return _ok({"filepath": out})

    return _err("E_UNSUPPORTED", f"Unsupported tool: {tool_name}")


def main() -> int:
    tool_name = sys.argv[1] if len(sys.argv) > 1 else os.getenv("ALPHANUS_TOOL_NAME", "").strip()
    if not tool_name:
        print(json.dumps(_err("E_VALIDATION", "Missing tool name"), ensure_ascii=False))
        return 2

    try:
        args = _read_args()
        memory = _memory()
        out = _run(tool_name, args, memory)
    except ValueError as exc:
        out = _err("E_VALIDATION", str(exc))
    except FileNotFoundError as exc:
        out = _err("E_NOT_FOUND", str(exc))
    except PermissionError as exc:
        out = _err("E_POLICY", str(exc))
    except TimeoutError as exc:
        out = _err("E_TIMEOUT", str(exc))
    except Exception as exc:  # pragma: no cover - defensive safeguard
        out = _err("E_IO", str(exc))

    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
