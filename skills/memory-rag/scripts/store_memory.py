#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from _common import emit, load_memory, read_args


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


def _find_conflicting_fact_ids(memory, fact_key: str, fact_value: str) -> list[int]:
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
        if memory_id <= 0:
            continue
        if _normalize_whitespace(hit_text) == text_norm:
            continue
        out.append(memory_id)
    return out


def main() -> int:
    args = read_args()
    memory = load_memory()

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

    emit({"ok": True, "data": item, "error": None, "meta": meta})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
