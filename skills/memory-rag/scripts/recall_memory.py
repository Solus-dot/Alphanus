#!/usr/bin/env python3
from __future__ import annotations

import re

from _common import emit, load_memory, read_args


def _lexical_fallback(memory, query: str, top_k: int, memory_type):
    tokens = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 1]
    if not tokens:
        return []

    scored = []
    for item in list(memory.memories):
        if memory_type and item.type != memory_type:
            continue
        text = str(item.text).lower()
        overlap = sum(1 for token in tokens if token in text)
        if overlap <= 0:
            continue
        score = overlap / max(1, len(tokens))
        scored.append((score, item))

    scored.sort(key=lambda pair: (pair[0], pair[1].timestamp), reverse=True)
    out = []
    for score, item in scored[: max(1, top_k)]:
        record = memory._to_public(item)  # noqa: SLF001
        record["score"] = round(float(score), 4)
        out.append(record)
    return out


def main() -> int:
    args = read_args()
    memory = load_memory()
    top_k = int(args.get("top_k", 5))
    min_score = args.get("min_score")
    threshold = 0.18 if min_score is None else float(min_score)
    memory_type = args.get("memory_type")

    hits = memory.search(
        query=args["query"],
        top_k=top_k,
        memory_type=memory_type,
        min_score=threshold,
    )
    if not hits:
        hits = _lexical_fallback(memory, str(args["query"]), top_k=top_k, memory_type=memory_type)
    emit({"hits": hits})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
