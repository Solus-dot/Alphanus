#!/usr/bin/env python3
from __future__ import annotations

from _common import emit, load_memory, read_args


def main() -> int:
    args = read_args()
    memory = load_memory()
    hits = memory.search(
        query=args["query"],
        top_k=int(args.get("top_k", 5)),
        memory_type=args.get("memory_type"),
    )
    emit({"hits": hits})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
