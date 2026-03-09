#!/usr/bin/env python3
from __future__ import annotations

from _common import emit, load_memory, read_args


def main() -> int:
    args = read_args()
    rec = load_memory().list_recent(count=int(args.get("count", 5)))
    emit({"memories": rec})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
