#!/usr/bin/env python3
from __future__ import annotations

from _common import emit, load_memory, read_args


def main() -> int:
    args = read_args()
    deleted = load_memory().forget(int(args["memory_id"]))
    if not deleted:
        raise FileNotFoundError("Memory id not found")
    emit({"deleted": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
