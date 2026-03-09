#!/usr/bin/env python3
from __future__ import annotations

from _common import emit, load_memory


def main() -> int:
    emit(load_memory().stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
