#!/usr/bin/env python3
from __future__ import annotations

from _common import emit, load_workspace, read_args


def main() -> int:
    args = read_args()
    max_depth = max(1, int(args.get("max_depth", 3)))
    emit({"tree": load_workspace().workspace_tree(max_depth=max_depth)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
