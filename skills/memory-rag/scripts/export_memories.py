#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from _common import emit, load_memory, read_args


def main() -> int:
    args = read_args()
    workspace_root = os.getenv("ALPHANUS_WORKSPACE_ROOT", "").strip()
    if not workspace_root:
        raise ValueError("ALPHANUS_WORKSPACE_ROOT is required")
    path = args.get("filepath") or str(Path(workspace_root) / "memories_export.txt")
    out = load_memory().export_txt(str(path))
    emit({"filepath": out})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
