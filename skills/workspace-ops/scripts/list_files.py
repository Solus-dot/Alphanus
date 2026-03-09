#!/usr/bin/env python3
from __future__ import annotations

from _common import emit, load_workspace, read_args


def main() -> int:
    args = read_args()
    names = load_workspace().list_files(str(args.get("path", ".")))
    emit({"files": names})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
