#!/usr/bin/env python3
from __future__ import annotations

from _common import emit, load_workspace, read_args


def main() -> int:
    args = read_args()
    path = load_workspace().edit_file(str(args["filepath"]), str(args["content"]))
    emit({"filepath": path})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
