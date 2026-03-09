#!/usr/bin/env python3
from __future__ import annotations

from _common import emit, load_workspace, read_args


def main() -> int:
    args = read_args()
    text = load_workspace().read_file(str(args["filepath"]))
    emit({"content": text})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
