#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.tool_script import emit, read_args  # noqa: E402


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main() -> int:
    args = read_args()
    query = str(args["query"]).lower()
    home_root = Path(os.path.expanduser(os.getenv("ALPHANUS_HOME_ROOT", str(Path.home())))).resolve()
    directory = str(args.get("directory") or str(home_root))
    root = Path(os.path.expanduser(directory)).resolve()
    if not _is_under(root, home_root):
        raise PermissionError("Search directory outside home root")

    matches = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if query in name.lower():
                matches.append(str(Path(dirpath) / name))
                if len(matches) >= 200:
                    break
        if len(matches) >= 200:
            break
    emit({"matches": matches})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
