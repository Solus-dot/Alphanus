import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from alphanus_cli import main

_MAIN_DEPRECATION_MESSAGE = (
    "Deprecation warning: `uv run main.py` is retained for repo checkout convenience but is deprecated. "
    "Use `uv run alphanus` instead. Planned removal date: September 1, 2026."
)


if __name__ == "__main__":
    print(_MAIN_DEPRECATION_MESSAGE, file=sys.stderr)
    raise SystemExit(main())
