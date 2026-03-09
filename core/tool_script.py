from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict


def read_args() -> Dict[str, Any]:
    raw = os.getenv("ALPHANUS_TOOL_ARGS_JSON", "").strip()
    if not raw:
        raw = sys.stdin.read().strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Tool args must be a JSON object")
    return parsed


def read_config() -> Dict[str, Any]:
    raw = os.getenv("ALPHANUS_CONFIG_JSON", "").strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False))
