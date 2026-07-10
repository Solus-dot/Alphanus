from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TextIO

PROTOCOL_VERSION = 1
EXIT_SUCCESS = 0
EXIT_MODEL_FAILURE = 10
EXIT_POLICY_DENIED = 11
EXIT_INVALID_INPUT = 12
EXIT_CANCELLED = 13
EXIT_INTERNAL = 14


@dataclass(slots=True)
class JsonlEmitter:
    stream: TextIO
    _sequence: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, event_type: str, **data: Any) -> None:
        with self._lock:
            self._sequence += 1
            record = {
                "schema_version": PROTOCOL_VERSION,
                "sequence": self._sequence,
                "timestamp": datetime.now(UTC).isoformat(),
                "type": event_type,
                "data": data,
            }
            self.stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.stream.flush()


def parse_jsonl_request(line: str) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON input: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("JSONL input must be an object")
    version = value.get("schema_version")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported schema_version {version!r}; expected {PROTOCOL_VERSION}")
    prompt = value.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("JSONL input requires a non-empty string prompt")
    return value

