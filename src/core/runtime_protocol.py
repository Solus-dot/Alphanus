from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, TextIO

RUNTIME_PROTOCOL_VERSION = 1
MAX_RUNTIME_FRAME_BYTES = 1024 * 1024


class RuntimeProtocolError(ValueError):
    pass


def decode_runtime_frame(line: str) -> dict[str, Any]:
    if len(line.encode("utf-8")) > MAX_RUNTIME_FRAME_BYTES:
        raise RuntimeProtocolError("runtime frame exceeds 1 MiB")
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeProtocolError(f"invalid runtime JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise RuntimeProtocolError("runtime frame must be an object")
    if value.get("protocol_version") != RUNTIME_PROTOCOL_VERSION:
        raise RuntimeProtocolError(
            f"unsupported runtime protocol {value.get('protocol_version')!r}; expected {RUNTIME_PROTOCOL_VERSION}"
        )
    message_type = value.get("type")
    request_id = value.get("request_id")
    if not isinstance(message_type, str) or not message_type.strip():
        raise RuntimeProtocolError("runtime frame requires a non-empty type")
    if not isinstance(request_id, str) or not request_id.strip():
        raise RuntimeProtocolError("runtime frame requires a non-empty request_id")
    data = value.get("data", {})
    if not isinstance(data, dict):
        raise RuntimeProtocolError("runtime frame data must be an object")
    return value


@dataclass(slots=True)
class RuntimeEmitter:
    stream: TextIO
    _sequence: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(
        self,
        event_type: str,
        *,
        request_id: str,
        data: dict[str, Any] | None = None,
        turn_id: str | None = None,
        approval_id: str | None = None,
    ) -> None:
        with self._lock:
            self._sequence += 1
            frame: dict[str, Any] = {
                "protocol_version": RUNTIME_PROTOCOL_VERSION,
                "type": event_type,
                "request_id": request_id,
                "sequence": self._sequence,
                "data": data or {},
            }
            if turn_id:
                frame["turn_id"] = turn_id
            if approval_id:
                frame["approval_id"] = approval_id
            encoded = json.dumps(frame, ensure_ascii=False, separators=(",", ":"))
            if len(encoded.encode("utf-8")) > MAX_RUNTIME_FRAME_BYTES:
                raise RuntimeProtocolError(f"outbound {event_type} frame exceeds 1 MiB")
            self.stream.write(encoded + "\n")
            self.stream.flush()
