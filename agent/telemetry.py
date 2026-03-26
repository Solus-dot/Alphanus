from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _compact(value: Any, depth: int = 0) -> Any:
    if depth >= 8:
        return "[truncated]"
    if isinstance(value, str):
        if value.startswith("data:image/"):
            prefix, _, encoded = value.partition(",")
            return f"{prefix},...[{len(encoded)} base64 chars]"
        if len(value) <= 4000:
            return value
        return value[:4000] + f"...[truncated {len(value) - 4000} chars]"
    if isinstance(value, list):
        items = [_compact(item, depth + 1) for item in value[:50]]
        if len(value) > 50:
            items.append(f"...[{len(value) - 50} more items]")
        return items
    if isinstance(value, dict):
        items = list(value.items())
        out: Dict[str, Any] = {}
        for key, item in items[:80]:
            out[str(key)] = _compact(item, depth + 1)
        if len(items) > 80:
            out["__truncated_keys__"] = len(items) - 80
        return out
    return value


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        event = getattr(record, "event_name", "")
        if event:
            payload["event"] = event
        event_payload = getattr(record, "event_payload", None)
        if event_payload is not None:
            payload["payload"] = _compact(event_payload)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(config: Dict[str, Any]) -> logging.Logger:
    logging_cfg = config.get("logging", {}) if isinstance(config.get("logging"), dict) else {}
    level_name = str(logging_cfg.get("level", "INFO")).strip().upper() or "INFO"
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger("alphanus")
    root.setLevel(level)
    root.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_level = level if level >= logging.WARNING else logging.WARNING
    stream_handler.setLevel(stream_level)
    if str(logging_cfg.get("format", "plain")).strip().lower() == "json":
        formatter: logging.Formatter = JsonLineFormatter()
    else:
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    path_text = str(logging_cfg.get("path", "")).strip()
    if path_text:
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(JsonLineFormatter())
        root.addHandler(file_handler)
    root.propagate = False
    return root


class TelemetryEmitter:
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("alphanus")

    def emit(self, event_name: str, **payload: Any) -> None:
        try:
            self.logger.info(
                event_name,
                extra={"event_name": event_name, "event_payload": payload},
            )
        except Exception:
            return
