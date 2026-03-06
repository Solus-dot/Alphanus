from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

SCHEMA_VERSION = "1.0.0"


@dataclass
class ChatSession:
    chat_id: str
    last_message_id: str
    history: List[dict]
    updated_at_utc: float

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "chat_id": self.chat_id,
            "last_message_id": self.last_message_id,
            "history": self.history,
            "updated_at_utc": self.updated_at_utc,
        }

    @staticmethod
    def from_dict(data: dict) -> "ChatSession":
        schema = data.get("schema_version", "1.0.0")
        if int(schema.split(".", 1)[0]) != int(SCHEMA_VERSION.split(".", 1)[0]):
            raise ValueError(f"Unsupported session schema_version: {schema}")
        return ChatSession(
            chat_id=data["chat_id"],
            last_message_id=data.get("last_message_id", ""),
            history=list(data.get("history", [])),
            updated_at_utc=float(data.get("updated_at_utc", time.time())),
        )


class SessionStore:
    def __init__(self, root: str, ttl_seconds: int = 24 * 3600):
        self.root = Path(os.path.expanduser(root)).resolve()
        self.ttl_seconds = ttl_seconds
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, chat_id: str) -> Path:
        safe = chat_id.replace("/", "_")
        return self.root / f"{safe}.json"

    def load(self, chat_id: str) -> Optional[ChatSession]:
        path = self._path(chat_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return ChatSession.from_dict(data)

    def save(self, session: ChatSession) -> None:
        path = self._path(session.chat_id)
        path.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def cleanup(self) -> int:
        now = time.time()
        deleted = 0
        for path in self.root.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                updated = float(data.get("updated_at_utc", 0))
            except Exception:
                updated = 0
            if now - updated > self.ttl_seconds:
                path.unlink(missing_ok=True)
                deleted += 1
        return deleted
