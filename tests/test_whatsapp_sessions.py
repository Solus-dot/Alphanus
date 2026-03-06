from __future__ import annotations

import time
from pathlib import Path

from whatsapp.sessions import ChatSession, SessionStore


def test_corrupt_session_file_is_quarantined(tmp_path: Path):
    store = SessionStore(str(tmp_path))
    path = tmp_path / "chat-1.json"
    path.write_text("{not json", encoding="utf-8")

    loaded = store.load("chat-1")
    assert loaded is None
    assert (tmp_path / "chat-1.json.corrupted").exists()


def test_session_save_load_roundtrip(tmp_path: Path):
    store = SessionStore(str(tmp_path))
    session = ChatSession(
        chat_id="chat-2",
        last_message_id="m42",
        history=[{"role": "user", "content": "hi"}],
        updated_at_utc=time.time(),
    )

    store.save(session)
    loaded = store.load("chat-2")

    assert loaded is not None
    assert loaded.chat_id == "chat-2"
    assert loaded.last_message_id == "m42"
    assert loaded.history[0]["content"] == "hi"
