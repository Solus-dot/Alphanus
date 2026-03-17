from __future__ import annotations

import threading
import urllib.request

from core.streaming import stream_chat_completions


class _StallingResponse:
    status = 200
    reason = "OK"

    def __init__(self, stop_event: threading.Event) -> None:
        self._stop_event = stop_event
        self.readline_calls = 0
        self.fp = self
        self.raw = self
        self._sock = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def fileno(self):
        return 0

    def readline(self):
        self.readline_calls += 1
        if self._stop_event.is_set():
            return b""
        raise AssertionError("readline should not be called before select reports readiness")


def test_stream_chat_completions_cancels_during_stalled_read(mocker):
    stop_event = threading.Event()
    response = _StallingResponse(stop_event)

    def fake_urlopen(req, timeout=None, context=None):
        return response

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)
    select_calls = {"count": 0}

    def fake_select(reads, writes, errs, timeout=None):
        select_calls["count"] += 1
        stop_event.set()
        return ([], [], [])

    mocker.patch("core.streaming.select.select", side_effect=fake_select)

    chunks = []
    gen = stream_chat_completions(
        "http://127.0.0.1:8080/v1/chat/completions",
        {"messages": [], "stream": True},
        timeout_s=5,
        stop_event=stop_event,
    )

    chunks.extend(gen)

    assert chunks == []
    assert select_calls["count"] >= 1
    assert response.readline_calls == 0
