from __future__ import annotations

import threading
import time
import urllib.request

import pytest

from core.streaming import MAX_SSE_LINE_BYTES, StreamError, stream_chat_completions


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

    def __exit__(self, *_args):
        return False

    def fileno(self):
        return 0

    def readline(self, _size: int = -1):
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

    def fake_select(_reads, _writes, _errs, timeout=None):
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


def test_stream_chat_completions_cancels_without_a_selectable_socket(mocker) -> None:
    stop_event = threading.Event()
    release = threading.Event()
    reading = threading.Event()

    class Response:
        status = 200
        reason = "OK"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            release.set()
            return False

        def readline(self, _size: int = -1):
            reading.set()
            release.wait(2)
            return b""

    mocker.patch.object(urllib.request, "urlopen", return_value=Response())

    def cancel() -> None:
        reading.wait(1)
        stop_event.set()

    threading.Thread(target=cancel, daemon=True).start()
    started = time.monotonic()
    chunks = list(stream_chat_completions("http://localhost/v1/chat/completions", {"stream": True}, timeout_s=5, stop_event=stop_event))

    assert chunks == []
    assert time.monotonic() - started < 1


def test_stream_chat_completions_rejects_oversized_lines(mocker) -> None:
    class Response:
        status = 200
        reason = "OK"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def readline(self, size: int = -1):
            return b"x" * max(size, MAX_SSE_LINE_BYTES + 1)

    mocker.patch.object(urllib.request, "urlopen", return_value=Response())
    with pytest.raises(StreamError, match="stream line exceeds"):
        list(stream_chat_completions("http://localhost/v1/chat/completions", {"stream": True}, timeout_s=1))
