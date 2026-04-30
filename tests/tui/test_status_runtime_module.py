from __future__ import annotations

import time
from types import SimpleNamespace

from core.types import ModelStatus
from core.runtime_config import UiTimingConfig
from tui.status_runtime import (
    StatusRuntimeState,
    apply_model_status,
    maybe_refresh_model_status,
    should_startup_readiness_poll,
    start_startup_readiness_poll,
)


class _Host:
    def __init__(self) -> None:
        self._status_runtime = StatusRuntimeState(model_status=ModelStatus(state="offline"))
        self.agent = SimpleNamespace(
            models_endpoint="http://127.0.0.1:8080/v1/models",
            llm_client=SimpleNamespace(_is_local_endpoint=lambda _url: True),
        )
        self._timing = UiTimingConfig()
        self.refresh_calls = 0
        self.startup_calls = 0
        self.update_status_calls = 0
        self.update_topbar_calls = 0

    def _timing_config(self) -> UiTimingConfig:
        return self._timing

    def _refresh_model_status_worker(self) -> None:
        self.refresh_calls += 1

    def _startup_readiness_worker(self) -> None:
        self.startup_calls += 1

    def _update_status1(self) -> None:
        self.update_status_calls += 1

    def _update_topbar(self) -> None:
        self.update_topbar_calls += 1


def test_start_startup_readiness_poll_sets_flag_and_starts_worker() -> None:
    host = _Host()

    assert should_startup_readiness_poll(host) is True
    start_startup_readiness_poll(host)

    assert host._status_runtime.startup_readiness_inflight is True
    assert host.startup_calls == 1


def test_maybe_refresh_model_status_triggers_worker_when_due() -> None:
    host = _Host()
    host._status_runtime.last_model_refresh = time.monotonic() - 99

    maybe_refresh_model_status(host)

    assert host.refresh_calls == 1
    assert host._status_runtime.refresh_inflight is True


def test_apply_model_status_updates_host_views() -> None:
    host = _Host()
    status = ModelStatus(state="online", model_name="qwen", context_window=8192)

    apply_model_status(host, status)

    assert host._status_runtime.model_name == "qwen"
    assert host._status_runtime.model_context_window == 8192
    assert host.update_status_calls == 1
    assert host.update_topbar_calls == 1


def test_apply_model_status_keeps_last_known_model_when_offline() -> None:
    host = _Host()
    host._status_runtime.model_name = "qwen"
    host._status_runtime.model_context_window = 8192

    apply_model_status(host, ModelStatus(state="offline", model_name=None, context_window=None))

    assert host._status_runtime.model_name == "qwen"
    assert host._status_runtime.model_context_window == 8192
