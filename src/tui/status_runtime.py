from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from agent.types import ModelStatus
from core.runtime_config import UiTimingConfig


@dataclass(slots=True)
class StatusRuntimeState:
    model_status: ModelStatus = field(default_factory=ModelStatus)
    model_name: Optional[str] = None
    model_context_window: Optional[int] = None
    refresh_inflight: bool = False
    last_model_refresh: float = 0.0
    refresh_fast_until: float = 0.0
    startup_readiness_inflight: bool = False

    def current_model_refresh_interval(self, timing: UiTimingConfig, *, now: Optional[float] = None) -> float:
        current_time = time.monotonic() if now is None else now
        if self.model_status.state == "offline" or current_time < self.refresh_fast_until:
            return timing.model_refresh_fast_interval_s
        return timing.model_refresh_interval_s

    def should_startup_readiness_poll(self, *, is_local_endpoint: bool) -> bool:
        if self.startup_readiness_inflight:
            return False
        if self.model_status.last_success_at > 0:
            return False
        return is_local_endpoint

    def should_refresh(self, timing: UiTimingConfig, *, force: bool = False, now: Optional[float] = None) -> bool:
        if self.refresh_inflight:
            return False
        if force:
            return True
        current_time = time.monotonic() if now is None else now
        return (current_time - self.last_model_refresh) >= self.current_model_refresh_interval(timing, now=current_time)

    def mark_refresh_started(self, *, now: Optional[float] = None) -> None:
        self.refresh_inflight = True
        self.last_model_refresh = time.monotonic() if now is None else now

    def apply_model_status(self, status: ModelStatus, timing: UiTimingConfig, *, now: Optional[float] = None) -> bool:
        self.refresh_inflight = False
        current_time = time.monotonic() if now is None else now
        previous_name = self.model_name
        previous_context = self.model_context_window
        self.model_status = status
        self.model_name = status.model_name if status.state != "offline" else None
        self.model_context_window = (
            status.context_window if isinstance(status.context_window, int) and status.context_window > 0 else None
        )
        changed = previous_name != self.model_name or previous_context != self.model_context_window
        if changed:
            self.refresh_fast_until = current_time + timing.model_refresh_fast_window_s
        return changed
