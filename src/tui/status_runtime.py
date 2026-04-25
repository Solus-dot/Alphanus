from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from agent.types import ModelStatus
from core.runtime_config import UiTimingConfig
from tui.status import status_left_markup, topbar_center, topbar_left, topbar_right


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


class StatusRuntimeHost(Protocol):
    _status_runtime: StatusRuntimeState
    agent: Any

    def _timing_config(self) -> UiTimingConfig: ...

    def _update_status1(self) -> None: ...

    def _update_topbar(self) -> None: ...

    def _refresh_model_status_worker(self) -> None: ...

    def _startup_readiness_worker(self) -> None: ...


def current_model_refresh_interval(host: StatusRuntimeHost) -> float:
    return host._status_runtime.current_model_refresh_interval(host._timing_config())


def should_startup_readiness_poll(host: StatusRuntimeHost) -> bool:
    return host._status_runtime.should_startup_readiness_poll(
        is_local_endpoint=host.agent.llm_client._is_local_endpoint(host.agent.models_endpoint)
    )


def start_startup_readiness_poll(host: StatusRuntimeHost) -> None:
    if not should_startup_readiness_poll(host):
        return
    host._status_runtime.startup_readiness_inflight = True
    host._startup_readiness_worker()


def finish_startup_readiness_poll(host: StatusRuntimeHost, status: ModelStatus) -> None:
    host._status_runtime.startup_readiness_inflight = False
    apply_model_status(host, status)


def maybe_refresh_model_status(host: StatusRuntimeHost, *, force: bool = False) -> None:
    now = time.monotonic()
    state = host._status_runtime
    if not state.should_refresh(host._timing_config(), force=force, now=now):
        return
    state.mark_refresh_started(now=now)
    host._refresh_model_status_worker()


def apply_model_status(host: StatusRuntimeHost, status: ModelStatus) -> None:
    host._status_runtime.apply_model_status(status, host._timing_config())
    host._update_status1()
    host._update_topbar()


def update_status2(app: Any) -> None:
    colors = app._theme_spec().colors if hasattr(app, "_theme_spec") else None
    left = status_left_markup(
        await_shell_confirm=app._await_shell_confirm,
        streaming=app.streaming,
        spinner_frame=app._spin_frames[app._spin_i % len(app._spin_frames)],
        stop_requested=app._stop_event.is_set(),
        esc_pending=app._esc_pending,
        auto_follow_stream=app._auto_follow_stream,
        focus_panel=app._focused_panel,
        width=app.size.width,
        colors=colors,
    )
    if left == app._last_status_left:
        return
    app._last_status_left = left
    app.query_one("#status-left").update(left)


def update_topbar(app: Any) -> None:
    workspace_root = str(app.agent.skill_runtime.workspace.workspace_root)
    width = app.size.width
    colors = app._theme_spec().colors if hasattr(app, "_theme_spec") else None
    app.query_one("#topbar-left").update(topbar_left(workspace_root, width=width, colors=colors))
    app.query_one("#topbar-center").update(
        topbar_center(
            session_name=app._session_title or "Session",
            branch_name=app._current_branch_name(),
            width=width,
            colors=colors,
        )
    )
    app.query_one("#topbar-right").update(
        topbar_right(
            endpoint=app.agent.model_endpoint,
            context_tokens=app._context_tokens(),
            context_window=app._context_window_tokens(),
            width=width,
            endpoint_state=app._status_runtime.model_status.state,
            collaboration_mode=str(getattr(app, "_collaboration_mode", "execute")),
            colors=colors,
        )
    )
