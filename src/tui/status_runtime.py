from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from agent.types import ModelStatus
from tui.status import status_left_markup, topbar_center, topbar_left, topbar_right


@dataclass(slots=True)
class StatusRuntimeState:
    model_status: ModelStatus = field(default_factory=ModelStatus)
    model_name: str | None = None
    model_context_window: int | None = None
    refresh_inflight: bool = False
    last_model_refresh: float = 0.0
    refresh_fast_until: float = 0.0
    startup_readiness_inflight: bool = False

    def current_model_refresh_interval(self, timing: Any, *, now: float | None = None) -> float:
        current_time = time.monotonic() if now is None else now
        return (
            timing.model_refresh_fast_interval_s
            if self.model_status.state == "offline" or current_time < self.refresh_fast_until
            else timing.model_refresh_interval_s
        )

    def should_startup_readiness_poll(self, *, is_local_endpoint: bool) -> bool:
        return is_local_endpoint and not self.startup_readiness_inflight and self.model_status.last_success_at <= 0

    def should_refresh(self, timing: Any, *, force: bool = False, now: float | None = None) -> bool:
        if self.refresh_inflight:
            return False
        current_time = time.monotonic() if now is None else now
        return force or (current_time - self.last_model_refresh) >= self.current_model_refresh_interval(timing, now=current_time)

    def mark_refresh_started(self, *, now: float | None = None) -> None:
        self.refresh_inflight = True
        self.last_model_refresh = time.monotonic() if now is None else now

    def apply_model_status(self, status: ModelStatus, timing: Any, *, now: float | None = None) -> bool:
        self.refresh_inflight = False
        current_time = time.monotonic() if now is None else now
        previous_state = (self.model_name, self.model_context_window)
        self.model_status = status
        self.model_name = str(status.model_name or "").strip() or self.model_name
        self.model_context_window = (
            status.context_window if isinstance(status.context_window, int) and status.context_window > 0 else self.model_context_window
        )
        changed = previous_state != (self.model_name, self.model_context_window)
        if changed:
            self.refresh_fast_until = current_time + timing.model_refresh_fast_window_s
        return changed


def current_model_refresh_interval(host: Any) -> float:
    return host._status_runtime.current_model_refresh_interval(host._timing_config())


def should_startup_readiness_poll(host: Any) -> bool:
    return host._status_runtime.should_startup_readiness_poll(
        is_local_endpoint=host.agent.llm_client._is_local_endpoint(host.agent.models_endpoint)
    )


def start_startup_readiness_poll(host: Any) -> None:
    if not should_startup_readiness_poll(host):
        return
    host._status_runtime.startup_readiness_inflight = True
    host._startup_readiness_worker()


def finish_startup_readiness_poll(host: Any, status: ModelStatus) -> None:
    host._status_runtime.startup_readiness_inflight = False
    apply_model_status(host, status)


def maybe_refresh_model_status(host: Any, *, force: bool = False) -> None:
    now = time.monotonic()
    if not host._status_runtime.should_refresh(host._timing_config(), force=force, now=now):
        return
    host._status_runtime.mark_refresh_started(now=now)
    host._refresh_model_status_worker()


def apply_model_status(host: Any, status: ModelStatus) -> None:
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
    backend_info = {}
    llm_client = getattr(app.agent, "llm_client", None)
    backend_info_fn = getattr(llm_client, "backend_profile_info", None)
    if callable(backend_info_fn):
        try:
            payload = backend_info_fn()
            if isinstance(payload, dict):
                backend_info = payload
        except Exception:
            backend_info = {}
    app.query_one("#topbar-left").update(topbar_left(workspace_root, width=width, colors=colors))
    app.query_one("#topbar-center").update(
        topbar_center(session_name=app._session_title or "Session", branch_name=app._current_branch_name(), width=width, colors=colors)
    )
    app.query_one("#topbar-right").update(
        topbar_right(
            endpoint=app.agent.model_endpoint,
            context_tokens=app._context_tokens(),
            context_window=app._context_window_tokens(),
            width=width,
            endpoint_state=app._status_runtime.model_status.state,
            collaboration_mode=str(getattr(app, "_collaboration_mode", "execute")),
            backend_profile=str(backend_info.get("selected", "")),
            model_integrity=str(backend_info.get("model_integrity", "unknown")),
            colors=colors,
        )
    )
