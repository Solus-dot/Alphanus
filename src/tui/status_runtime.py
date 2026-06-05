from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.provider_failure_policy import is_local_endpoint
from agent.types import ModelStatus
from tui.status import metadata_center_markup, metadata_right_markup, status_left_markup


@dataclass(slots=True)
class StatusRuntimeState:
    model_status: ModelStatus = field(default_factory=ModelStatus)
    model_name: str | None = None
    model_context_window: int | None = None
    refresh_inflight: bool = False
    startup_readiness_inflight: bool = False

    def should_startup_readiness_poll(self, *, is_local_endpoint: bool) -> bool:
        return is_local_endpoint and not self.startup_readiness_inflight and self.model_status.last_success_at <= 0

    def mark_refresh_started(self) -> None:
        self.refresh_inflight = True

    def apply_model_status(self, status: ModelStatus) -> bool:
        self.refresh_inflight = False
        previous_state = (self.model_name, self.model_context_window)
        self.model_status = status
        self.model_name = str(status.model_name or "").strip() or self.model_name
        self.model_context_window = (
            status.context_window if isinstance(status.context_window, int) and status.context_window > 0 else self.model_context_window
        )
        return previous_state != (self.model_name, self.model_context_window)


def should_startup_readiness_poll(host: Any) -> bool:
    return host._status_runtime.should_startup_readiness_poll(is_local_endpoint=is_local_endpoint(host.agent.models_endpoint))


def start_startup_readiness_poll(host: Any) -> None:
    if not should_startup_readiness_poll(host):
        return
    host._status_runtime.startup_readiness_inflight = True
    host._startup_readiness_worker()


def finish_startup_readiness_poll(host: Any, status: ModelStatus) -> None:
    host._status_runtime.startup_readiness_inflight = False
    apply_model_status(host, status)


def maybe_refresh_model_status(host: Any, *, force: bool = False) -> None:
    if not force or host._status_runtime.refresh_inflight:
        return
    host._status_runtime.mark_refresh_started()
    host._refresh_model_status_worker()


def apply_model_status(host: Any, status: ModelStatus) -> None:
    host._status_runtime.apply_model_status(status)
    host._update_status1()
    host._update_metadata()


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
    app.query_one("#status-right").update(left)


def update_metadata(app: Any) -> None:
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
    app.query_one("#meta-center").update(
        metadata_center_markup(session_name=app._session_title or "Session", branch_name=app._current_branch_name(), width=width, colors=colors)
    )
    app.query_one("#meta-right").update(
        metadata_right_markup(
            endpoint=app.agent.model_endpoint,
            context_tokens=app._context_tokens(),
            context_window=app._context_window_tokens(),
            width=width,
            endpoint_state=app._status_runtime.model_status.state,
            model_integrity=str(backend_info.get("model_integrity", "unknown")),
            colors=colors,
        )
    )
