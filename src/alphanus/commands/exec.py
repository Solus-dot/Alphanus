import signal
import sys
import threading
from typing import Any

from alphanus.paths import get_app_paths
from alphanus.runtime_factory import _build_agent_runtime, _close_memory, _load_runtime_config
from core.headless_protocol import (
    EXIT_CANCELLED,
    EXIT_INTERNAL,
    EXIT_INVALID_INPUT,
    EXIT_MODEL_FAILURE,
    EXIT_POLICY_DENIED,
    EXIT_SUCCESS,
    JsonlEmitter,
    parse_jsonl_request,
)


def _run_exec(args: Any) -> int:
    emitter = JsonlEmitter(sys.stdout)
    stop_event = threading.Event()
    memory: Any = None
    previous_handlers: dict[int, Any] = {}

    def cancel(_signum: int, _frame: Any) -> None:
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, cancel)
    try:
        from agent.telemetry import configure_logging

        raw = str(getattr(args, "prompt", "") or "")
        if not raw:
            raw = sys.stdin.readline() if args.input == "jsonl" else sys.stdin.read()
        if args.input == "jsonl":
            request = parse_jsonl_request(raw.strip())
            prompt = str(request["prompt"])
        else:
            prompt = raw.strip()
            if not prompt:
                raise ValueError("a non-empty prompt is required")

        app_paths = get_app_paths()
        config, warnings = _load_runtime_config(app_paths, args)
        configure_logging(config)
        for warning in warnings:
            print(f"config warning: {warning}", file=sys.stderr)
        _project, memory, _runtime, agent = _build_agent_runtime(app_paths, config, debug=args.debug)
        emitter.emit("run.started", workspace=str(agent.skill_runtime.project.project_root))

        def on_event(event: dict[str, Any]) -> None:
            event_type = str(event.get("type") or "agent.event")
            emitter.emit(event_type, **{str(k): v for k, v in event.items() if k != "type"})

        def request_approval(request: dict[str, Any]) -> bool:
            allowed = args.approval_policy == "allow-boundary"
            emitter.emit("approval.requested", request=request, decision="approved" if allowed else "denied")
            return allowed

        result = agent.run_turn(
            history_messages=[],
            user_input=prompt,
            thinking=not args.no_thinking,
            stop_event=stop_event,
            on_event=on_event,
            request_approval=request_approval,
        )
        if stop_event.is_set() or result.status == "cancelled":
            emitter.emit("run.completed", status="cancelled")
            return EXIT_CANCELLED
        if result.status not in {"ok", "done"}:
            error = str(result.error or "model execution failed")
            denied = "approval" in error.casefold() or "policy" in error.casefold() or "permission" in error.casefold()
            emitter.emit("run.error", category="policy" if denied else "model", message=error)
            emitter.emit("run.completed", status="error")
            return EXIT_POLICY_DENIED if denied else EXIT_MODEL_FAILURE
        emitter.emit("assistant.final", content=result.content)
        emitter.emit("run.completed", status="success")
        return EXIT_SUCCESS
    except (FileNotFoundError, ValueError) as exc:
        emitter.emit("run.error", category="input", message=str(exc))
        emitter.emit("run.completed", status="error")
        return EXIT_INVALID_INPUT
    except Exception as exc:
        emitter.emit("run.error", category="internal", message=f"{type(exc).__name__}: {exc}")
        emitter.emit("run.completed", status="error")
        return EXIT_INTERNAL
    finally:
        _close_memory(memory)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
