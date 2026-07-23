import sys
from pathlib import Path
from typing import Any

from alphanus.paths import get_app_paths
from alphanus.runtime_factory import _build_agent_runtime, _close_memory, _load_runtime_config
from core.headless_protocol import EXIT_INTERNAL


def _run_runtime(args: Any) -> int:
    # Private bidirectional backend for the bundled frontend.
    app_paths = get_app_paths()
    memory: Any = None
    try:
        from agent.telemetry import configure_logging

        config, warnings = _load_runtime_config(app_paths, args)
        logger = configure_logging(config)
        for warning in warnings:
            logger.warning(f"config: {warning}")
        _project, memory, _skill_runtime, agent = _build_agent_runtime(app_paths, config, debug=args.debug)
        from core.runtime_server import RuntimeServer

        server = RuntimeServer(
            agent=agent,
            memory=memory,
            state_root=Path(app_paths.state_root).resolve(),
            config_path=app_paths.config_path,
            input_stream=sys.stdin,
            output_stream=sys.stdout,
        )
        memory = None  # RuntimeServer owns orderly cleanup after construction.
        return server.serve()
    except Exception as exc:
        print(f"runtime startup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INTERNAL
    finally:
        _close_memory(memory)
