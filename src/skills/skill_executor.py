from __future__ import annotations

import queue
import subprocess
import threading
import time
from typing import Any, cast

from core.errors import OperationCancelled
from core.message_types import JSONValue
from core.tool_results import error_result, ok_result


def _err(code: str, message: str, duration_ms: int) -> dict[str, Any]:
    return cast(dict[str, Any], error_result(code, message, duration_ms=duration_ms))


def _schema_type_matches(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, float)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "null":
        return value is None
    return True


def _validate_schema_value(field_name: str, value: Any, schema: dict[str, Any]) -> None:
    enum = schema.get("enum")
    if isinstance(enum, list) and enum and value not in enum:
        raise ValueError(f"Invalid '{field_name}': expected one of {', '.join(map(str, enum[:10]))}")
    raw_type = schema.get("type")
    expected = (
        [raw_type]
        if isinstance(raw_type, str)
        else [str(item) for item in raw_type if isinstance(item, str)]
        if isinstance(raw_type, list)
        else []
    )
    if expected and not any(_schema_type_matches(value, item) for item in expected):
        raise ValueError(f"Invalid '{field_name}': expected {' or '.join(expected)}")
    if isinstance(value, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            for item in schema.get("required") or []:
                key = str(item).strip()
                if key and key not in value:
                    raise ValueError(f"Missing required argument: {field_name}.{key}")
            for key, child in props.items():
                if key in value and isinstance(child, dict):
                    _validate_schema_value(f"{field_name}.{key}", value[key], child)
            if schema.get("additionalProperties") is False:
                unknown = [key for key in value if key not in props]
                if unknown:
                    raise ValueError(f"Unexpected arguments for '{field_name}': {', '.join(sorted(unknown)[:5])}")
    if isinstance(value, list) and isinstance(items := schema.get("items"), dict):
        for index, item in enumerate(value):
            _validate_schema_value(f"{field_name}[{index}]", item, items)


def _validate_tool_args(reg, args: dict[str, Any]) -> dict[str, Any]:
    cleaned = {key: value for key, value in args.items() if not str(key).startswith("_")}
    schema = reg.parameters if isinstance(reg.parameters, dict) else {}
    if schema.get("type") not in {None, "object"}:
        raise ValueError(f"Tool '{reg.name}' must declare an object parameters schema")
    for item in schema.get("required") or []:
        key = str(item).strip()
        if key and key not in cleaned:
            raise ValueError(f"Missing required argument: {key}")
    props = schema.get("properties")
    if isinstance(props, dict):
        for key, child in props.items():
            if key in cleaned and isinstance(child, dict):
                _validate_schema_value(key, cleaned[key], child)
        if schema.get("additionalProperties") is False:
            unknown = [key for key in cleaned if key not in props]
            if unknown:
                raise ValueError(f"Unexpected arguments: {', '.join(sorted(unknown)[:5])}")
    return cleaned


def _resolve_tool_call(runtime, tool_name: str, selected, ctx):
    reg = runtime.tool_registration(tool_name)
    if not reg:
        raise LookupError(f"No adapter for tool '{tool_name}'")
    if reg.name not in runtime.allowed_tool_names(selected, ctx=ctx):
        raise PermissionError(f"Tool '{tool_name}' is not enabled by the current skill configuration")
    owner = None if reg.name in runtime.always_available_tool_names else runtime.get_skill(reg.skill_id)
    return reg, owner


def _require_bundled_executable_skill(skill) -> None:
    # Executable code is restricted to reviewed bundled skills.
    if skill is None or not skill.path:
        raise FileNotFoundError("Selected skill root is unavailable")
    if not skill.execution_allowed:
        raise PermissionError(f"Executable user skill '{skill.id}' is disabled; only reviewed bundled skills may execute")


def execute_registered_tool(runtime, reg, args: dict[str, Any], env, ctx) -> Any:
    if reg.name == runtime.skills_list_tool_name:
        return runtime.skills_list()
    if reg.name == runtime.skill_view_tool_name:
        return runtime.skill_view(str(args.get("name", "")).strip(), str(args.get("file_path", "")).strip(), ctx)
    if reg.name == runtime.request_user_input_tool_name:
        if not env.request_user_input:
            raise PermissionError("User input runtime is unavailable")
        return env.request_user_input(args)
    skill = runtime.get_skill(str(getattr(reg, "skill_id", "")).strip())
    if skill is not None:
        _require_bundled_executable_skill(skill)
    if reg.module is None and reg.module_path:
        reg.module = runtime._load_module(reg.module_path, reg.module_name or f"alphanus_tools_{reg.skill_id}")
    if reg.module is None or not hasattr(reg.module, "execute"):
        raise runtime.ToolProtocolError(f"Tool '{reg.name}' has no callable execute() handler")
    return reg.module.execute(reg.name, args, env)


def _execute_with_timeout(runtime, reg, args, env, ctx, timeout_s: float | None, stop_event) -> Any:
    if timeout_s is None:
        return execute_registered_tool(runtime, reg, args, env, ctx)
    outcome: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            outcome.put((True, execute_registered_tool(runtime, reg, args, env, ctx)))
        except BaseException as exc:
            outcome.put((False, exc))

    # ponytail: timed-out handlers must cooperate with stop_event; move tools to subprocesses if late mutations appear.
    worker = threading.Thread(target=run, daemon=True, name=f"alphanus-tool-{reg.name}")
    worker.start()
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while worker.is_alive():
        if stop_event is not None and stop_event.is_set():
            raise OperationCancelled(f"Tool '{reg.name}' cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if stop_event is not None:
                stop_event.set()
            raise TimeoutError(f"Tool '{reg.name}' exceeded its turn deadline")
        worker.join(min(0.05, remaining))
    ok, value = outcome.get_nowait()
    if ok:
        return value
    raise value


def execute_tool_call(
    runtime,
    tool_name: str,
    args: dict[str, Any],
    selected,
    ctx,
    request_approval=None,
    request_user_input=None,
    stop_event=None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        reg, _owner = _resolve_tool_call(runtime, tool_name, selected, ctx)
        normalized_args = _validate_tool_args(reg, args)
        env = runtime.ToolExecutionEnv(
            project=runtime.project,
            memory=runtime.memory,
            config=runtime.config,
            debug=runtime.debug,
            request_approval=request_approval,
            request_user_input=request_user_input,
            stop_event=stop_event,
        )
        result = _execute_with_timeout(runtime, reg, normalized_args, env, ctx, timeout_s, stop_event)
        duration = int((time.perf_counter() - start) * 1000)
        return normalize_result(runtime, result, duration)
    except LookupError as exc:
        return _err("E_UNSUPPORTED", str(exc), int((time.perf_counter() - start) * 1000))
    except ValueError as exc:
        return _err("E_VALIDATION", str(exc), int((time.perf_counter() - start) * 1000))
    except FileNotFoundError as exc:
        return _err("E_NOT_FOUND", str(exc), int((time.perf_counter() - start) * 1000))
    except PermissionError as exc:
        return _err("E_POLICY", str(exc), int((time.perf_counter() - start) * 1000))
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        return _err("E_TIMEOUT", str(exc), int((time.perf_counter() - start) * 1000))
    except OperationCancelled as exc:
        return _err("E_CANCELLED", str(exc), int((time.perf_counter() - start) * 1000))
    except runtime.ToolProtocolError as exc:
        return _err("E_PROTOCOL", str(exc), int((time.perf_counter() - start) * 1000))
    except RuntimeError as exc:
        message = str(exc).strip() or "Tool raised RuntimeError"
        return _err("E_IO", message, int((time.perf_counter() - start) * 1000))
    except Exception as exc:
        detail = str(exc).strip()
        message = f"Tool raised {exc.__class__.__name__}"
        if detail:
            message = f"{message}: {detail}"
        return _err("E_IO", message, int((time.perf_counter() - start) * 1000))


def normalize_result(runtime, result: Any, duration_ms: int) -> dict[str, Any]:
    if isinstance(result, dict) and {"ok", "data", "error"}.issubset(result.keys()):
        out = dict(result)
        raw_meta = out.get("meta")
        meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        meta["duration_ms"] = int(meta.get("duration_ms", duration_ms))
        out["meta"] = meta
        return out
    return cast(dict[str, Any], ok_result(cast(JSONValue, result), duration_ms=duration_ms))
