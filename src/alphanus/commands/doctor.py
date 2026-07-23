import json
from pathlib import Path
from typing import Any

from alphanus.console import _CliTheme, _doctor_state, _print_doctor_group, _print_review_group
from alphanus.paths import get_app_paths
from alphanus.runtime_factory import _as_object, _build_agent_runtime, _load_runtime_config


def _run_doctor(args: Any) -> int:
    theme = _CliTheme()
    app_paths = get_app_paths()
    try:
        config, warnings = _load_runtime_config(app_paths, args)
    except (FileNotFoundError, OSError, ValueError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc), "config_warnings": []}, indent=2))
            return 2
        print(f"{theme.error('doctor config error:')} {exc}")
        return 2

    try:
        project, memory, _runtime, agent = _build_agent_runtime(app_paths, config, debug=args.debug)
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc), "config_warnings": warnings}, indent=2))
            return 2
        print(f"{theme.error('doctor runtime error:')} {exc}")
        return 2
    report = agent.doctor_report()
    report_obj = report if isinstance(report, dict) else {}
    agent_status = _as_object(report_obj.get("agent"))
    project_status = _as_object(report_obj.get("project"))
    search_status = _as_object(report_obj.get("search"))
    retrieval_status = _as_object(report_obj.get("retrieval"))
    failures = []
    if str(agent_status.get("endpoint_policy_error") or "").strip():
        failures.append("endpoint-policy")
    if not bool(agent_status.get("ready")):
        failures.append("model-readiness")
    if not bool(project_status.get("exists")) or not bool(project_status.get("writable")):
        failures.append("project")
    if not bool(search_status.get("ready")):
        failures.append("search")
    if not bool(retrieval_status.get("ready")):
        failures.append("retrieval")
    if args.json:
        payload: dict[str, Any] = dict(report_obj) if report_obj else {"doctor": report}
        payload["ok"] = not failures
        payload["failures"] = failures
        payload["config_warnings"] = warnings
        print(json.dumps(payload, indent=2))
    else:
        endpoint_error = str(agent_status.get("endpoint_policy_error") or "").strip()
        endpoint_ok = not endpoint_error
        model_ok = bool(agent_status.get("ready"))
        project_ok = bool(project_status.get("exists")) and bool(project_status.get("writable"))
        search_ok = bool(search_status.get("ready"))
        retrieval_ok = bool(retrieval_status.get("ready"))

        print("")
        print(theme.brand(" ALPHANUS DOCTOR "))
        print(theme.rule())
        print(theme.muted("Validating model connectivity, project access, search, and local state."))
        print("")

        if warnings:
            print(theme.rule("Config Warnings"))
            for warning in warnings:
                print(f"  {theme.warn('!')} {warning}")
            print("")

        _print_review_group(
            theme,
            "Runtime Paths",
            [
                ("Config", str(app_paths.config_path)),
                ("Sessions", str((Path(app_paths.state_root) / "sessions").resolve())),
                ("Memory", str(memory.storage_path)),
                ("Project", str(project.project_root)),
            ],
        )

        search_detail = str(search_status.get("reason") or "").strip() or "ok"
        retrieval_detail = str(retrieval_status.get("reason") or "").strip() or "ok"
        _print_doctor_group(
            theme,
            "Checks",
            [
                ("Endpoint policy", "ok" if endpoint_ok else "fail", "" if endpoint_ok else endpoint_error),
                ("Model readiness", "ok" if model_ok else "wait", ""),
                ("Project access", "ok" if project_ok else "fail", ""),
                ("Search provider", "ok" if search_ok else "wait", "" if search_detail == "ok" else search_detail),
                ("Retrieval store", "ok" if retrieval_ok else "fail", "" if retrieval_detail == "ok" else retrieval_detail),
            ],
        )

    if not args.json:
        print(theme.rule("Result"))
        if failures:
            print(f"  {_doctor_state(theme, 'fail')} {theme.label('Attention required')}  {theme.muted(', '.join(failures))}")
            print(f"  {theme.muted('Fix failing checks, then rerun `uv run alphanus doctor`.')}")
        else:
            print(f"  {_doctor_state(theme, 'ok')} {theme.label('Ready')}")
        print("")
    return 1 if failures else 0
