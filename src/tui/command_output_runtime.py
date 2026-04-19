from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.markup import escape as esc

from tui.status import context_usage_percent
from tui.tree_render import render_tree_rows


def _normalized_validation_errors(values: Any) -> list[str]:
    return [item for item in (str(value).strip() for value in (values or [])) if item]


def _skill_capability_summary(
    *,
    execution_allowed: bool,
    adapter: Any,
    tools: Any,
    scripts: Any,
    entrypoints: Any,
    user_invocable: bool,
    model_invocable: bool,
) -> str:
    parts = [
        f"execution={'yes' if execution_allowed else 'no'}",
        f"adapter={adapter}",
        f"user={'yes' if user_invocable else 'no'}",
        f"model={'yes' if model_invocable else 'no'}",
    ]
    if tools:
        parts.append(f"tools={len(tools)}")
    if scripts:
        parts.append(f"scripts={len(scripts)}")
    if entrypoints:
        parts.append(f"entrypoints={len(entrypoints)}")
    return " · ".join(parts)


def _write_validation_error(app: Any, validation_errors: Any) -> None:
    errors = _normalized_validation_errors(validation_errors)
    if errors:
        app._write(f"    [#f59e0b]validation:[/#f59e0b] [#a1a1aa]{esc(errors[0])}[/#a1a1aa]")


def cmd_help(app: Any, *, help_sections, accent_color: str) -> None:
    app._write("")
    col = max((len(command) for _, rows in help_sections for command, _ in rows), default=22) + 2
    for section, rows in help_sections:
        app._write_section_heading(section)
        for c, desc in rows:
            app._write_command_row(c, desc, col=col)
    app._write("")


def cmd_tree(app: Any, *, accent_color: str) -> None:
    app._write_section_heading("Tree")
    for text, tag, active in render_tree_rows(app.conv_tree, width=80):
        if tag == app.conv_tree.current_id:
            app._write(f"  [bold {accent_color}]{esc(text)}[/bold {accent_color}]")
        elif active:
            app._write(f"  [{accent_color}]{esc(text)}[/{accent_color}]")
        else:
            app._write(f"  [#a1a1aa]{esc(text)}[/#a1a1aa]")
    app._write("")


def cmd_skills(app: Any, *, accent_color: str) -> None:
    skills = app.agent.skill_runtime.list_skills()
    loaded_skill_ids = set(getattr(app, "_loaded_skill_ids", []))
    app._write_section_heading("Skills")
    for skill in skills:
        state, color = app.agent.skill_runtime.skill_status_label(skill)
        source = app.agent.skill_runtime.skill_source_label(skill)
        provenance = app.agent.skill_runtime.skill_provenance_label(skill)
        app._write(
            f"  [bold {accent_color}]{esc(skill.id)}[/bold {accent_color}] "
            f"[#a1a1aa]({esc(skill.version)})[/#a1a1aa] "
            f"[{color}]{state}[/{color}]"
            f"{' [bold #22c55e]loaded[/bold #22c55e]' if skill.id in loaded_skill_ids else ''}"
        )
        app._write(f"    [#a1a1aa]{esc(skill.description)}[/#a1a1aa]")
        source_bits = provenance
        if source:
            source_bits += f" · {source}"
        app._write(f"    [#71717a]{esc(source_bits)}[/#71717a]")
        tools = app.agent.skill_runtime._reported_skill_tools(skill)
        scripts = app.agent.skill_runtime._reported_skill_scripts(skill)
        entrypoints = app.agent.skill_runtime._reported_skill_entrypoints(skill)
        capabilities = _skill_capability_summary(
            execution_allowed=getattr(skill, "execution_allowed", True),
            adapter=getattr(skill, "adapter", "agentskills"),
            tools=tools,
            scripts=scripts,
            entrypoints=entrypoints,
            user_invocable=skill.user_invocable,
            model_invocable=not skill.disable_model_invocation,
        )
        app._write(f"    [#71717a]{esc(capabilities)}[/#71717a]")
        if not skill.available and skill.availability_reason:
            code = esc(skill.availability_code or "blocked")
            app._write(f"    [bold {accent_color}]blocked ({code}):[/bold {accent_color}] [#a1a1aa]{esc(skill.availability_reason)}[/#a1a1aa]")
        _write_validation_error(app, getattr(skill, "validation_errors", []))
    app._write("")


def load_skill_into_session(app: Any, skill_id: str) -> bool:
    skill = app.agent.skill_runtime.get_skill(skill_id)
    if not skill or not skill.enabled or not skill.available:
        app._write_error(f"Skill not found or unavailable: {skill_id}")
        return False
    if not hasattr(app, "_loaded_skill_ids"):
        app._loaded_skill_ids = []
    if skill_id not in app._loaded_skill_ids:
        app._loaded_skill_ids.append(skill_id)
    app._loaded_skill_ids = [skill.id for skill in app.agent.skill_runtime.skills_by_ids(app._loaded_skill_ids)]
    app._save_active_session()
    return True


def unload_skill_from_session(app: Any, skill_id: str) -> bool:
    if skill_id not in getattr(app, "_loaded_skill_ids", []):
        app._write_error(f"Skill not loaded: {skill_id}")
        return False
    app._loaded_skill_ids = [item for item in app._loaded_skill_ids if item != skill_id]
    app._save_active_session()
    return True


def cmd_skill(app: Any, arg: str) -> bool:
    parts = arg.split()
    if not parts:
        return app._write_usage("/skill-on <id> | /skill-off <id> | /skill-unload <id> | /skill-unload-all | /skill-reload | /skill-info <id>")

    sub = parts[0].lower()
    if sub == "reload":
        return app._reload_skills()

    if sub in {"on", "off"}:
        if len(parts) < 2:
            return app._write_usage("/skill-on <id> | /skill-off <id>")
        skill_id = parts[1]
        ok = app.agent.skill_runtime.set_enabled(skill_id, sub == "on")
        if not ok:
            app._write_error(f"Skill not found: {skill_id}")
        else:
            if sub == "off":
                app._loaded_skill_ids = [item for item in app._loaded_skill_ids if item != skill_id]
                app._save_active_session()
            app._write_info(f"Skill {skill_id} {'enabled' if sub == 'on' else 'disabled'}")
        return True

    if sub == "unload-all":
        app._loaded_skill_ids = []
        app._save_active_session()
        app._write_info("Unloaded all session skills")
        return True

    if sub == "unload":
        if len(parts) < 2:
            return app._write_usage("/skill-unload <id>")
        if app._unload_skill_from_session(parts[1]):
            app._write_info(f"Unloaded skill {parts[1]}")
        return True

    if sub == "info":
        if len(parts) < 2:
            return app._write_usage("/skill-info <id>")
        skill = app.agent.skill_runtime.get_skill(parts[1])
        if not skill:
            app._write_error(f"Skill not found: {parts[1]}")
            return True
        app._write_section_heading(skill.name)
        app._write(f"  [#a1a1aa]{esc(skill.description)}[/#a1a1aa]")
        enabled, color = app.agent.skill_runtime.skill_status_label(skill)
        app._write_detail_line("id", skill.id)
        app._write_detail_line("version", skill.version)
        app._write_detail_line("status", f"[{color}]{enabled}[/{color}]", value_markup=True)
        app._write_detail_line("provenance", app.agent.skill_runtime.skill_provenance_label(skill))
        app._write_detail_line("source", app.agent.skill_runtime.skill_source_label(skill) or "unknown")
        app._write_detail_line("execution_allowed", str(bool(getattr(skill, "execution_allowed", True))).lower())
        app._write_detail_line("adapter", getattr(skill, "adapter", "agentskills"))
        app._write_detail_line("availability_code", skill.availability_code or "ready")
        app._write_detail_line("availability", skill.availability_reason or "ready")
        app._write_detail_line("tools", ", ".join(app.agent.skill_runtime._reported_skill_tools(skill)) or "none")
        app._write_detail_line("user_invocable", str(skill.user_invocable).lower())
        app._write_detail_line("model_invocable", str((not skill.disable_model_invocation)).lower())
        scripts = ", ".join(app.agent.skill_runtime._reported_skill_scripts(skill)) or "none"
        entrypoints = ", ".join(entry.name for entry in app.agent.skill_runtime._reported_skill_entrypoints(skill)) or "none"
        app._write_detail_line("scripts", scripts)
        app._write_detail_line("entrypoints", entrypoints)
        app._write_detail_line("validation_errors", "; ".join(getattr(skill, "validation_errors", []) or []) or "none")
        app._write("")
        return True

    return app._write_usage("/skill-on <id> | /skill-off <id> | /skill-unload <id> | /skill-unload-all | /skill-reload | /skill-info <id>")


def cmd_memory(app: Any, arg: str) -> bool:
    sub = arg.strip().lower()
    if sub == "stats":
        stats = app.agent.skill_runtime.memory.stats()
        app._write_section_heading("Memory Stats")
        app._write_detail_line("count", str(stats["count"]))
        app._write_detail_line("backend", str(stats.get("backend", "lexical")))
        app._write_detail_line("mode", str(stats.get("mode_label", "")))
        app._write_detail_line("min_score_default", str(stats.get("min_score_default", "")))
        app._write_detail_line("load_recovery_count", str(stats.get("load_recovery_count", 0)))
        app._write_detail_line("backup_revisions", str(stats.get("backup_revisions", 0)))
        app._write_detail_line("by_type", json.dumps(stats["by_type"]))
        app._write("")
        return True
    return app._write_usage("/memory-stats")


def cmd_context(app: Any, arg: str) -> bool:
    if arg.strip():
        return app._write_usage("/context")
    used = app._context_tokens()
    total = app._context_window_tokens()
    percent = context_usage_percent(used, total)
    app._write_section_heading("Context")
    app._write_detail_line("usage", "—" if percent is None else f"{percent}%")
    if used is None or total is None:
        token_line = f"{'—' if used is None else used} / {'—' if total is None else total}"
    else:
        token_line = f"{used} / {total}"
    app._write_detail_line("tokens", token_line)
    app._write("")
    return True


def cmd_doctor(app: Any, *, accent_color: str) -> None:
    report = app.agent.doctor_report()
    app._write_section_heading("Doctor")
    agent = report.get("agent", {})
    workspace = report.get("workspace", {})
    memory = report.get("memory", {})
    search = report.get("search", {})
    app._write_detail_line("endpoint_ready", str(agent.get("ready", False)).lower())
    if agent.get("endpoint_policy_error"):
        app._write_detail_line("endpoint_policy", str(agent.get("endpoint_policy_error")))
    app._write_detail_line("workspace", str(workspace.get("path", "")))
    app._write_detail_line("workspace_writable", str(workspace.get("writable", False)).lower())
    app._write_detail_line("memory_backend", str(memory.get("backend", "lexical")))
    app._write_detail_line("memory_mode", str(memory.get("mode", "")))
    app._write_detail_line("memory_min_score_default", str(memory.get("min_score_default", "")))
    app._write_detail_line("memory_load_recovery_count", str(memory.get("load_recovery_count", 0)))
    app._write_detail_line("memory_backup_revisions", str(memory.get("backup_revisions", 0)))
    app._write_detail_line("search_provider", str(search.get("provider", "")))
    app._write_detail_line("search_ready", str(search.get("ready", False)).lower())
    if search.get("reason"):
        app._write_detail_line("search_reason", str(search.get("reason")))
    app._write_section_heading("Skills")
    for skill in report.get("skills", []):
        line = (
            f"  [bold {accent_color}]{esc(str(skill.get('id', '')))}[/bold {accent_color}] "
            f"[#a1a1aa]({esc(str(skill.get('provenance', '')))} · {esc(str(skill.get('availability_code', 'ready')))})[/#a1a1aa] "
            f"[#a1a1aa]{esc(str(skill.get('status', 'unknown')))}[/#a1a1aa]"
        )
        app._write(line)
        reason = str(skill.get("availability_reason", "")).strip()
        if reason and reason != "ready":
            app._write(f"    [#a1a1aa]{esc(reason)}[/#a1a1aa]")
        capabilities = _skill_capability_summary(
            execution_allowed=skill.get("execution_allowed", True),
            adapter=skill.get("adapter", "agentskills"),
            tools=skill.get("tools", []),
            scripts=skill.get("scripts", []),
            entrypoints=skill.get("entrypoints", []),
            user_invocable=skill.get("user_invocable", True),
            model_invocable=skill.get("model_invocable", True),
        )
        app._write(f"    [#71717a]{esc(capabilities)}[/#71717a]")
        _write_validation_error(app, skill.get("validation_errors", []))
    app._write("")


def cmd_report(app: Any, arg: str) -> bool:
    path = arg.strip() or "alphanus-support-report.json"
    payload = app.agent.build_support_bundle(app.conv_tree.to_dict())
    try:
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        app._write_error(f"Report save failed: {exc}")
        return True
    app._write_info(f"Saved support bundle to {path}")
    return True


def cmd_workspace(app: Any, arg: str) -> bool:
    sub = arg.strip().lower()
    if sub == "tree":
        tree = app.agent.skill_runtime.workspace.workspace_tree()
        app._write_section_heading("Workspace Tree")
        app._write_muted_lines(tree.splitlines())
        app._write("")
        return True
    return app._write_usage("/workspace-tree")


def cmd_code(app: Any, arg: str) -> bool:
    target = arg.strip().lower() or "last"
    if not app._code_blocks:
        app._write_error("No code blocks available yet")
        return True
    if target == "last":
        app._open_code_block(len(app._code_blocks))
        return True
    try:
        index = int(target)
    except ValueError:
        return app._write_usage("/code [n|last]")
    app._open_code_block(index)
    return True
