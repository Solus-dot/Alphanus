from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.file_audit import build_file_audit_from_skill_exchanges
from core.conv_tree import ConvTree
from core.secure_io import atomic_write_text

COMMAND_SECTIONS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "CONVERSATION",
        (
            ("/help", "Show this help"),
            ("/shortcuts", "Show keyboard shortcuts"),
            ("/details", "Toggle tool execution details"),
            ("/think", "Toggle thinking mode"),
            ("/mode [plan|execute]", "Show or set collaboration mode"),
            ("/clear", "Clear the active conversation"),
            ("/sessions", "Open sessions"),
            ("/rename <name>", "Rename the active session"),
            ("/save [name]", "Save the active session"),
            ("/file [path]", "Attach a file or open the picker"),
            ("/detach [n|last|all]", "Remove pending attachments"),
            ("/quit", "Exit Alphanus"),
        ),
    ),
    (
        "BRANCHING",
        (
            ("/branch [label]", "Arm the next message as a branch"),
            ("/unbranch", "Return to the nearest fork"),
            ("/branches", "List child branches"),
            ("/switch <n>", "Switch to a child branch"),
            ("/tree", "Show the conversation tree"),
        ),
    ),
    (
        "SKILLS",
        (
            ("/skills", "List installed skills"),
            ("/reload", "Reload skills"),
            ("/doctor", "Run readiness diagnostics"),
            ("/health", "Open project health"),
            ("/skill-on <id>", "Enable a skill"),
            ("/skill-off <id>", "Disable a skill"),
            ("/skill-unload <id>", "Unload a session skill"),
            ("/skill-unload-all", "Unload all session skills"),
            ("/skill-reload", "Reload skills"),
            ("/skill-info <id>", "Show skill details"),
        ),
    ),
    (
        "UTILITIES",
        (
            ("/memory-stats", "Show memory statistics"),
            ("/context", "Show context usage"),
            ("/audit", "Show turn file changes"),
            ("/project-tree", "Show the project tree"),
            ("/theme", "Open the theme picker"),
            ("/config", "Edit configuration"),
            ("/report [file]", "Save a support report"),
            ("/code [n|last]", "Open a code block"),
        ),
    ),
)

SHORTCUT_SECTIONS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "KEYMAP",
        (
            ("F1 / ?", "Show keyboard shortcuts"),
            ("Ctrl+P", "Open command palette"),
            ("Ctrl+K", "Open global palette"),
            ("Ctrl+F", "Open file picker"),
            ("Ctrl+B", "Toggle conversation tree"),
            ("Ctrl+G / Ctrl+H / Ctrl+L", "Focus composer, transcript, or tree"),
            ("Tab / Shift+Tab", "Cycle active panels"),
            ("F2 / F3", "Toggle tool details or thinking mode"),
            ("Ctrl+C / Ctrl+D", "Quit Alphanus"),
        ),
    ),
    ("TRANSCRIPT", (("PgUp / PgDn / wheel", "Scroll transcript"),)),
    (
        "TREE",
        (
            ("j / k", "Move selection"),
            ("Enter / o", "Open selected node"),
            ("[ / ]", "Jump sibling branches"),
            ("g / G", "Jump top or bottom"),
        ),
    ),
    (
        "INPUT",
        (
            ("Enter", "Send message"),
            ("Esc", "Clear input or stop the active turn"),
            ("Ctrl+Backspace", "Remove the last attachment"),
            ("Ctrl+Shift+Backspace", "Clear attachments"),
            ("Ctrl+U", "Clear the draft"),
            ("Ctrl+Shift+K", "Delete to end of line"),
        ),
    ),
)


def command_catalog() -> list[dict[str, str]]:
    return [
        {"section": section, "command": command, "description": description}
        for section, rows in COMMAND_SECTIONS
        for command, description in rows
    ]


def shortcut_catalog() -> list[dict[str, str]]:
    return [
        {"section": section, "key": key, "description": description}
        for section, rows in SHORTCUT_SECTIONS
        for key, description in rows
    ]


def palette_command_catalog() -> list[dict[str, str]]:
    return [
        dict(kind="command", value=row["command"].split()[0], prompt=row["command"], description=row["description"])
        for row in command_catalog()
    ]


def _result(*lines: str, ok: bool = True, action: str = "", state_changed: bool = False, **extra: Any) -> dict[str, Any]:
    return {"ok": ok, "lines": list(lines), "action": action, "state_changed": state_changed, **extra}


def execute_ui_command(server: Any, raw: str) -> dict[str, Any]:
    text = raw.strip()
    parts = text.split(None, 1)
    cmd = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""
    tree = server.session.tree

    if cmd == "/help":
        lines: list[str] = []
        for section, rows in COMMAND_SECTIONS:
            lines.append(section)
            lines.extend(f"  {command:<26} {description}" for command, description in rows)
            lines.append("")
        return _result(*lines, action="help")
    if cmd in {"/shortcuts", "/keymap", "/keys"}:
        lines = []
        for section, rows in SHORTCUT_SECTIONS:
            lines.append(section)
            lines.extend(f"  {key:<28} {description}" for key, description in rows)
            lines.append("")
        return _result(*lines, action="shortcuts")
    if cmd in {"/quit", "/exit", "/q"}:
        return _result(action="quit")
    if cmd == "/sessions":
        return _result(action="sessions")
    if cmd == "/theme":
        return _result(action="theme")
    if cmd == "/config":
        return _result(action="config")
    if cmd == "/health":
        return _result(action="health", report=server.agent.doctor_report(probe_ready=False))
    if cmd == "/details":
        return _result(action="toggle_details")
    if cmd == "/think":
        return _result(action="toggle_thinking")
    if cmd == "/mode":
        if not arg:
            return _result(f"Collaboration mode: {server.session.collaboration_mode}")
        if arg.lower() not in {"plan", "execute"}:
            return _result("Usage: /mode [plan|execute]", ok=False)
        server.session.collaboration_mode = arg.lower()
        server._save()
        return _result(f"Collaboration mode set to '{arg.lower()}'", state_changed=True)
    if cmd == "/rename":
        if not arg:
            return _result("Usage: /rename <name>", ok=False)
        if len(arg) > 200:
            return _result("Session title must be at most 200 characters", ok=False)
        server._save(title=arg)
        return _result(f"Renamed session to '{arg}'", state_changed=True)
    if cmd == "/save":
        if len(arg) > 200:
            return _result("Session title must be at most 200 characters", ok=False)
        server._save(title=arg or None)
        return _result(f"Saved session '{server.session.title}'", state_changed=True)
    if cmd == "/clear":
        server._require_idle()
        server.session.tree = ConvTree()
        server.session.loaded_skill_ids = []
        server.pending_attachments.clear()
        server._save()
        return _result("Conversation cleared", action="clear", state_changed=True)
    if cmd == "/branch":
        tree.arm_branch(arg)
        server._save()
        return _result(f"Branch armed '{tree._pending_branch_label}'", state_changed=True)
    if cmd == "/unbranch":
        if tree._pending_branch:
            tree.clear_pending_branch()
            message = "Disarmed pending branch"
        elif tree.unbranch() is not None:
            message = "Returned to fork point"
        else:
            return _result("No branch to leave", ok=False)
        server._save()
        return _result(message, state_changed=True)
    if cmd == "/branches":
        children = tree.current.children
        lines = [f"{index + 1}. {tree.nodes[node_id].short(60)}" for index, node_id in enumerate(children)]
        return _result(*(lines or ["No child branches"]))
    if cmd == "/switch":
        try:
            index = int(arg) - 1
        except ValueError:
            return _result("/switch requires an integer index", ok=False)
        if tree.switch_child(index) is None:
            return _result(f"No child {index + 1} at current node", ok=False)
        server._save()
        return _result(f"Switched to branch {index + 1}", state_changed=True)
    if cmd == "/tree":
        lines = []
        for node in tree.nodes.values():
            if node.id == "root":
                continue
            marker = "●" if node.id == tree.current_id else "○"
            lines.append(f"{marker} {node.label + ': ' if node.label else ''}{node.short(72)}")
        return _result(*(lines or ["No turns yet"]))
    if cmd in {"/file", "/image"}:
        return _result(action="file_picker") if not arg else _result(action="attach", path=arg)
    if cmd == "/detach":
        if not arg:
            lines = [f"{index + 1}. {Path(path).name} ({kind})" for index, (path, kind) in enumerate(server.pending_attachments)]
            return _result(*(lines or ["No pending attachments"]))
        detach_target: str | int = arg.lower()
        if detach_target not in {"last", "all"}:
            try:
                detach_target = int(detach_target) - 1
            except ValueError:
                return _result("Usage: /detach [n|last|all]", ok=False)
            if detach_target < 0:
                return _result("Attachment index must be at least 1", ok=False)
        return _result(action="detach", target=detach_target)
    if cmd in {"/reload", "/skill-reload"}:
        count = server.agent.reload_skills()
        return _result(f"Reloaded {count} skills", state_changed=True)
    if cmd == "/skills":
        loaded = set(server.session.loaded_skill_ids)
        lines = [
            f"{skill.id} v{skill.version} · {'loaded' if skill.id in loaded else 'available'} · {skill.description}"
            for skill in server.agent.skill_runtime.list_skills()
        ]
        return _result(*(lines or ["No skills installed"]))
    if cmd.startswith("/skill-"):
        action = cmd[7:]
        if action == "unload-all":
            server.session.loaded_skill_ids = []
            server._save()
            return _result("Unloaded all session skills", state_changed=True)
        if not arg:
            return _result(f"Usage: {cmd} <id>", ok=False)
        skill = server.agent.skill_runtime.get_skill(arg)
        if not skill:
            return _result(f"Skill not found: {arg}", ok=False)
        if action in {"on", "off"}:
            server.agent.skill_runtime.set_enabled(arg, action == "on")
            if action == "off":
                server.session.loaded_skill_ids = [item for item in server.session.loaded_skill_ids if item != arg]
            server._save()
            return _result(f"Skill {arg} {'enabled' if action == 'on' else 'disabled'}", state_changed=True)
        if action == "unload":
            server.session.loaded_skill_ids = [item for item in server.session.loaded_skill_ids if item != arg]
            server._save()
            return _result(f"Unloaded skill {arg}", state_changed=True)
        if action == "info":
            return _result(
                skill.name,
                skill.description,
                f"id: {skill.id}",
                f"version: {skill.version}",
                f"source: {server.agent.skill_runtime.skill_source_label(skill)}",
                f"available: {str(skill.available).lower()}",
                f"reason: {skill.availability_reason or 'ready'}",
            )
    if cmd == "/memory-stats":
        stats = server.agent.skill_runtime.memory.stats()
        return _result(*(f"{key}: {json.dumps(value) if isinstance(value, (dict, list)) else value}" for key, value in stats.items()))
    if cmd == "/context":
        report = getattr(server.agent, "last_context_report", None) or {}
        lines = [f"{key}: {value}" for key, value in report.items()]
        return _result(*(lines or ["Context usage is not available yet"]))
    if cmd == "/doctor":
        report = server.agent.doctor_report(probe_ready=False)
        return _result(json.dumps(report, indent=2, ensure_ascii=False))
    if cmd == "/project-tree":
        root = server.agent.skill_runtime.project.project_root
        project_rows: list[str] = []
        for path in sorted(root.rglob("*")):
            if len(project_rows) >= 500:
                project_rows.append("… project tree truncated")
                break
            if any(part in {".git", ".venv", "node_modules", "__pycache__"} for part in path.parts):
                continue
            project_rows.append(str(path.relative_to(root)) + ("/" if path.is_dir() else ""))
        return _result(*(project_rows or ["Project is empty"]))
    if cmd == "/audit":
        rows = build_file_audit_from_skill_exchanges(
            tree.current.skill_exchanges,
            project_root=server.agent.skill_runtime.project.project_root,
        )
        return _result(json.dumps(rows, indent=2, ensure_ascii=False) if rows else "No file changes recorded for the current turn")
    if cmd == "/report":
        root = Path(server.agent.skill_runtime.project.project_root)
        target = Path(arg).expanduser() if arg else root / "alphanus-support.json"
        if not target.is_absolute():
            target = root / target
        target = target.resolve()
        if not target.is_relative_to(root.resolve()):
            return _result("Support reports must be written inside the workspace", ok=False)
        bundle = server.agent.build_support_bundle(tree.to_dict())
        atomic_write_text(target, json.dumps(bundle, indent=2, ensure_ascii=False) + "\n")
        return _result(f"Saved support report to {target}")
    if cmd == "/code":
        return _result(action="code", target=arg or "last")
    if cmd.startswith("/"):
        return _result(f"Unknown command: {cmd}", ok=False)
    return _result("Not a command", ok=False)
