from __future__ import annotations


def handle_command(app, text: str) -> bool:
    parts = text.strip().split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in {"/quit", "/exit", "/q"}:
        app.exit()
        return True

    if cmd == "/help":
        app._cmd_help()
        return True

    if cmd in {"/keyboard-shortcuts", "/shortcuts", "/keymap", "/keys"}:
        app._show_keyboard_shortcuts()
        return True

    if cmd == "/details":
        app._toggle_tool_details()
        return True

    if cmd == "/think":
        app._toggle_thinking_mode()
        return True

    if cmd == "/sessions":
        if app.streaming:
            app._write_error("Stop the active response before changing sessions.")
            return True
        app._cmd_sessions()
        return True

    if app.streaming and cmd == "/clear":
        app._write_error("Stop the active response before changing sessions.")
        return True

    if cmd == "/new":
        app._write_error("Use /sessions to manage sessions.")
        return True

    if cmd == "/rename":
        if not arg:
            return app._write_usage("/rename <name>")
        session = app._save_active_session(rename_to=arg)
        app._update_topbar()
        app._write_command_action(f"Renamed session to '{session.title}'", icon="✓")
        return True

    if cmd == "/branch":
        app.conv_tree.arm_branch(arg)
        app._save_active_session()
        label = app.conv_tree._pending_branch_label
        app._write_command_action(f"Branch armed '{label}'", icon="⎇", color="#6366f1")
        app._update_status1()
        app._update_input_placeholder()
        return True

    if cmd == "/unbranch":
        if app.conv_tree._pending_branch:
            app.conv_tree.clear_pending_branch()
            app._save_active_session()
            app._write_command_action("Disarmed pending branch", icon="↩", color="#6366f1")
            app._update_status1()
            app._update_input_placeholder()
            return True
        moved = app.conv_tree.unbranch()
        if moved is None:
            app._write_error("No branch to leave.")
        else:
            app._save_active_session()
            app._write_command_action("Returned to fork point", icon="↩", color="#6366f1")
            app._rebuild_viewport()
            app._update_sidebar()
        app._update_status1()
        return True

    if cmd == "/branches":
        children = app.conv_tree.current.children
        if not children:
            app._write_info("No child branches from current turn.")
        else:
            app._write_section_heading("Children")
            app._write_indexed_dim_lines([app.conv_tree.nodes[cid].short(60) for cid in children])
        return True

    if cmd == "/switch":
        try:
            idx = int(arg)
        except ValueError:
            app._write_error("/switch requires an integer index")
            return True
        turn = app.conv_tree.switch_child(idx)
        if not turn:
            app._write_error(f"No child {idx} at current node")
        else:
            app._save_active_session()
            app._write_command_action(f"Switched to branch {idx}", icon="↪")
            app._rebuild_viewport()
            app._update_sidebar()
        return True

    if cmd == "/tree":
        app._cmd_tree()
        return True

    if cmd == "/save":
        try:
            session = app._save_active_session(rename_to=arg or None)
            app._update_topbar()
            app._write_command_action(f"Saved session '{session.title}'", icon="✓")
        except Exception as exc:
            app._write_error(f"Save failed: {exc}")
        return True

    if cmd == "/load":
        app._write_error("Use /sessions to manage sessions.")
        return True

    if cmd == "/clear":
        app.conv_tree = app._new_conv_tree()
        app._loaded_skill_ids = []
        app._reset_context_usage()
        app.pending.clear()
        app._log().clear_entries()
        app._set_partial_renderable(None)
        app._save_active_session()
        app._update_pending_attachments()
        app._update_status1()
        app._update_status2()
        app._update_sidebar()
        app._update_input_placeholder()
        app._update_topbar()
        return True

    if cmd in {"/file", "/image"}:
        if not arg:
            app._open_attachment_picker(".")
            return True
        try:
            path = app._resolve_attachment_path(arg)
        except FileNotFoundError:
            app._write_error(f"File not found: {arg}")
            return True
        app._attach_file_path(path)
        return True

    if cmd == "/skills":
        app._cmd_skills()
        return True

    if cmd == "/reload":
        return app._reload_skills()

    if cmd == "/doctor":
        app._cmd_doctor()
        return True

    if cmd == "/skill-on":
        return app._cmd_skill(f"on {arg}".strip())

    if cmd == "/skill-off":
        return app._cmd_skill(f"off {arg}".strip())

    if cmd == "/skill-unload":
        return app._cmd_skill(f"unload {arg}".strip())

    if cmd == "/skill-unload-all":
        return app._cmd_skill("unload-all")

    if cmd == "/skill-reload":
        return app._cmd_skill("reload")

    if cmd == "/skill-info":
        return app._cmd_skill(f"info {arg}".strip())

    if cmd == "/memory-stats":
        return app._cmd_memory("stats")

    if cmd == "/context":
        return app._cmd_context(arg)

    if cmd == "/workspace-tree":
        return app._cmd_workspace("tree")

    if cmd == "/config":
        app._open_config_editor()
        return True

    if cmd == "/report":
        return app._cmd_report(arg)

    if cmd == "/code":
        return app._cmd_code(arg)

    if cmd.startswith("/"):
        app._write_error(f"Unknown command: {cmd}")
        return True

    return False
