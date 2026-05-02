from __future__ import annotations

import os


def handle_command(app, text: str) -> bool:
    parts = text.strip().split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in {"/quit", "/exit", "/q"}:
        app.exit()
        return True

    if cmd in {"/shortcuts", "/keymap", "/keys"}:
        app._show_keyboard_shortcuts()
        return True

    if method := {
        "/help": "_cmd_help",
        "/details": "_toggle_tool_details",
        "/think": "_toggle_thinking_mode",
        "/tree": "_cmd_tree",
        "/skills": "_cmd_skills",
        "/doctor": "_cmd_doctor",
        "/config": "_open_config_editor",
    }.get(cmd):
        getattr(app, method)()
        return True

    if cmd == "/mode":
        if not arg:
            app._write_info(f"Collaboration mode: {str(getattr(app, '_collaboration_mode', 'execute'))}")
            return True
        target = arg.lower()
        if target not in {"plan", "execute"}:
            return app._write_usage("/mode [plan|execute]")
        previous = str(getattr(app, "_collaboration_mode", "execute")).strip().lower() or "execute"
        current = app._set_collaboration_mode(target, persist=True)
        (
            app._write_info(f"Collaboration mode already '{current}'.")
            if current == previous
            else app._write_command_action(f"Collaboration mode set to '{current}'", icon="✓")
        )
        return True

    if app.streaming and cmd in {"/sessions", "/clear"}:
        app._write_error("Stop the active response before changing sessions.")
        return True

    if cmd == "/sessions":
        return app._open_session_manager() or True

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
        app._write_command_action(f"Branch armed '{label}'", icon="⎇")
        app._update_status1()
        app._update_input_placeholder()
        return True

    if cmd == "/unbranch":
        if app.conv_tree._pending_branch:
            app.conv_tree.clear_pending_branch()
            app._save_active_session()
            app._write_command_action("Disarmed pending branch", icon="↩")
            for update in (app._update_status1, app._update_input_placeholder):
                update()
            return True
        moved = app.conv_tree.unbranch()
        if moved is None:
            app._write_error("No branch to leave.")
        else:
            app._save_active_session()
            app._write_command_action("Returned to fork point", icon="↩")
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

    if cmd == "/save":
        try:
            session = app._save_active_session(rename_to=arg or None)
            app._update_topbar()
            app._write_command_action(f"Saved session '{session.title}'", icon="✓")
        except OSError as exc:
            app._write_error(f"Save failed: {exc}")
        return True

    if cmd == "/clear":
        app.conv_tree = app._new_conv_tree()
        app._loaded_skill_ids = []
        app._reset_context_usage()
        app.pending.clear()
        app._log().clear_entries()
        app._set_partial_renderable(None)
        app._save_active_session()
        for update in (
            app._update_pending_attachments,
            app._update_status1,
            app._update_status2,
            app._update_sidebar,
            app._update_input_placeholder,
            app._update_topbar,
        ):
            update()
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

    if cmd == "/detach":

        def finish_detach(message: str) -> bool:
            app._update_pending_attachments()
            app._update_status1()
            app._write_command_action(message, icon="−")
            return True

        if not app.pending:
            app._write_info("No pending attachments.")
            return True

        if not arg:
            app._write_section_heading("Pending Attachments")
            app._write_muted_lines([f"{index}. {os.path.basename(path)}" for index, (path, _kind) in enumerate(app.pending, start=1)])
            app._write_info("Use /detach <n|last|all> to remove attachments.")
            return True

        target = arg.strip().lower()
        if target == "all":
            count = len(app.pending)
            app.pending.clear()
            return finish_detach(f"Removed {count} attachment{'s' if count != 1 else ''}")

        if target == "last":
            removed_path, _removed_kind = app.pending.pop()
            return finish_detach(f"Removed attachment '{os.path.basename(removed_path)}'")

        try:
            raw_index = int(target)
        except ValueError:
            app._write_error("/detach expects an index, 'last', or 'all'.")
            return True

        if raw_index <= 0:
            app._write_error("/detach index must be 1 or greater.")
            return True

        remove_index = raw_index - 1
        if remove_index >= len(app.pending):
            app._write_error(f"No pending attachment at index {raw_index}.")
            return True

        removed_path, _removed_kind = app.pending.pop(remove_index)
        return finish_detach(f"Removed attachment '{os.path.basename(removed_path)}'")

    if cmd == "/reload":
        return app._reload_skills()

    if cmd.startswith("/skill-"):
        action = cmd[7:]
        if action in {"on", "off", "unload", "info"}:
            return app._cmd_skill(f"{action} {arg}".strip())
        if action in {"unload-all", "reload"}:
            return app._cmd_skill(action)

    if cmd == "/memory-stats":
        return app._cmd_memory("stats")

    if cmd == "/context":
        return app._cmd_context(arg)

    if cmd == "/workspace-tree":
        return app._cmd_workspace("tree")

    if cmd == "/theme":
        return app._write_usage("/theme") if arg else (app._cmd_theme() or True)

    if cmd == "/report":
        return app._cmd_report(arg)

    if cmd == "/code":
        return app._cmd_code(arg)

    if cmd.startswith("/"):
        app._write_error(f"Unknown command: {cmd}")
        return True

    return False
