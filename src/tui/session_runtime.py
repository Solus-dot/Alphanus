from __future__ import annotations

from typing import Optional

from core.sessions import ChatSession


def activate_session_state(app, session: ChatSession) -> None:
    app._session_id = session.id
    app._session_title = session.title
    app._session_created_at = session.created_at
    agent = getattr(app, "agent", None)
    runtime = getattr(agent, "skill_runtime", None)
    if runtime and hasattr(runtime, "skills_by_ids"):
        app._loaded_skill_ids = [
            skill.id
            for skill in runtime.skills_by_ids(list(getattr(session, "loaded_skill_ids", []) or []))
        ]
    else:
        app._loaded_skill_ids = [
            str(item).strip()
            for item in (getattr(session, "loaded_skill_ids", []) or [])
            if str(item).strip()
        ]
    tree = app._apply_tree_compaction_policy(session.tree)
    if tree.current_id == "root" and tree.nodes["root"].children and not tree._pending_branch:
        for node_id in reversed(list(tree.nodes.keys())):
            if node_id != "root" and not tree.nodes[node_id].children:
                tree.current_id = node_id
                break
    app.conv_tree = tree
    app._tree_cursor_id = app.conv_tree.current_id


def save_active_session(app, rename_to: Optional[str] = None) -> ChatSession:
    title = (rename_to or app._session_title or "").strip() or app._session_title or "Untitled Session"
    loaded_skill_ids = list(getattr(app, "_loaded_skill_ids", []))
    try:
        session = app._session_store.save_tree(
            app._session_id,
            title,
            app.conv_tree,
            loaded_skill_ids=loaded_skill_ids,
            created_at=app._session_created_at,
            activate=True,
        )
    except TypeError:
        session = app._session_store.save_tree(
            app._session_id,
            title,
            app.conv_tree,
            created_at=app._session_created_at,
            activate=True,
        )
    app._session_title = session.title
    app._session_created_at = session.created_at
    return session


def current_session_is_blank(app) -> bool:
    return (
        app.conv_tree.current_id == "root"
        and len(app.conv_tree.nodes) == 1
        and not app.conv_tree.current.children
        and not app.conv_tree._pending_branch
    )


def open_new_session(app, title: str = "") -> ChatSession:
    normalized = title.strip()
    if current_session_is_blank(app):
        return app._save_active_session(rename_to=normalized or None)
    app._save_active_session()
    return app._session_store.create_session(normalized)


def load_session_from_manager(app, session_id: str) -> ChatSession:
    app._save_active_session()
    loaded = app._session_store.load_session(session_id)
    app._switch_to_session(loaded)
    return loaded


def delete_session_from_manager(app, session_id: str) -> None:
    try:
        active_id = app._session_id
        app._session_store.delete_session(session_id)
        deleted_active = session_id == active_id
        if deleted_active:
            remaining = app._session_store.list_sessions()
            if remaining:
                app._switch_to_session(app._session_store.load_session(remaining[0].id))
            else:
                app._switch_to_session(app._session_store.create_session())
        app._open_session_manager()
    except Exception as exc:
        app._write_error(f"Delete failed: {exc}")


def switch_to_session(app, session: ChatSession, *, clear_pending: bool = True) -> None:
    app._activate_session_state(session)
    app._reset_context_usage()
    if clear_pending:
        app.pending.clear()
    app._rebuild_viewport()
    app._update_sidebar()
    app._update_pending_attachments()
    app._update_status1()
    app._update_status2()
    app._update_input_placeholder()
    app._update_topbar()
