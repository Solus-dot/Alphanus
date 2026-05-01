from __future__ import annotations

from core.sessions import ChatSession


def activate_session_state(app, session: ChatSession) -> None:
    app._session_id = session.id
    app._session_title = session.title
    app._session_created_at = session.created_at
    app._collaboration_mode = "plan" if str(getattr(session, "collaboration_mode", "execute")).strip().lower() == "plan" else "execute"
    app._loaded_skill_ids = [skill.id for skill in app.agent.skill_runtime.skills_by_ids(list(session.loaded_skill_ids))]
    tree = app._apply_tree_compaction_policy(session.tree)
    if tree.current_id == "root" and tree.nodes["root"].children and not tree._pending_branch:
        newest_leaf = next((node_id for node_id in reversed(tree.nodes) if node_id != "root" and not tree.nodes[node_id].children), None)
        if newest_leaf:
            tree.current_id = newest_leaf
    app.conv_tree = tree
    app._tree_cursor_id = app.conv_tree.current_id


def save_active_session(app, rename_to: str | None = None) -> ChatSession:
    title = (rename_to or app._session_title or "").strip() or app._session_title or "Untitled Session"
    session = app._session_store.save_tree(
        app._session_id,
        title,
        app.conv_tree,
        loaded_skill_ids=list(getattr(app, "_loaded_skill_ids", [])),
        collaboration_mode=str(getattr(app, "_collaboration_mode", "execute")),
        created_at=app._session_created_at,
        activate=True,
    )
    app._session_title = session.title
    app._session_created_at = session.created_at
    app._collaboration_mode = "plan" if str(getattr(session, "collaboration_mode", "execute")).strip().lower() == "plan" else "execute"
    return session


def current_session_is_blank(app) -> bool:
    tree = app.conv_tree
    return tree.current_id == "root" and len(tree.nodes) == 1 and not tree.current.children and not tree._pending_branch


def open_new_session(app, title: str = "") -> ChatSession:
    if current_session_is_blank(app):
        return app._save_active_session(rename_to=title.strip() or None)
    app._save_active_session()
    return app._session_store.create_session(title.strip())


def load_session_from_manager(app, session_id: str) -> ChatSession:
    app._save_active_session()
    loaded = app._session_store.load_session(session_id)
    app._switch_to_session(loaded)
    return loaded


def delete_session_from_manager(app, session_id: str) -> None:
    try:
        app._session_store.delete_session(session_id)
        if session_id == app._session_id:
            remaining = app._session_store.list_sessions()
            app._switch_to_session(app._session_store.load_session(remaining[0].id) if remaining else app._session_store.create_session())
        app._open_session_manager()
    except (ValueError, OSError, KeyError) as exc:
        app._write_error(f"Delete failed: {exc}")


def switch_to_session(app, session: ChatSession, *, clear_pending: bool = True) -> None:
    app._activate_session_state(session)
    app._reset_context_usage()
    if clear_pending:
        app.pending.clear()
    app._rebuild_viewport()
    for update in (
        app._update_sidebar,
        app._update_pending_attachments,
        app._update_status1,
        app._update_status2,
        app._update_input_placeholder,
        app._update_topbar,
    ):
        update()
