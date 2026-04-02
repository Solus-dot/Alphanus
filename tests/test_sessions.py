from __future__ import annotations

from pathlib import Path
from core.sessions import SessionStore


def test_bootstrap_creates_default_active_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")

    session = store.bootstrap()
    summaries = store.list_sessions()

    assert session.title == "Session 1"
    assert len(summaries) == 1
    assert summaries[0].title == "Session 1"
    assert summaries[0].is_active is True


def test_save_tree_roundtrip_preserves_branching_state(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    session = store.bootstrap()

    tree = session.tree
    root = tree.add_turn("root")
    tree.complete_turn(root.id, "root-done")
    tree.arm_branch("alt")
    branch = tree.add_turn("alternate")
    tree.complete_turn(branch.id, "alt-done")

    saved = store.save_tree(session.id, "Refactor Session", tree, created_at=session.created_at)
    loaded_by_id = store.load_session(saved.id, activate=False)
    loaded_by_title = store.load_session("Refactor Session", activate=False)
    loaded_by_index = store.load_session("1", activate=False)

    for loaded in (loaded_by_id, loaded_by_title, loaded_by_index):
        assert loaded.title == "Refactor Session"
        assert loaded.tree.current_id == branch.id
        assert loaded.tree.nodes[branch.id].branch_root is True
        assert loaded.tree.nodes[branch.id].assistant_content == "alt-done"


def test_bootstrap_skips_unreadable_fallback_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    first = store.create_session("Broken Session")
    second = store.create_session("Healthy Session")
    store._session_path(first.id).write_text("{", encoding="utf-8")

    bootstrapped = store.bootstrap()

    assert bootstrapped.id == second.id
    assert bootstrapped.title == "Healthy Session"


def test_delete_session_removes_file_and_manifest_entry(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    session = store.create_session("Delete Me")

    store.delete_session(session.id)

    assert not store._session_path(session.id).exists()
    assert all(summary.id != session.id for summary in store.list_sessions())


def test_delete_active_session_clears_manifest_active_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    session = store.bootstrap()

    store.delete_session(session.id)

    assert store._load_manifest()["active_session_id"] == ""
