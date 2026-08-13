from __future__ import annotations

import sqlite3
from pathlib import Path

from core.conv_tree import ConvTree
from core.sessions import SessionStore


def test_bootstrap_and_roundtrip_branching_state(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    session = store.bootstrap()
    turn = session.tree.add_turn("root")
    session.tree.complete_turn(turn.id, "done")
    saved = store.save_tree(session.id, "Refactor", session.tree, created_at=session.created_at)

    loaded = store.load_session(saved.id, activate=False)
    assert loaded.title == "Refactor"
    assert loaded.tree.nodes[turn.id].assistant_content == "done"
    assert store.list_sessions()[0].is_active


def test_bootstrap_can_defer_loading_the_active_tree(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    session = store.bootstrap()
    turn = session.tree.add_turn("existing")
    session.tree.complete_turn(turn.id, "reply")
    store.save_tree(session.id, session.title, session.tree)

    shell = store.bootstrap(load_tree=False)

    assert shell.id == session.id
    assert list(shell.tree.nodes) == ["root"]
    assert turn.id in store.load_session(shell.id).tree.nodes


def test_search_is_indexed_and_bounded(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    tree = ConvTree()
    turn = tree.add_turn("Investigate transcript virtualization")
    turn.skill_exchanges.append({"role": "tool", "name": "project_tree", "content": "{}"})
    tree.complete_turn(turn.id, "The viewport renderer should stay stable")
    session = store.create_session("Performance Notes", tree=tree)

    assert store.search_sessions("viewport", limit=1)[0].session_id == session.id
    assert store.search_sessions("project_tree", limit=1)[0].kind == "tool"
    assert len(store.search_sessions("viewport", limit=1)) == 1


def test_soft_delete_and_compaction(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    session = store.bootstrap()
    store.delete_session(session.id)
    assert store.list_sessions() == []
    assert store.compact(deleted_before="9999-01-01T00:00:00+00:00") == 1


def test_schema_and_wal_are_enabled(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    store.bootstrap()
    connection = sqlite3.connect(store.database_path)
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 3
    assert connection.execute("SELECT COUNT(*) FROM session_turns").fetchone()[0] == 2


def test_session_turn_updates_are_incremental(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    session = store.bootstrap()
    turn = session.tree.add_turn("hello")
    session.tree.complete_turn(turn.id, "first")
    store.save_tree(session.id, session.title, session.tree)
    with store._connection:
        store._connection.executescript(
            "CREATE TABLE write_audit(turn_id TEXT);"
            "CREATE TRIGGER audit_turn_update AFTER UPDATE ON session_turns "
            "BEGIN INSERT INTO write_audit VALUES (NEW.turn_id); END;"
        )
    session.tree.complete_turn(turn.id, "second")
    store.save_tree(session.id, session.title, session.tree)
    writes = store._connection.execute("SELECT turn_id FROM write_audit").fetchall()
    assert [row[0] for row in writes] == [turn.id]


def test_concurrent_store_connections_do_not_corrupt_state(tmp_path: Path) -> None:
    directory = tmp_path / "sessions"
    first = SessionStore(tmp_path, directory)
    second = SessionStore(tmp_path, directory)
    one = first.create_session("one")
    two = second.create_session("two")
    assert {item.id for item in first.list_sessions()} == {one.id, two.id}
    first.close()
    second.close()


def test_active_and_exact_sessions_remain_reachable_beyond_first_page(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    old = store.create_session("Old Active", activate=True)
    rows = [
        (f"bulk-{index:04d}", f"Bulk {index}", "9999-01-01", f"9999-01-{(index % 28) + 1:02d}")
        for index in range(1005)
    ]
    with store._connection:
        store._connection.executemany("INSERT INTO sessions(id,title,created_at,updated_at) VALUES (?,?,?,?)", rows)

    assert len(store.list_sessions()) == 1000
    assert len(store.list_sessions(limit=10, offset=1000)) == 6
    assert store.load_session(old.id, activate=False).title == "Old Active"
    assert store.bootstrap().id == old.id
