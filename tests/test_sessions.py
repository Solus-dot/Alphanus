from __future__ import annotations

from pathlib import Path

from core.conv_tree import ConvTree
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


def test_search_sessions_matches_content_and_turn_ids(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    tree = ConvTree()
    first = tree.add_turn("Investigate transcript virtualization")
    tree.complete_turn(first.id, "The viewport renderer should keep scroll stable.")
    tree.arm_branch("perf branch")
    second = tree.add_turn("Check health panel")
    second.skill_exchanges.append({"role": "tool", "name": "project_tree", "content": "{}"})
    tree.complete_turn(second.id, "done")
    session = store.create_session("Performance Notes", tree=tree)

    results = store.search_sessions("viewport")

    assert results
    assert results[0].session_id == session.id
    assert results[0].turn_id == first.id
    assert results[0].kind == "assistant"
    assert "viewport renderer" in results[0].preview

    tool_results = store.search_sessions("project_tree")
    assert tool_results[0].turn_id == second.id
    assert tool_results[0].kind == "tool"


def test_search_sessions_skips_failed_assistant_content(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    tree = ConvTree()
    failed = tree.add_turn("Look up the weather")
    tree.fail_turn(failed.id, "[agent error] Finalization failed: stale persisted failure")
    done = tree.add_turn("Summarize viewport")
    tree.complete_turn(done.id, "The viewport renderer should keep scroll stable.")
    store.create_session("Failure Replay", tree=tree)

    failed_results = store.search_sessions("stale persisted failure")
    done_results = store.search_sessions("viewport renderer")

    assert failed_results == []
    assert done_results
    assert done_results[0].kind == "assistant"


def test_search_sessions_title_match_does_not_target_root_turn(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    tree = ConvTree()
    turn = tree.add_turn("Keep this position")
    tree.complete_turn(turn.id, "done")
    tree.current_id = turn.id
    session = store.create_session("Architecture Notes", tree=tree)

    results = store.search_sessions("Architecture")

    assert results
    assert results[0].session_id == session.id
    assert results[0].kind == "title"
    assert results[0].turn_id == ""


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


def test_save_tree_roundtrip_preserves_collaboration_mode(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    session = store.bootstrap()

    saved = store.save_tree(
        session.id,
        session.title,
        session.tree,
        collaboration_mode="plan",
        created_at=session.created_at,
    )
    loaded = store.load_session(saved.id, activate=False)

    assert saved.collaboration_mode == "plan"
    assert loaded.collaboration_mode == "plan"


def test_save_tree_roundtrip_preserves_context_summary(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    session = store.bootstrap()

    saved = store.save_tree(
        session.id,
        session.title,
        session.tree,
        context_summary="Earlier turns edited src/app.py and left tests pending.",
        created_at=session.created_at,
    )
    loaded = store.load_session(saved.id, activate=False)

    assert loaded.context_summary == "Earlier turns edited src/app.py and left tests pending."


def test_session_store_loads_existing_session_without_version_gate(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    session = store.bootstrap()

    loaded = store.load_session(session.id, activate=False)

    assert loaded.id == session.id


def test_session_store_writes_manifest_without_version_gate(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    session = store.bootstrap()
    manifest_path = store.storage_dir / "manifest.json"

    loaded = store.bootstrap()
    disk = manifest_path.read_text(encoding="utf-8")

    assert loaded.id == session.id
    assert "active_session_id" in disk
