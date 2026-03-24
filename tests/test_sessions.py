from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_import_tree_creates_new_active_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    original = store.bootstrap()
    tree = ConvTree()
    turn = tree.add_turn("hello")
    tree.complete_turn(turn.id, "world")
    export_path = store.export_session_tree("Importable Session", tree)
    imported = store.import_tree(export_path)

    summaries = store.list_sessions()

    assert imported.id != original.id
    assert imported.title == "Importable Session"
    imported_summary = next(summary for summary in summaries if summary.id == imported.id)
    assert imported_summary.is_active is True
    assert imported.tree.nodes[turn.id].assistant_content == "world"


def test_import_tree_rejects_legacy_raw_tree_exports(tmp_path: Path) -> None:
    legacy_export = tmp_path / "legacy-export.json"
    tree = ConvTree()
    turn = tree.add_turn("hello")
    tree.complete_turn(turn.id, "world")
    legacy_export.write_text(json.dumps(tree.to_dict()), encoding="utf-8")

    store = SessionStore(tmp_path, tmp_path / "sessions")

    with pytest.raises(ValueError, match="missing 'tree'"):
        store.import_tree(legacy_export)


def test_export_roundtrip_uses_workspace_exports_folder(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    tree = ConvTree()
    turn = tree.add_turn("hello")
    tree.complete_turn(turn.id, "world")

    path = store.export_session_tree("Test Export", tree)
    exports = store.list_exports()
    imported = store.import_tree(path, activate=False)

    assert path.parent == tmp_path / ".alphanus" / "exports"
    assert exports[0].title == "Test Export"
    assert exports[0].filename == path.name
    assert imported.tree.nodes[turn.id].assistant_content == "world"


def test_bootstrap_skips_unreadable_fallback_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path, tmp_path / "sessions")
    first = store.create_session("Broken Session")
    second = store.create_session("Healthy Session")
    store._session_path(first.id).write_text("{", encoding="utf-8")

    bootstrapped = store.bootstrap()

    assert bootstrapped.id == second.id
    assert bootstrapped.title == "Healthy Session"
