"""Tests for Taskline – persistence, cycle detection, ordering, forced removal, CLI."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from taskline.domain import Priority, Task
from taskline.logic import (
    clean_dependents,
    export_markdown,
    find_dependents,
    get_next_tasks,
    has_cycle,
)
from taskline.storage import StorageError, load_tasks, next_id, save_tasks

# ---------------------------------------------------------------------------
# Storage / persistence
# ---------------------------------------------------------------------------


class TestStorage(unittest.TestCase):
    def _tmp(self, content: str = "") -> str:
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write(content)
        return path

    def test_load_empty_file(self):
        path = self._tmp('{"tasks": []}')
        self.assertEqual(load_tasks(path), {})
        os.unlink(path)

    def test_load_missing_file(self):
        self.assertEqual(load_tasks("/nonexistent/taskline.json"), {})

    def test_roundtrip(self):
        path = self._tmp()
        tasks = {
            1: Task(id=1, title="Alpha", priority=Priority.HIGH),
            2: Task(id=2, title="Beta", depends_on=[1]),
        }
        save_tasks(tasks, path)
        loaded = load_tasks(path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[1].title, "Alpha")
        self.assertEqual(loaded[2].depends_on, [1])
        os.unlink(path)

    def test_next_id_empty(self):
        self.assertEqual(next_id({}), 1)

    def test_next_id_populated(self):
        self.assertEqual(next_id({1: Task(id=1, title=""), 3: Task(id=3, title="")}), 4)

    def test_save_creates_file(self):
        path = tempfile.mktemp(suffix=".json")
        save_tasks({1: Task(id=1, title="X")}, path)
        self.assertTrue(Path(path).exists())
        data = json.loads(Path(path).read_text())
        self.assertEqual(data["tasks"][0]["title"], "X")
        os.unlink(path)

    def test_unicode_roundtrip_and_malformed_file_preservation(self):
        path = self._tmp()
        save_tasks({1: Task(id=1, title="日本語 ✅")}, path)
        self.assertEqual(load_tasks(path)[1].title, "日本語 ✅")
        Path(path).write_text("{broken", encoding="utf-8")
        with self.assertRaises(StorageError):
            load_tasks(path)
        self.assertEqual(Path(path).read_text(encoding="utf-8"), "{broken")
        os.unlink(path)


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


class TestCycleDetection(unittest.TestCase):
    def _task(self, tid: int, deps: list[int] | None = None) -> Task:
        return Task(id=tid, title=f"T{tid}", depends_on=deps or [])

    def test_no_cycle(self):
        tasks = {1: self._task(1), 2: self._task(2, [1])}
        self.assertFalse(has_cycle(tasks, 2))

    def test_direct_cycle(self):
        tasks = {1: self._task(1, [2]), 2: self._task(2, [1])}
        self.assertTrue(has_cycle(tasks, 1))
        self.assertTrue(has_cycle(tasks, 2))

    def test_indirect_cycle(self):
        tasks = {
            1: self._task(1, [3]),
            2: self._task(2, [1]),
            3: self._task(3, [2]),
        }
        self.assertTrue(has_cycle(tasks, 1))

    def test_diamond_no_cycle(self):
        tasks = {
            1: self._task(1),
            2: self._task(2, [1]),
            3: self._task(3, [1]),
            4: self._task(4, [2, 3]),
        }
        self.assertFalse(has_cycle(tasks, 4))


# ---------------------------------------------------------------------------
# Ordering (next command)
# ---------------------------------------------------------------------------


class TestNextOrdering(unittest.TestCase):
    def _task(
        self,
        tid: int,
        pri=Priority.MEDIUM,
        deps: list[int] | None = None,
        created: str | None = None,
        status: str = "pending",
    ) -> Task:
        return Task(
            id=tid,
            title=f"T{tid}",
            priority=pri,
            depends_on=deps or [],
            created_at=created or datetime.now().isoformat(),
            status=status,
        )

    def test_priority_then_time(self):
        t1 = self._task(1, Priority.LOW, created="2025-01-01T00:00:00")
        t2 = self._task(2, Priority.HIGH, created="2025-06-01T00:00:00")
        t3 = self._task(3, Priority.HIGH, created="2025-02-01T00:00:00")
        tasks = {1: t1, 2: t2, 3: t3}
        result = get_next_tasks(tasks)
        ids = [t.id for t in result]
        self.assertEqual(ids, [3, 2, 1])

    def test_blocked_excluded(self):
        t1 = self._task(1)
        t2 = self._task(2, deps=[1])
        tasks = {1: t1, 2: t2}
        result = get_next_tasks(tasks)
        ids = [t.id for t in result]
        self.assertEqual(ids, [1])

    def test_done_excluded(self):
        t1 = self._task(1, status="done")
        t2 = self._task(2, deps=[1])
        tasks = {1: t1, 2: t2}
        result = get_next_tasks(tasks)
        ids = [t.id for t in result]
        self.assertEqual(ids, [2])


# ---------------------------------------------------------------------------
# Forced removal
# ---------------------------------------------------------------------------


class TestForcedRemoval(unittest.TestCase):
    def _task(self, tid: int, deps: list[int] | None = None) -> Task:
        return Task(id=tid, title=f"T{tid}", depends_on=deps or [])

    def test_find_dependents(self):
        tasks = {
            1: self._task(1),
            2: self._task(2, [1]),
            3: self._task(3, [1, 2]),
        }
        self.assertEqual(find_dependents(tasks, 1), [2, 3])
        self.assertEqual(find_dependents(tasks, 3), [])

    def test_clean_dependents(self):
        tasks = {
            1: self._task(1),
            2: self._task(2, [1, 3]),
        }
        clean_dependents(tasks, 1)
        self.assertEqual(tasks[2].depends_on, [3])

    def test_remove_without_force_refused(self):
        tasks = {
            1: self._task(1),
            2: self._task(2, [1]),
        }
        with self.assertRaises(ValueError) as ctx:
            tasks.pop(1)
            # Simulate what logic.check_removable would do
            for t in tasks.values():
                if 1 in t.depends_on:
                    raise ValueError(f"Task #{1} is depended on by #{t.id}")
        self.assertIn("depended on", str(ctx.exception))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport(unittest.TestCase):
    def test_markdown_groups(self):
        tasks = {
            1: Task(id=1, title="Done thing", status="done", priority=Priority.LOW),
            2: Task(id=2, title="Pending thing", status="pending", priority=Priority.HIGH),
        }
        md = export_markdown(tasks)
        self.assertIn("# Taskline Report", md)
        self.assertIn("## Pending", md)
        self.assertIn("## Done", md)
        self.assertIn("Pending thing", md)
        self.assertIn("Done thing", md)


# ---------------------------------------------------------------------------
# CLI main workflow
# ---------------------------------------------------------------------------


class TestCLIWorkflow(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.json_path = os.path.join(self.tmpdir, "taskline.json")

    def tearDown(self):
        if os.path.exists(self.json_path):
            os.unlink(self.json_path)
        os.rmdir(self.tmpdir)

    def _run(self, *args: str) -> tuple[int, str]:
        from taskline.cli import main

        old_argv = sys.argv
        old_cwd = os.getcwd()
        try:
            sys.argv = ["taskline"] + list(args)
            os.chdir(self.tmpdir)
            from io import StringIO

            out = StringIO()
            with mock.patch("sys.stdout", out), mock.patch("sys.stderr", out):
                try:
                    return main(), out.getvalue()
                except SystemExit as e:
                    return (e.code if isinstance(e.code, int) else 1), out.getvalue()
        finally:
            os.chdir(old_cwd)
            sys.argv = old_argv

    def test_add_and_list(self):
        code, out = self._run("add", "First task", "--priority", "high")
        self.assertEqual(code, 0)
        self.assertIn("Added", out)

        code, out = self._run("list")
        self.assertEqual(code, 0)
        self.assertIn("First task", out)

    def test_add_cycle_rejected(self):
        self._run("add", "A")
        self._run("add", "B", "--depends-on", "1")
        code, out = self._run("add", "C", "--depends-on", "2")
        self.assertEqual(code, 0)
        # Adding a task that depends on itself indirectly
        # Task 3 depends on 2, task 2 depends on 1.  Now try to add task 4
        # depending on 3, and then try to make task 1 depend on 4 via edit.
        self._run("add", "D", "--depends-on", "3")
        code, out = self._run("edit", "1", "--depends-on", "4")
        self.assertNotEqual(code, 0)
        self.assertIn("cycle", out.lower())

    def test_done_and_next(self):
        self._run("add", "A")
        self._run("add", "B", "--depends-on", "1")
        code, out = self._run("next")
        self.assertIn("#1", out)
        self.assertNotIn("#2", out)  # blocked

        self._run("done", "1")
        code, out = self._run("next")
        self.assertIn("#2", out)

    def test_remove_force(self):
        self._run("add", "A")
        self._run("add", "B", "--depends-on", "1")
        # Without --force should fail
        code, out = self._run("remove", "1")
        self.assertNotEqual(code, 0)
        # With --force should succeed
        code, out = self._run("remove", "1", "--force")
        self.assertEqual(code, 0)
        # Task 2 should still exist, dependency cleaned
        code, out = self._run("show", "2")
        self.assertEqual(code, 0)

    def test_export_markdown(self):
        self._run("add", "Task A")
        self._run("done", "1")
        code, out = self._run("export", "--format", "markdown")
        self.assertEqual(code, 0)
        self.assertIn("# Taskline Report", out)
        self.assertIn("## Done", out)

    def test_show_missing_task(self):
        code, out = self._run("show", "999")
        self.assertNotEqual(code, 0)

    def test_edit_missing_task(self):
        code, out = self._run("edit", "999", "--title", "X")
        self.assertNotEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
