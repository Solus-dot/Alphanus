"""Persistence layer – read/write taskline.json."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

from taskline.domain import Priority, Task


class StorageError(Exception):
    """Raised when persistence operations fail."""


def _resolve_path(path: str | None = None) -> Path:
    if path:
        return Path(path)
    return Path.cwd() / "taskline.json"


def load_tasks(path: str | None = None) -> dict[int, Task]:
    """Load tasks from JSON file. Returns empty dict if file doesn't exist."""
    store_path = _resolve_path(path)
    if not store_path.exists():
        return {}
    try:
        with open(store_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageError(f"Failed to read {store_path}: {exc}") from exc

    tasks: dict[int, Task] = {}
    for item in data.get("tasks", []):
        task = Task(
            id=item["id"],
            title=item["title"],
            description=item.get("description", ""),
            status=item.get("status", "pending"),
            priority=Priority(item.get("priority", 1)),
            tags=item.get("tags", []),
            depends_on=item.get("depends_on", []),
            created_at=item.get("created_at", datetime.utcnow().isoformat()),
            completed_at=item.get("completed_at"),
        )
        tasks[task.id] = task
    return tasks


def save_tasks(tasks: dict[int, Task], path: str | None = None) -> None:
    """Persist tasks to JSON file."""
    store_path = _resolve_path(path)
    sorted_tasks = sorted(tasks.values(), key=lambda t: t.id)
    data = {
        "tasks": [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "status": t.status,
                "priority": int(t.priority),
                "tags": t.tags,
                "depends_on": t.depends_on,
                "created_at": t.created_at,
                "completed_at": t.completed_at,
            }
            for t in sorted_tasks
        ]
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=store_path.parent, delete=False) as file:
            temporary = Path(file.name)
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.flush()
        temporary.replace(store_path)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise StorageError(f"Failed to write {store_path}: {exc}") from exc


def next_id(tasks: dict[int, Task]) -> int:
    """Return the next available integer ID."""
    if not tasks:
        return 1
    return max(tasks.keys()) + 1
