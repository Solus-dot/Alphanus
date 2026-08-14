"""Domain model – Task dataclass and priority helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum


class Priority(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass
class Task:
    id: int
    title: str
    description: str = ""
    status: str = "pending"  # pending | done
    priority: Priority = Priority.MEDIUM
    tags: list[str] = field(default_factory=list)
    depends_on: list[int] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: str | None = None

    # -- helpers ----------------------------------------------------------------

    def is_blocked(self, tasks: dict[int, Task]) -> bool:
        """True if any unresolved dependency is still pending."""
        for dep_id in self.depends_on:
            dep = tasks.get(dep_id)
            if dep is not None and dep.status != "done":
                return True
        return False
