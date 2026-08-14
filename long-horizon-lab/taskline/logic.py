"""Domain logic – business operations on task collections."""

from __future__ import annotations

from taskline.domain import Task

# -- cycle detection -----------------------------------------------------------


def has_cycle(tasks: dict[int, Task], start: int) -> bool:
    """Return True if adding dependencies of *start* would create a cycle."""
    visited: set[int] = set()
    stack: list[int] = list(tasks[start].depends_on)
    while stack:
        node = stack.pop()
        if node == start:
            return True
        if node in visited:
            continue
        visited.add(node)
        dep_task = tasks.get(node)
        if dep_task:
            stack.extend(dep_task.depends_on)
    return False


# -- next-task selection -------------------------------------------------------


def get_next_tasks(tasks: dict[int, Task]) -> list[Task]:
    """Return unfinished, unblocked tasks sorted by priority (desc) then creation time (asc)."""
    pending = [t for t in tasks.values() if t.status == "pending"]
    unblocked = [t for t in pending if not t.is_blocked(tasks)]
    unblocked.sort(key=lambda t: (-t.priority, t.created_at))
    return unblocked


# -- removal helpers -----------------------------------------------------------


def find_dependents(tasks: dict[int, Task], task_id: int) -> list[int]:
    """Return IDs of tasks that depend on *task_id*."""
    return [t.id for t in tasks.values() if task_id in t.depends_on]


def clean_dependents(tasks: dict[int, Task], task_id: int) -> None:
    """Remove *task_id* from the depends_on list of any task that referenced it."""
    for t in tasks.values():
        t.depends_on = [d for d in t.depends_on if d != task_id]


# -- export --------------------------------------------------------------------


def export_markdown(tasks: dict[int, Task]) -> str:
    """Return a Markdown report grouped by status."""
    lines: list[str] = ["# Taskline Report\n"]
    groups: dict[str, list[Task]] = {}
    for t in tasks.values():
        groups.setdefault(t.status, []).append(t)

    for status in ("pending", "done"):
        group = groups.get(status, [])
        lines.append(f"## {status.capitalize()} ({len(group)})\n")
        if not group:
            lines.append("*No tasks.*\n")
            continue
        for t in sorted(group, key=lambda x: (-x.priority, x.created_at)):
            deps = ", ".join(f"`#{d}`" for d in t.depends_on) if t.depends_on else "—"
            tags = ", ".join(f"`{tag}`" for tag in t.tags) if t.tags else "—"
            lines.append(f"### `#{t.id}` {t.title}\n")
            lines.append(f"- **Priority:** {t.priority.name}")
            lines.append(f"- **Tags:** {tags}")
            lines.append(f"- **Depends on:** {deps}")
            if t.description:
                lines.append(f"- **Description:** {t.description}")
            lines.append(f"- **Created:** {t.created_at}")
            if t.completed_at:
                lines.append(f"- **Completed:** {t.completed_at}")
            lines.append("")
    return "\n".join(lines)
