"""CLI interface – argparse-based command handler."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from taskline.domain import Priority, Task
from taskline.logic import (
    clean_dependents,
    export_markdown,
    find_dependents,
    get_next_tasks,
    has_cycle,
)
from taskline.storage import StorageError, load_tasks, next_id, save_tasks

# -- sub-command handlers -------------------------------------------------------


def _cmd_add(args: argparse.Namespace, tasks: dict[int, Task]) -> int:
    new_id = next_id(tasks)
    dep_ids = [int(d) for d in args.depends_on] if args.depends_on else []
    tag_list = [t.strip() for t in args.tags.split(",")] if args.tags else []
    pri = Priority[args.priority.upper()] if args.priority else Priority.MEDIUM

    # Validate dependencies exist
    missing = [d for d in dep_ids if d not in tasks]
    if missing:
        print(f"Error: dependency task(s) {missing} do not exist.", file=sys.stderr)
        return 1

    # Temporarily insert to check for cycles
    tasks[new_id] = Task(
        id=new_id,
        title=args.title,
        description=args.description or "",
        priority=pri,
        tags=tag_list,
        depends_on=dep_ids,
    )
    if has_cycle(tasks, new_id):
        del tasks[new_id]
        print("Error: adding this task would create a dependency cycle.", file=sys.stderr)
        return 1

    save_tasks(tasks, args.data)
    print(f"Added task #{new_id}: {args.title}")
    return 0


def _cmd_list(args: argparse.Namespace, tasks: dict[int, Task]) -> int:
    if not tasks:
        print("No tasks.")
        return 0
    for t in sorted(tasks.values(), key=lambda x: x.id):
        blocked = " [blocked]" if t.is_blocked(tasks) else ""
        print(f"  #{t.id} [{t.status}] {t.title} (P{int(t.priority)}){blocked}")
    return 0


def _cmd_show(args: argparse.Namespace, tasks: dict[int, Task]) -> int:
    tid = int(args.task_id)
    t = tasks.get(tid)
    if not t:
        print(f"Error: task #{tid} not found.", file=sys.stderr)
        return 1
    blocked = "Yes" if t.is_blocked(tasks) else "No"
    print(f"  #{t.id} {t.title}")
    print(f"  Status:    {t.status}")
    print(f"  Priority:  {t.priority.name}")
    print(f"  Tags:      {', '.join(t.tags) if t.tags else '—'}")
    print(f"  Depends:   {', '.join(f'#{d}' for d in t.depends_on) if t.depends_on else '—'}")
    print(f"  Blocked:   {blocked}")
    if t.description:
        print(f"  Description: {t.description}")
    print(f"  Created:   {t.created_at}")
    if t.completed_at:
        print(f"  Completed: {t.completed_at}")
    return 0


def _cmd_edit(args: argparse.Namespace, tasks: dict[int, Task]) -> int:
    tid = int(args.task_id)
    t = tasks.get(tid)
    if not t:
        print(f"Error: task #{tid} not found.", file=sys.stderr)
        return 1
    if args.title:
        t.title = args.title
    if args.description is not None:
        t.description = args.description
    if args.priority:
        t.priority = Priority[args.priority.upper()]
    if args.tags:
        t.tags = [tag.strip() for tag in args.tags.split(",")]
    if args.depends_on is not None:
        dependencies = [int(dependency) for dependency in args.depends_on]
        missing = [dependency for dependency in dependencies if dependency not in tasks]
        if missing:
            print(f"Error: dependency task(s) {missing} do not exist.", file=sys.stderr)
            return 1
        previous = t.depends_on
        t.depends_on = dependencies
        if has_cycle(tasks, tid):
            t.depends_on = previous
            print("Error: this edit would create a dependency cycle.", file=sys.stderr)
            return 1
    save_tasks(tasks, args.data)
    print(f"Updated task #{tid}.")
    return 0


def _cmd_done(args: argparse.Namespace, tasks: dict[int, Task]) -> int:
    tid = int(args.task_id)
    t = tasks.get(tid)
    if not t:
        print(f"Error: task #{tid} not found.", file=sys.stderr)
        return 1
    if t.is_blocked(tasks):
        blockers = [d for d in t.depends_on if tasks.get(d) and tasks[d].status != "done"]
        print(
            f"Error: task #{tid} is blocked by unfinished task(s) {blockers}.",
            file=sys.stderr,
        )
        return 1
    t.status = "done"
    t.completed_at = datetime.utcnow().isoformat()
    save_tasks(tasks, args.data)
    print(f"Task #{tid} marked done.")
    return 0


def _cmd_remove(args: argparse.Namespace, tasks: dict[int, Task]) -> int:
    tid = int(args.task_id)
    t = tasks.get(tid)
    if not t:
        print(f"Error: task #{tid} not found.", file=sys.stderr)
        return 1
    dependents = find_dependents(tasks, tid)
    if dependents and not args.force:
        print(
            f"Error: task(s) {dependents} depend on #{tid}. Use --force to remove.",
            file=sys.stderr,
        )
        return 1
    if dependents and args.force:
        clean_dependents(tasks, tid)
    del tasks[tid]
    save_tasks(tasks, args.data)
    print(f"Removed task #{tid}.")
    return 0


def _cmd_next(args: argparse.Namespace, tasks: dict[int, Task]) -> int:
    unblocked = get_next_tasks(tasks)
    if not unblocked:
        print("No unblocked pending tasks.")
        return 0
    for t in unblocked:
        print(f"  #{t.id} [{t.priority.name}] {t.title}")
    return 0


def _cmd_export(args: argparse.Namespace, tasks: dict[int, Task]) -> int:
    fmt = args.format if args.format else "markdown"
    if fmt == "markdown":
        print(export_markdown(tasks))
    else:
        print(f"Error: unsupported format '{fmt}'.", file=sys.stderr)
        return 1
    return 0


# -- argument parser ------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="taskline",
        description="Dependency-aware task CLI",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to taskline.json (default: ./taskline.json)",
    )
    sub = parser.add_subparsers(dest="command")

    # add
    p_add = sub.add_parser("add", help="Add a new task")
    p_add.add_argument("title", help="Task title")
    p_add.add_argument("-d", "--description", default="", help="Description")
    p_add.add_argument("-p", "--priority", choices=["low", "medium", "high"], default="medium")
    p_add.add_argument("-t", "--tags", default="", help="Comma-separated tags")
    p_add.add_argument("--depends-on", nargs="*", default=[], help="Dependency task IDs")

    # list
    sub.add_parser("list", help="List all tasks")

    # show
    p_show = sub.add_parser("show", help="Show task details")
    p_show.add_argument("task_id", help="Task ID")

    # edit
    p_edit = sub.add_parser("edit", help="Edit a task")
    p_edit.add_argument("task_id", help="Task ID")
    p_edit.add_argument("--title", default=None)
    p_edit.add_argument("--description", default=None)
    p_edit.add_argument("--priority", choices=["low", "medium", "high"], default=None)
    p_edit.add_argument("--tags", default=None, help="Comma-separated tags")
    p_edit.add_argument("--depends-on", nargs="*", default=None, help="Dependency task IDs")

    # done
    p_done = sub.add_parser("done", help="Mark task as done")
    p_done.add_argument("task_id", help="Task ID")

    # remove
    p_rm = sub.add_parser("remove", help="Remove a task")
    p_rm.add_argument("task_id", help="Task ID")
    p_rm.add_argument("--force", action="store_true", help="Remove even if depended upon")

    # next
    sub.add_parser("next", help="Show next unblocked tasks")

    # export
    p_exp = sub.add_parser("export", help="Export tasks")
    p_exp.add_argument("--format", choices=["markdown"], default="markdown")

    return parser


# -- main -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    try:
        tasks = load_tasks(args.data)
    except StorageError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    handlers = {
        "add": _cmd_add,
        "list": _cmd_list,
        "show": _cmd_show,
        "edit": _cmd_edit,
        "done": _cmd_done,
        "remove": _cmd_remove,
        "next": _cmd_next,
        "export": _cmd_export,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args, tasks)


if __name__ == "__main__":
    sys.exit(main())
