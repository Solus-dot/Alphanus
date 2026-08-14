# Taskline – Dependency-Aware Task CLI

A minimal, dependency-aware task manager built entirely with the Python standard library. Tasks support priorities, tags, dependency tracking with cycle detection, and Markdown export.

## Quick Start

```bash
cd long-horizon-lab/taskline
python -m taskline add "Write tests" --priority high --tags dev,test
python -m taskline add "Deploy app" --depends-on 1
python -m taskline list
python -m taskline next
python -m taskline done 1
python -m taskline export --format markdown
```

## Commands

| Command | Description |
|---------|-------------|
| `add` | Add a new task (required: `TITLE`) |
| `list` | List all tasks |
| `show` | Show details for a specific task (`ID`) |
| `edit` | Edit title, description, priority, or tags of a task (`ID`) |
| `done` | Mark a task as completed (`ID`) |
| `remove` | Delete a task (`ID`); use `--force` if other tasks depend on it |
| `next` | Show unfinished, unblocked tasks ordered by priority then creation time |
| `export` | Export tasks to a report (`--format markdown`) |

## Options

- `--priority {low,medium,high}` – set task priority (default: medium)
- `--description TEXT` – task description
- `--tags a,b,c` – comma-separated tags
- `--depends-on 1,2` – comma-separated dependency task IDs
- `--data PATH` – custom path for `taskline.json` (default: current directory)
- `--force` – force removal even when other tasks depend on the target

## Dependency Rules

- Dependencies must reference existing task IDs.
- Adding a dependency that creates a cycle is rejected.
- Removing a task that others depend on requires `--force`; forced removal cleans the dependency references from dependent tasks.

## Running Tests

```bash
cd long-horizon-lab
python -m pytest tests/
# or
python -m unittest discover -s tests
```
