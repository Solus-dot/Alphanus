Prefer workspace-scoped operations for coding tasks.

Rules:
- Use `read_file` before `edit_file` when patching existing files.
- Use `create_file` for new files.
- Keep all write/delete paths inside the workspace root.
- If a path is denied by policy, report the denial and continue safely.
