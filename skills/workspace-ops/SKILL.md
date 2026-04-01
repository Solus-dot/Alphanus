---
name: workspace-ops
description: Read, write, edit, inspect, move, and delete workspace files and folders safely.
allowed-tools: create_directory create_file edit_file read_file read_files list_files search_code move_path delete_path workspace_tree run_checks
metadata:
  version: "1.2.0"
  tags:
    - file
    - write
    - edit
    - read
    - workspace
    - code
  triggers:
    keywords:
      - file
      - write
      - edit
      - read
      - workspace
      - code
    file_ext:
      - .py
      - .js
      - .ts
      - .md
      - .json
      - .toml
---
Prefer workspace-scoped operations for coding tasks.

Rules:
- If the user asks for code, an example, or a snippet without explicitly asking to save, create, or modify a file, answer inline and do not call workspace tools.
- Use `create_directory` for explicit folder creation inside the workspace.
- For multi-file scaffolds or programs, prefer separate `create_file` calls so progress is visible file by file.
- Use `read_file` before `edit_file` when patching existing files.
- Use `read_files` when you need to inspect several local files together.
- Prefer `edit_file` with `old_string` and `new_string` for small localized edits.
- Use `edit_file` with full `content` only when replacing most or all of a file.
- Use `create_file` only when the user explicitly wants a new file or a workspace change.
- Prefer `search_code` over `shell_command` for repo-wide text lookup.
- Use `move_path` for workspace file or directory renames/moves instead of shell `mv`.
- Use `delete_path` for file deletion, directory deletion, or recursive cleanup inside the workspace.
- Use `run_checks` for tests, lint, and validation commands instead of `shell_command` when possible.
- Do not use `shell_command` to create folders or inspect local workspace files when a workspace tool can do it.
- Do not use network tools such as `web_search` or `fetch_url` for local workspace file inspection or scaffolding.
- Keep all write/delete paths inside the workspace root.
- If a path is denied by policy, report the denial and continue safely.
