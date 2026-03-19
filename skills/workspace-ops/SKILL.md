---
name: workspace-ops
description: Read, write, edit, inspect, and delete workspace files and folders safely.
allowed-tools: create_directory create_file create_files edit_file read_file list_files delete_file delete_path workspace_tree
metadata:
  version: "1.1.0"
  categories:
    - coding
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
- Use `create_files` when creating several new files for the same local scaffold or feature.
- Use `read_file` before `edit_file` when patching existing files.
- Prefer `edit_file` with `old_string` and `new_string` for small localized edits.
- Use `edit_file` with full `content` only when replacing most or all of a file.
- Use `create_file` only when the user explicitly wants a new file or a workspace change.
- Use `delete_path` for directory deletion or recursive cleanup inside the workspace.
- Do not use `shell_command` to create folders or inspect local workspace files when a workspace tool can do it.
- Do not use network tools such as `web_search` or `fetch_url` for local workspace file inspection or scaffolding.
- Keep all write/delete paths inside the workspace root.
- If a path is denied by policy, report the denial and continue safely.
