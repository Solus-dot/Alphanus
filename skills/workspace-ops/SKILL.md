---
name: workspace-ops
description: Read, write, edit, inspect, and delete workspace files and folders safely.
allowed-tools: create_file edit_file read_file list_files delete_file delete_path workspace_tree
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
- Use `read_file` before `edit_file` when patching existing files.
- Use `create_file` only when the user explicitly wants a new file or a workspace change.
- Use `delete_path` for directory deletion or recursive cleanup inside the workspace.
- Keep all write/delete paths inside the workspace root.
- If a path is denied by policy, report the denial and continue safely.
