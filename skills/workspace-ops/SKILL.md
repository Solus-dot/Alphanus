---
name: workspace-ops
description: Read, write, edit, and inspect workspace files safely.
version: 1.0.0
categories:
  - coding
tags:
  - file
  - write
  - edit
  - read
  - workspace
  - code
tools:
  allowed-tools:
    - create_file
    - edit_file
    - read_file
    - list_files
    - delete_file
    - workspace_tree
x-alphanus:
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
- Use `read_file` before `edit_file` when patching existing files.
- Use `create_file` for new files.
- Keep all write/delete paths inside the workspace root.
- If a path is denied by policy, report the denial and continue safely.
