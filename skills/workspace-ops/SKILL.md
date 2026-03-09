---
name: workspace-ops
description: Read, write, edit, and inspect workspace files safely.
allowed-tools: create_file edit_file read_file list_files delete_file workspace_tree
metadata:
  version: "1.0.0"
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
  tools:
    definitions:
      - name: create_file
        capability: workspace_write
        description: Create or overwrite a file in the workspace.
        command: python3 scripts/create_file.py
        timeout-s: 30
        parameters:
          type: object
          properties:
            filepath:
              type: string
            content:
              type: string
          required:
            - filepath
            - content
      - name: edit_file
        capability: workspace_edit
        description: Replace content of an existing workspace file.
        command: python3 scripts/edit_file.py
        timeout-s: 30
        parameters:
          type: object
          properties:
            filepath:
              type: string
            content:
              type: string
          required:
            - filepath
            - content
      - name: read_file
        capability: workspace_read
        description: Read a file under home/workspace policy.
        command: python3 scripts/read_file.py
        timeout-s: 30
        parameters:
          type: object
          properties:
            filepath:
              type: string
          required:
            - filepath
      - name: list_files
        capability: workspace_read
        description: List files in a directory.
        command: python3 scripts/list_files.py
        timeout-s: 30
        parameters:
          type: object
          properties:
            path:
              type: string
          required: []
      - name: delete_file
        capability: workspace_delete
        description: Delete a workspace file.
        command: python3 scripts/delete_file.py
        timeout-s: 30
        parameters:
          type: object
          properties:
            filepath:
              type: string
          required:
            - filepath
      - name: workspace_tree
        capability: workspace_tree
        description: Render the workspace tree.
        command: python3 scripts/workspace_tree.py
        timeout-s: 30
        parameters:
          type: object
          properties:
            max_depth:
              type: integer
          required: []
---
Prefer workspace-scoped operations for coding tasks.

Rules:
- Use `read_file` before `edit_file` when patching existing files.
- Use `create_file` for new files.
- Keep all write/delete paths inside the workspace root.
- If a path is denied by policy, report the denial and continue safely.
