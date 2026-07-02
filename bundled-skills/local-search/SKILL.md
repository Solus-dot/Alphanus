---
name: local-search
description: Search local files by filename and text content with safe snippets.
allowed-tools: search_local_files
metadata:
  version: "1.0.0"
  tags:
    - search
    - files
    - local
---
Use local-search when the user wants to find files or text on disk.

Rules:
- `search_local_files` is read-only and can run directly.
- Search only under the project root.
- Return concise matches and snippets; do not dump whole files.
- Use project tools for reading exact project files after search identifies them.
