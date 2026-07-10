---
name: project-ops
description: Read, write, edit, inspect, move, and delete project files and folders safely.
allowed-tools: create_directory create_file edit_file read_file read_files list_files find_files search_code move_path delete_path project_tree
metadata:
  version: "1.2.0"
  tags:
    - file
    - write
    - edit
    - read
    - project
    - code
  triggers:
    keywords:
      - file
      - write
      - edit
      - read
      - project
      - code
    file_ext:
      - .py
      - .js
      - .ts
      - .md
      - .json
      - .toml
---
Prefer project-scoped operations for coding tasks.

Rules:
- If the user asks for code, an example, or a snippet without explicitly asking to save, create, or modify a file, answer inline and do not call project tools.
- Use `create_directory` for explicit folder creation inside the project.
- For multi-file scaffolds or programs, prefer separate `create_file` calls so progress is visible file by file.
- Use `read_file` before `edit_file` when patching existing files.
- Use `read_files` when you need to inspect several local files together.
- Use `find_files` to locate files by name, path fragment, or glob instead of walking directories one level at a time.
- Prefer `edit_file` with `old_string` and `new_string` for small localized edits.
- Use `edit_file` with full `content` only when replacing most or all of a file.
- Use `create_file` only when the user explicitly wants a new file or a project change.
- A successful `create_file` response means the complete supplied content was written. If a UI preview, transcript, or compacted tool display says it was truncated, treat that as display truncation only; do not recreate or overwrite the file just because the preview was clipped.
- If you need to verify a newly written file, read it back with `read_file` or `read_files` using an adequate `max_chars_per_file` instead of issuing another full-file `create_file` call.
- Prefer `search_code` over `shell_command` for repo-wide text lookup.
- Use `move_path` for project file or directory renames/moves instead of shell `mv`.
- Use `delete_path` for file deletion, directory deletion, or recursive cleanup inside the project.
- Use `project_tree` with a `path` for directory summaries; directory symlinks are displayed but are not traversed.
- Use `list_files` only when you already know the exact directory and need a single-level listing.
- There is no project verification runner. If the user asks to run tests or commands, load `shell-ops` and use separate `shell_command` calls.
- Do not use `shell_command` to create folders or inspect local project files when a project tool can do it.
- Do not use network tools such as `web_search` or `fetch_url` for local project file inspection or scaffolding.
- Treat relative paths as project-relative.
- If the user explicitly names an absolute path outside the project, pass that exact path to the project tool instead of substituting the project root.
- Do not use relative traversal like `..` to escape the project root.
- Protected paths such as `.alphanus`, `.git`, secrets, and system locations remain off limits.
- If a path is denied by policy, report the denial and continue safely.
