---
name: app-control
description: List, open, focus, and quit desktop applications with confirmation for high-impact actions.
allowed-tools: list_apps open_app focus_app quit_app
metadata:
  version: "1.0.0"
  tags:
    - desktop
    - apps
    - windows
---
Use app-control when the user wants to inspect or manage local desktop applications.

Rules:
- `list_apps` is read-only and can run directly.
- `open_app`, `focus_app`, and `quit_app` require explicit confirmation through their `confirm_*` argument.
- If confirmation is missing, explain that the action needs approval and ask whether to proceed.
- Do not use app-control for shell commands or file operations.
