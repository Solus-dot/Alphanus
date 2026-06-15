---
name: browser-control
description: Open browser pages and inspect the current browser page when platform support exists.
allowed-tools: open_browser_url browser_search get_current_browser_page
metadata:
  version: "1.0.0"
  tags:
    - browser
    - web
    - open
---
Use browser-control for browser navigation and lightweight page inspection.

Rules:
- `open_browser_url` and `browser_search` require explicit confirmation.
- `get_current_browser_page` is read-only and can run directly.
- Do not click, type, submit forms, or automate page actions.
