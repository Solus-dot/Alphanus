---
name: utilities
description: Utility actions for weather, email metadata, file search, and URL open.
version: 1.0.0
categories:
  - productivity
tags:
  - weather
  - email
  - search
  - open
  - youtube
tools:
  allowed-tools:
    - get_weather
    - read_email
    - search_home_files
    - open_url
    - play_youtube
---
Use utility tools for quick lookup tasks.

Rules:
- Prefer direct answer when possible.
- If network actions fail, return a clear structured error.
- Keep utility responses concise and actionable.
