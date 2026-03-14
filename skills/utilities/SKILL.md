---
name: utilities
description: Utility actions for weather, local file search, opening URLs, and playing videos or songs on YouTube.
allowed-tools: get_weather search_home_files open_url play_youtube
metadata:
  version: "1.1.0"
  categories:
    - productivity
  tags:
    - weather
    - search
    - open
    - youtube
    - play
    - music
    - video
---
Use utility tools for quick lookup tasks.

Rules:
- Prefer direct answer when possible.
- If network actions fail, return a clear structured error.
- Keep utility responses concise and actionable.
- After `open_url`, confirm that the exact URL was opened.
- After `play_youtube`, inspect the tool result:
  - If `resolved_first_result` is true, say that the first playable YouTube result was opened.
  - If `resolved_first_result` is false, say that YouTube search results were opened as a fallback.
