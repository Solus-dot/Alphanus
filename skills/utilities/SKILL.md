---
name: utilities
description: Utility actions for weather, file search, URL open, and YouTube playback.
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
  tools:
    definitions:
      - name: open_url
        capability: utility_open_url
        description: Open URL in default browser.
        command: python3 scripts/open_url.py
        timeout-s: 30
        parameters:
          type: object
          properties:
            url:
              type: string
          required:
            - url
      - name: play_youtube
        capability: utility_play_youtube
        description: Open the first YouTube video result for a topic and autoplay when resolvable.
        command: python3 scripts/play_youtube.py
        timeout-s: 30
        parameters:
          type: object
          properties:
            topic:
              type: string
          required:
            - topic
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
