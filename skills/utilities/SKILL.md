---
name: utilities
description: Utility actions for weather, email metadata, file search, and URL open.
allowed-tools: get_weather read_email search_home_files open_url play_youtube
metadata:
  version: "1.0.0"
  categories:
    - productivity
  tags:
    - weather
    - email
    - search
    - open
    - youtube
  tools:
    definitions:
      - name: get_weather
        capability: utility_weather
        description: Fetch weather for a city.
        command: python3 scripts/get_weather.py
        timeout-s: 30
        parameters:
          type: object
          properties:
            city:
              type: string
          required:
            - city
      - name: read_email
        capability: utility_email_read
        description: Read latest email metadata via IMAP.
        command: python3 scripts/read_email.py
        timeout-s: 30
        parameters:
          type: object
          properties:
            count:
              type: integer
          required: []
      - name: search_home_files
        capability: utility_file_search
        description: Search filenames under home directory.
        command: python3 scripts/search_home_files.py
        timeout-s: 30
        parameters:
          type: object
          properties:
            query:
              type: string
            directory:
              type: string
          required:
            - query
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
