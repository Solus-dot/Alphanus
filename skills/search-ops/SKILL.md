---
name: search-ops
description: Search the web and fetch page content for research and up-to-date information.
allowed-tools: web_search fetch_url
metadata:
  version: "1.0.0"
  categories:
    - research
  tags:
    - web
    - internet
    - research
    - latest
    - lookup
  triggers:
    keywords:
      - web
      - internet
      - online
      - latest
      - news
      - research
      - look up
      - search the web
---
Use search tools when the user needs information from the public internet.

Rules:
- Prefer direct answers from existing context when the information is already available.
- Use `web_search` first to gather candidate sources, then `fetch_url` only for the results you actually need.
- Use internet search for time-sensitive or factual lookup tasks.
- Keep citations compact: mention the source title or domain when summarizing fetched pages.
- If a page fetch fails, continue with the remaining results instead of stopping the turn.
