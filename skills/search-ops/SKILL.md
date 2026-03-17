---
name: search-ops
description: Search the web and fetch page content for research and up-to-date information.
allowed-tools: web_search fetch_url
metadata:
  version: "1.3.0"
  categories:
    - research
  tags:
    - web
    - internet
    - research
    - latest
    - recent
    - current
    - news
    - update
    - lookup
  triggers:
    keywords:
      - web
      - internet
      - online
      - latest
      - recent
      - current
      - today
      - now
      - news
      - updates
      - current situation
      - up to date
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
- For current or recent topics, do not give a confident answer unless at least one fetched page contains usable source text.
- If search works but no fetched source text is available, say you could not verify the answer cleanly.
- Keep the search loop short:
  - use at most 3 `web_search` calls
  - fetch at most 2 URLs
  - after that, answer from the evidence you already have
- Do not keep reformulating the same search endlessly.
- If search results already contain enough evidence, answer without fetching more pages.
- If one fetch fails, try at most one alternative source, then answer.
