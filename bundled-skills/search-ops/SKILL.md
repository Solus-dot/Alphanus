---
name: search-ops
description: Search SearXNG or Tavily fallback, fetch sources, and retrieve indexed local knowledge for current research.
allowed-tools: web_search fetch_url retrieve_knowledge index_project retrieval_stats forget_retrieval_record
metadata:
  version: "1.3.0"
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
Use search and retrieval tools when the user needs current public internet information or indexed local knowledge.

Rules:
- Prefer direct answers from existing context when the information is already available.
- Use `web_search` first to gather SearXNG candidate sources, or Tavily candidates when the fallback is active, then `fetch_url` only for the results you actually need.
- Use `retrieve_knowledge` before searching when previously fetched sources, memories, explicitly indexed project files, or successful tool outcomes may answer the task.
- Use `index_project` only for files the user asked you to index or files directly needed for the current task.
- Do not use internet search tools to inspect local project files; use `project-ops` for local files.
- Use internet search for time-sensitive or factual lookup tasks.
- Prefer official, primary, or clearly attributable sources when they are available.
- Keep citations compact: mention the source title or domain when summarizing fetched pages, and include dates if the fetched metadata exposes them.
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
- `fetch_url` stores fetched pages in the local retrieval index when indexing is enabled; search snippets alone are not durable source evidence.
- After retrieving data, prefer this response shape when the user asked for current information:
  - first give the direct answer in 1-3 sentences
  - then briefly note the strongest sources checked
  - if the evidence is partial, stale, or mixed, say that plainly instead of sounding fully certain
- Use fetched excerpts and source metadata when available; do not just paraphrase snippets generically.
- Cite fetched web records when they materially influence the final answer.
