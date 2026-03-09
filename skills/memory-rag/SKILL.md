---
name: memory-rag
description: Store and recall durable user facts/preferences.
version: 1.0.0
categories:
  - productivity
tags:
  - remember
  - preference
  - recall
  - memory
tools:
  allowed-tools:
    - store_memory
    - recall_memory
    - list_memories
    - forget_memory
    - get_memory_stats
    - export_memories
x-alphanus:
  triggers:
    keywords:
      - remember
      - preference
      - my
      - "I like"
      - recall
      - memory
---
Use memory tools for user-specific information that should persist.

Rules:
- Store concise factual memories.
- Recall before asking repetitive personal-context questions.
- Keep memory text private and relevant.
- Use at most one `recall_memory` call per turn unless the first recall fails.
- When the user corrects/replaces older information, call `store_memory` once and let it replace stale memories:
  - Automatic replacement works for fact statements like "my <attribute> is <value>" / "user's <attribute> is <value>".
  - For other correction shapes, pass `replace_query` (and optionally `replace_min_score`) to target memories to replace.
