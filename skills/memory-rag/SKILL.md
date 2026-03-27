---
name: memory-rag
description: Store and recall durable user facts/preferences.
allowed-tools: store_memory recall_memory list_memories forget_memory get_memory_stats export_memories
metadata:
  version: "1.1.0"
  tags:
    - remember
    - preference
    - recall
    - memory
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
- Memory retrieval is semantic only. Do not rely on regex or lexical fallback behavior.
- When the user corrects/replaces older information, call `store_memory` with `replace_query` or `replace_ids` to target stale memories explicitly.
