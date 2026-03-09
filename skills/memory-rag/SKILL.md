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
  definitions:
    - name: store_memory
      capability: memory_store
      description: Persist a memory item.
      command: python3 scripts/ops.py store_memory
      timeout-s: 30
      parameters:
        type: object
        properties:
          text:
            type: string
          memory_type:
            type: string
          importance:
            type: number
          replace_existing:
            type: boolean
          replace_query:
            type: string
          replace_top_k:
            type: integer
          replace_min_score:
            type: number
          replace_ids:
            type: array
            items:
              type: integer
          metadata:
            type: object
        required:
          - text
    - name: recall_memory
      capability: memory_recall
      description: Semantic search over memories.
      command: python3 scripts/ops.py recall_memory
      timeout-s: 30
      parameters:
        type: object
        properties:
          query:
            type: string
          top_k:
            type: integer
          memory_type:
            type: string
        required:
          - query
    - name: list_memories
      capability: memory_list
      description: List recent memories.
      command: python3 scripts/ops.py list_memories
      timeout-s: 30
      parameters:
        type: object
        properties:
          count:
            type: integer
        required: []
    - name: forget_memory
      capability: memory_forget
      description: Delete memory by id.
      command: python3 scripts/ops.py forget_memory
      timeout-s: 30
      parameters:
        type: object
        properties:
          memory_id:
            type: integer
        required:
          - memory_id
    - name: get_memory_stats
      capability: memory_stats
      description: Get memory statistics.
      command: python3 scripts/ops.py get_memory_stats
      timeout-s: 30
      parameters:
        type: object
        properties: {}
        required: []
    - name: export_memories
      capability: memory_export
      description: Export memories to a text file.
      command: python3 scripts/ops.py export_memories
      timeout-s: 30
      parameters:
        type: object
        properties:
          filepath:
            type: string
        required: []
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
