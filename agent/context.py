from __future__ import annotations

from typing import Dict, List


class ContextWindowManager:
    def __init__(self, context_limit: int = 8192, keep_last_n: int = 10, safety_margin: int = 500):
        self.context_limit = int(context_limit)
        self.keep_last_n = int(keep_last_n)
        self.safety_margin = int(safety_margin)

    @staticmethod
    def estimate_tokens(messages: List[Dict]) -> int:
        chars = 0
        for msg in messages:
            chars += len(msg.get("role", "")) + 4
            content = msg.get("content", "")
            if isinstance(content, str):
                chars += len(content)
            else:
                chars += len(str(content))
            if msg.get("tool_calls"):
                chars += len(str(msg["tool_calls"]))
        return max(1, chars // 4)

    def prune(self, messages: List[Dict], max_tokens: int) -> List[Dict]:
        if len(messages) <= 2:
            return messages

        estimated = self.estimate_tokens(messages)
        if estimated + max_tokens <= self.context_limit - self.safety_margin:
            return messages

        keep_tail = self.keep_last_n
        if len(messages) <= keep_tail + 1:
            return messages

        head = messages[:1]
        tail = messages[-keep_tail:]
        pruned = [
            {
                "role": "system",
                "content": "[...Earlier conversation history pruned for length...]",
            }
        ]
        return head + pruned + tail
