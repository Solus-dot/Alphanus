from __future__ import annotations

from agent.context import ContextWindowManager
from main import DEFAULT_CONFIG, deep_merge


def test_prune_keeps_single_leading_system_message():
    mgr = ContextWindowManager(context_limit=80, keep_last_n=2, safety_margin=0)
    messages = [
        {"role": "system", "content": "base prompt"},
        {"role": "user", "content": "x" * 60},
        {"role": "assistant", "content": "y" * 60},
        {"role": "user", "content": "z" * 60},
        {"role": "assistant", "content": "w" * 60},
    ]

    pruned = mgr.prune(messages, max_tokens=40)
    assert pruned[0]["role"] == "system"
    assert sum(1 for msg in pruned if msg.get("role") == "system") == 1


def test_deep_merge_does_not_mutate_default_config():
    original = DEFAULT_CONFIG["agent"]["model_endpoint"]
    merged = deep_merge(DEFAULT_CONFIG, {"agent": {"model_endpoint": "http://example.local/v1/chat/completions"}})

    assert DEFAULT_CONFIG["agent"]["model_endpoint"] == original
    assert merged["agent"]["model_endpoint"] == "http://example.local/v1/chat/completions"
