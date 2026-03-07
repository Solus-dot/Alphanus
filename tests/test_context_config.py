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


def test_prune_keeps_required_assistant_for_tool_message():
    mgr = ContextWindowManager(context_limit=140, keep_last_n=2, safety_margin=0)
    messages = [
        {"role": "system", "content": "base prompt"},
        {"role": "user", "content": "write file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "create_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "create_file", "content": '{"ok": true}'},
    ]

    pruned = mgr.prune(messages, max_tokens=120)
    roles = [msg.get("role") for msg in pruned]
    assert roles[0] == "system"
    if "tool" in roles:
        assert "assistant" in roles


def test_prune_hard_fallback_enforces_budget():
    mgr = ContextWindowManager(context_limit=90, keep_last_n=2, safety_margin=0)
    messages = [
        {"role": "system", "content": "base prompt"},
        {"role": "user", "content": "run the tool"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_big",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"filepath":"huge.txt"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_big",
            "name": "read_file",
            "content": "x" * 5000,
        },
    ]

    pruned = mgr.prune(messages, max_tokens=40)
    assert mgr.estimate_tokens(pruned) + 40 <= mgr.context_limit - mgr.safety_margin
