from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportTypedDictNotRequiredAccess=false, reportOptionalMemberAccess=false
from agent.context import ContextWindowManager
from core.configuration import DEFAULT_CONFIG, deep_merge


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


def test_estimate_tokens_floors_after_aggregating_whole_prompt():
    mgr = ContextWindowManager()
    messages = [
        {"role": "user", "content": "ab"},
        {"role": "assistant", "content": "cd"},
        {"role": "user", "content": "ef"},
    ]

    manual_chars = 0
    for message in messages:
        manual_chars += len(message["role"]) + 4 + len(message["content"])

    assert mgr.estimate_tokens(messages) == max(1, manual_chars // 4)


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


def test_default_config_includes_tui_memory_limits():
    tui = DEFAULT_CONFIG.get("tui", {})
    assert int(tui.get("chat_log_max_lines", 0)) > 0
    tree = tui.get("tree_compaction", {})
    assert bool(tree.get("enabled", False))


def test_prune_keeps_at_least_one_user_message_when_tool_bundle_is_large():
    mgr = ContextWindowManager(context_limit=900, keep_last_n=10, safety_margin=0)
    messages = [{"role": "system", "content": "base prompt"}]
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + ("A" * 2000)}},
                {"type": "text", "text": "animate gradient with mouse tracking"},
            ],
        }
    )

    for idx in range(6):
        call_id = f"call_{idx}"
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "edit_file", "arguments": '{"content":"' + ("x" * 2000) + '"}'},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "edit_file",
                "content": "y" * 4000,
            }
        )

    pruned = mgr.prune(messages, max_tokens=200)
    assert any(msg.get("role") == "user" for msg in pruned)


def test_prune_preserves_latest_multimodal_user_message_under_hard_budget():
    mgr = ContextWindowManager(context_limit=8192, keep_last_n=10, safety_margin=500)
    messages = [{"role": "system", "content": "x" * 7771}]

    for idx in range(12):
        messages.append({"role": "assistant", "content": "y" * 400})

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[Attachments: image.png (image)]\n\nWhat's this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + ("A" * 20000)}},
            ],
        }
    )

    pruned = mgr.prune(messages, max_tokens=2048)

    assert len(pruned) >= 2
    assert pruned[-1]["role"] == "user"
    assert isinstance(pruned[-1]["content"], list)
    assert any(part.get("type") == "image_url" for part in pruned[-1]["content"])


def test_prune_preserves_multimodal_structure_in_final_hard_fallback():
    mgr = ContextWindowManager(context_limit=120, keep_last_n=1, safety_margin=0)
    messages = [
        {"role": "system", "content": "x" * 1000},
        {"role": "assistant", "content": "y" * 500},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[Attachments: image.png (image)]\n\nWhat's this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + ("A" * 20000)}},
            ],
        },
    ]

    pruned = mgr.prune(messages, max_tokens=40)

    assert len(pruned) == 2
    assert pruned[-1]["role"] == "user"
    assert isinstance(pruned[-1]["content"], list)
    assert any(part.get("type") == "image_url" for part in pruned[-1]["content"])
