from __future__ import annotations

from pathlib import Path

from core.conv_tree import ConvTree


def test_add_complete_and_cancel_turn():
    tree = ConvTree()

    turn_ok = tree.add_turn("hello")
    tree.complete_turn(turn_ok.id, "world")
    assert tree.nodes[turn_ok.id].assistant_content == "world"

    turn_bad = tree.add_turn("stop")
    tree.cancel_turn(turn_bad.id, "partial")
    content = tree.nodes[turn_bad.id].assistant_content
    assert content is not None
    assert "[interrupted]" in content
    assert "partial" in content


def test_branch_unbranch_and_switch():
    tree = ConvTree()
    a = tree.add_turn("a")
    tree.complete_turn(a.id, "A")

    tree.arm_branch("alt")
    b = tree.add_turn("b")
    tree.complete_turn(b.id, "B")

    # Go back to fork parent.
    parent = tree.unbranch()
    assert parent == a.id
    assert tree.current_id == a.id

    # Add another child on same parent and switch.
    c = tree.add_turn("c")
    tree.complete_turn(c.id, "C")

    tree.current_id = a.id
    chosen = tree.switch_child(0)
    assert chosen is not None
    assert tree.current_id in {b.id, c.id}


def test_history_messages_include_skill_exchanges():
    tree = ConvTree()
    turn = tree.add_turn("user")
    tree.append_skill_exchange(turn.id, {"role": "assistant", "tool_calls": [{"id": "x"}]})
    tree.append_skill_exchange(turn.id, {"role": "tool", "tool_call_id": "x", "name": "read_file", "content": "{}"})
    tree.complete_turn(turn.id, "done")

    msgs = tree.history_messages()
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[2]["role"] == "tool"
    assert msgs[3]["role"] == "assistant"


def test_save_load_roundtrip(tmp_path: Path):
    tree = ConvTree()
    turn = tree.add_turn("hello")
    tree.cancel_turn(turn.id, "partial")

    path = tmp_path / "tree.json"
    tree.save(str(path))
    loaded = ConvTree.load(str(path))

    assert loaded.current_id == tree.current_id
    assert loaded.nodes.keys() == tree.nodes.keys()
    assert "[interrupted]" in (loaded.nodes[turn.id].assistant_content or "")


def test_user_text_strips_inline_attachment_blocks():
    tree = ConvTree()
    content = [
        {
            "type": "text",
            "text": "[File: foo.py]\n```py\nprint('x')\n```\n\nPlease refactor this.",
        }
    ]
    turn = tree.add_turn(content)
    assert turn.user_text() == "Please refactor this."


def test_inactive_branch_compaction_truncates_large_payloads_only_when_inactive():
    tree = ConvTree(
        compact_inactive_branches=True,
        inactive_assistant_char_limit=32,
        inactive_tool_argument_char_limit=32,
        inactive_tool_content_char_limit=32,
    )

    root_turn = tree.add_turn("root")
    tree.complete_turn(root_turn.id, "root-ok")

    tree.arm_branch("branch-a")
    a = tree.add_turn("work on a")
    tree.append_skill_exchange(
        a.id,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "create_file", "arguments": '{"content":"' + ("x" * 120) + '"}'},
                }
            ],
        },
    )
    tree.append_skill_exchange(
        a.id,
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "create_file",
            "content": "y" * 120,
        },
    )
    tree.complete_turn(a.id, "z" * 120)

    # Still active branch: no compaction should apply yet.
    active_args = tree.nodes[a.id].skill_exchanges[0]["tool_calls"][0]["function"]["arguments"]
    assert "...[compacted]" not in active_args

    # Move back and open sibling branch, making branch-a inactive.
    tree.unbranch()
    b = tree.add_turn("work on b")
    tree.complete_turn(b.id, "done")

    inactive_node = tree.nodes[a.id]
    assert "...[compacted]" in (inactive_node.assistant_content or "")
    compacted_args = inactive_node.skill_exchanges[0]["tool_calls"][0]["function"]["arguments"]
    compacted_tool_content = inactive_node.skill_exchanges[1]["content"]
    assert "...[compacted]" in compacted_args
    assert "...[compacted]" in compacted_tool_content
