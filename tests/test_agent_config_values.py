from __future__ import annotations

from agent.config_values import as_json_object, coerce_int, get_json_object


def test_coerce_int_handles_invalid_and_clamps() -> None:
    assert coerce_int("12", 3) == 12
    assert coerce_int("bad", 3) == 3
    assert coerce_int(True, 9) == 9
    assert coerce_int(-5, 1, minimum=0) == 0
    assert coerce_int(25, 1, maximum=20) == 20


def test_as_json_object_returns_empty_for_non_dict() -> None:
    assert as_json_object("nope") == {}
    assert as_json_object(7) == {}


def test_get_json_object_reads_nested_dict_or_default() -> None:
    config = {
        "agent": {"max_action_depth": 10},
        "context": "invalid",
    }
    assert get_json_object(config, "agent") == {"max_action_depth": 10}
    assert get_json_object(config, "context") == {}
    assert get_json_object(config, "missing") == {}
