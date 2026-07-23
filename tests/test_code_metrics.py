from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _metrics_module():
    path = Path(__file__).parents[1] / "tools" / "code_metrics.py"
    spec = importlib.util.spec_from_file_location("code_metrics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_code_budget_is_reproducible() -> None:
    root = Path(__file__).parents[1]
    metrics = _metrics_module().report()
    budget = json.loads((root / "tools" / "code_budget.json").read_text(encoding="utf-8"))

    assert metrics["production"] <= budget["current_maximum"]
    assert budget["final_maximum"] <= int(budget["baseline"] * 0.85)
    assert metrics["tests"] > 0
    assert "src-rust/app.rs" in metrics["production_files"]
    assert "tests/test_agent_streaming.py" in metrics["test_files"]
    assert budget["production_module_exceptions"] == {}
    assert budget["test_module_exceptions"] == {}
    assert budget["production_module_maximum"] == 800
    assert budget["test_module_maximum"] == 1200


def test_module_size_exceptions_cannot_hide_growth() -> None:
    metrics = _metrics_module()

    assert metrics._oversized_files({"large.py": 801}, 800, {}) == ["large.py: 801 > 800"]
    assert metrics._oversized_files({"large.py": 801}, 800, {"large.py": 801}) == []
    assert metrics._oversized_files({"large.py": 802}, 800, {"large.py": 801}) == ["large.py: 802 > 800"]
