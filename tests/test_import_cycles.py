from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
WATCHED_PACKAGES = {"core", "agent"}


def _module_name(path: Path) -> str:
    parts = path.relative_to(SRC_ROOT).parts
    if parts[-1] == "__init__.py":
        return ".".join(parts[:-1])
    return ".".join(Path(*parts).with_suffix("").parts)


def _build_import_graph() -> dict[str, set[str]]:
    modules = {_module_name(path): path for package in WATCHED_PACKAGES for path in (SRC_ROOT / package).glob("*.py")}
    graph = {module: set() for module in modules}

    for module, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level or not node.module:
                    continue
                names = (node.module, *(f"{node.module}.{alias.name}" for alias in node.names))
            else:
                continue

            for name in names:
                if name in modules:
                    graph[module].add(name)

    return graph


def _find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    components: list[list[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in graph[node]:
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[node] = min(lowlink[node], indices[neighbor])

        if lowlink[node] == indices[node]:
            component: list[str] = []
            while True:
                neighbor = stack.pop()
                on_stack.remove(neighbor)
                component.append(neighbor)
                if neighbor == node:
                    break
            components.append(component)

    for node in graph:
        if node not in indices:
            strongconnect(node)

    return [component for component in components if len(component) > 1]


def test_core_and_agent_import_graph_is_acyclic() -> None:
    graph = _build_import_graph()
    cycles = _find_cycles(graph)

    assert cycles == [], f"unexpected import cycles: {cycles}"
