#!/usr/bin/env python3
"""Report and enforce Alphanus source-code budgets."""

from __future__ import annotations

import argparse
import io
import json
import tokenize
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_GLOBS = ("src/**/*.py", "src-rust/**/*.rs", "bundled-skills/**/*.py")
TEST_GLOBS = ("tests/**/*.py",)
IGNORED_PYTHON_TOKENS = {
    tokenize.COMMENT,
    tokenize.ENCODING,
    tokenize.ENDMARKER,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.NEWLINE,
    tokenize.NL,
}


def _python_lines(path: Path) -> int:
    source = path.read_bytes()
    lines: set[int] = set()
    for token in tokenize.tokenize(io.BytesIO(source).readline):
        if token.type in IGNORED_PYTHON_TOKENS or not token.string.strip():
            continue
        lines.update(range(token.start[0], token.end[0] + 1))
    return len(lines)


def _rust_lines(path: Path) -> int:
    """Count non-comment Rust lines while respecting strings and nested comments."""
    source = path.read_text(encoding="utf-8")
    code_lines: set[int] = set()
    line = 1
    index = 0
    block_depth = 0
    string_quote = ""
    escaped = False
    while index < len(source):
        char = source[index]
        pair = source[index : index + 2]
        if char == "\n":
            line += 1
            escaped = False
            index += 1
            continue
        if block_depth:
            if pair == "/*":
                block_depth += 1
                index += 2
            elif pair == "*/":
                block_depth -= 1
                index += 2
            else:
                index += 1
            continue
        if string_quote:
            if not char.isspace():
                code_lines.add(line)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == string_quote:
                string_quote = ""
            index += 1
            continue
        if pair == "//":
            newline = source.find("\n", index)
            index = len(source) if newline < 0 else newline
            continue
        if pair == "/*":
            block_depth = 1
            index += 2
            continue
        if char == '"':
            string_quote = char
            code_lines.add(line)
        elif not char.isspace():
            code_lines.add(line)
        index += 1
    return len(code_lines)


def _files(globs: tuple[str, ...]) -> list[Path]:
    return sorted({path for pattern in globs for path in ROOT.glob(pattern) if path.is_file()})


def _measure(globs: tuple[str, ...]) -> tuple[int, dict[str, int]]:
    by_file: dict[str, int] = {}
    for path in _files(globs):
        count = _python_lines(path) if path.suffix == ".py" else _rust_lines(path)
        by_file[path.relative_to(ROOT).as_posix()] = count
    return sum(by_file.values()), by_file


def report() -> dict[str, object]:
    production, production_files = _measure(PRODUCTION_GLOBS)
    tests, test_files = _measure(TEST_GLOBS)
    return {
        "production": production,
        "tests": tests,
        "production_files": production_files,
        "test_files": test_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when the current production budget is exceeded")
    parser.add_argument("--details", action="store_true", help="include per-file measurements")
    parser.add_argument("--budget", type=Path, default=ROOT / "tools" / "code_budget.json")
    args = parser.parse_args()

    measured = report()
    output = measured if args.details else {key: measured[key] for key in ("production", "tests")}
    print(json.dumps(output, indent=2, sort_keys=True))
    if not args.check:
        return 0

    budget = json.loads(args.budget.read_text(encoding="utf-8"))
    current = cast(int, measured["production"])
    maximum = int(budget["current_maximum"])
    if current > maximum:
        print(f"production code budget exceeded: {current} > {maximum}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
