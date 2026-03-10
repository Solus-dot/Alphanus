from __future__ import annotations

from typing import Tuple

from rich.markup import escape as esc


def hanging_indent(line: str) -> int:
    stripped = line.lstrip(" ")
    lead = len(line) - len(stripped)
    if stripped.startswith(("- ", "* ", "+ ")):
        return lead + 2
    if ". " in stripped[:5] and stripped[0].isdigit():
        idx = stripped.find(". ")
        if idx > 0:
            return lead + idx + 2
    return lead


def render_md(line: str, in_fence: bool) -> Tuple[str, bool]:
    stripped = line.strip()
    if stripped.startswith("```") or stripped.startswith("~~~"):
        return f"[dim]{esc(line)}[/dim]", not in_fence
    if in_fence:
        return esc(line), in_fence

    out = []
    i = 0
    n = len(line)
    while i < n:
        if line[i] == "`":
            j = i + 1
            while j < n and line[j] != "`":
                j += 1
            if j < n:
                out.append(f"[bold yellow]{esc(line[i:j+1])}[/bold yellow]")
                i = j + 1
            else:
                out.append(esc(line[i:]))
                break
            continue
        if i + 1 < n and line[i : i + 2] == "**":
            j = line.find("**", i + 2)
            if j != -1:
                out.append(f"[bold]{esc(line[i+2:j])}[/bold]")
                i = j + 2
                continue
        out.append(esc(line[i]))
        i += 1
    return "".join(out), in_fence
