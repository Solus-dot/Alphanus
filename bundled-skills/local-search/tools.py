from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from skills.runtime import ToolExecutionEnv

TOOL_SPEC_ROWS = {  # fmt: skip
    "search_local_files": ("local_search", False, ("read", "list", "check"), "Search local filenames and text content under the project root.", {"query": {"type": "string"}, "root": {"type": "string"}, "mode": {"type": "string", "enum": ["filename", "content", "both"]}, "max_results": {"type": "integer"}}, ("query",), False),
}

_IGNORE_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".next", "dist", "build", ".cache"}
_DEFAULT_MAX_RESULTS = 50
_MAX_RESULTS_LIMIT = 500
_DEFAULT_MAX_TEXT_BYTES = 512_000
_BINARY_PROBE_BYTES = 4096
_SNIPPET_BEFORE_CHARS = 80
_SNIPPET_AFTER_CHARS = 120


def _ok(data: dict[str, object]) -> dict[str, object]:
    return {"ok": True, "data": data, "error": None, "meta": {}}


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _allowed_root(raw: str, env: ToolExecutionEnv) -> Path:
    project = Path(env.project.project_root).expanduser().resolve()
    root = Path(os.path.expanduser(raw or str(project))).resolve()
    if not _is_under(root, project):
        raise PermissionError("Search root is outside project root")
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Search root not found: {root}")
    return root


def _is_allowed_candidate(path: Path, env: ToolExecutionEnv, root: Path) -> bool:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return False
    if not _is_under(resolved, root):
        return False
    try:
        env.project._resolve_read_path(str(resolved))  # noqa: SLF001
    except (OSError, PermissionError):
        return False
    return not (
        env.project._is_secret_path(resolved)  # noqa: SLF001
        or env.project._is_protected_state_path(resolved)  # noqa: SLF001
    )


def _is_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:_BINARY_PROBE_BYTES]
    except OSError:
        return True
    return b"\x00" in chunk


def _max_text_bytes(env: ToolExecutionEnv) -> int:
    tools_cfg = env.config.get("tools")
    local_search_cfg = tools_cfg.get("local_search") if isinstance(tools_cfg, dict) else {}
    raw = local_search_cfg.get("max_text_bytes") if isinstance(local_search_cfg, dict) else None
    try:
        return max(1, int(raw if raw is not None else _DEFAULT_MAX_TEXT_BYTES))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_TEXT_BYTES


def _snippet(text: str, needle: str) -> str:
    lowered = text.lower()
    idx = lowered.find(needle.lower())
    if idx < 0:
        return ""
    start = max(0, idx - _SNIPPET_BEFORE_CHARS)
    end = min(len(text), idx + len(needle) + _SNIPPET_AFTER_CHARS)
    return " ".join(text[start:end].split())


def _search(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, object]:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    mode = str(args.get("mode") or "both").strip().lower()
    if mode not in {"filename", "content", "both"}:
        raise ValueError("mode must be filename, content, or both")
    max_results = max(1, min(_MAX_RESULTS_LIMIT, int(args.get("max_results") or _DEFAULT_MAX_RESULTS)))
    max_text_bytes = _max_text_bytes(env)
    root = _allowed_root(str(args.get("root") or ""), env)
    matches: list[dict[str, object]] = []
    scanned = 0
    skipped_binary = 0
    skipped_large = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _IGNORE_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            scanned += 1
            rel = str(path)
            if not _is_allowed_candidate(path, env, root):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            hit: dict[str, object] | None = None
            if mode in {"filename", "both"} and query.lower() in name.lower():
                hit = {"path": rel, "match_type": "filename", "snippet": "", "size_bytes": stat.st_size, "modified_at": stat.st_mtime}
            if hit is None and mode in {"content", "both"}:
                if stat.st_size > max_text_bytes:
                    skipped_large += 1
                    continue
                if _is_binary(path):
                    skipped_binary += 1
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                snip = _snippet(text, query)
                if snip:
                    hit = {"path": rel, "match_type": "content", "snippet": snip, "size_bytes": stat.st_size, "modified_at": stat.st_mtime}
            if hit is not None:
                matches.append(hit)
                if len(matches) >= max_results:
                    return _ok(
                        {
                            "query": query,
                            "root": str(root),
                            "mode": mode,
                            "matches": matches,
                            "truncated": True,
                            "scanned_files": scanned,
                            "skipped_binary": skipped_binary,
                            "skipped_large": skipped_large,
                        }
                    )
    return _ok(
        {
            "query": query,
            "root": str(root),
            "mode": mode,
            "matches": matches,
            "truncated": False,
            "scanned_files": scanned,
            "skipped_binary": skipped_binary,
            "skipped_large": skipped_large,
        }
    )


def execute(tool_name: str, args: dict[str, object], env: ToolExecutionEnv) -> dict[str, Any]:
    if tool_name == "search_local_files":
        return _search(args, env)
    raise ValueError(f"Unsupported tool: {tool_name}")
