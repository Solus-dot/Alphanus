from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional

DEFAULT_BLOCKED_PATTERNS = [
    ".ssh",
    ".aws",
    ".gnupg",
    "id_rsa",
    "id_ed25519",
    ".env",
    ".bash_history",
    ".zsh_history",
    ".DS_Store",
]

DANGEROUS_SHELL_PATTERNS = [
    r"\bsudo\b",
    r"\brm\s+-rf\s+/",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bchown\s+-R\s+/",
]

METACHAR_BLOCKLIST = ["&&", "||", "\n", "\r"]
MAX_TOOL_TEXT_BYTES = 20000
SAFE_CHECK_RUNNERS = {
    "pytest",
    "ruff",
    "mypy",
    "pyright",
    "eslint",
    "tsc",
    "vitest",
    "jest",
    "tox",
    "nox",
}


class WorkspaceManager:
    def __init__(
        self,
        workspace_root: str,
        home_root: Optional[str] = None,
        blocked_patterns: Optional[Iterable[str]] = None,
    ) -> None:
        self.workspace_root = Path(os.path.expanduser(workspace_root)).resolve()
        self.home_root = Path(os.path.expanduser(home_root or str(Path.home()))).resolve()
        self.blocked_patterns = list(blocked_patterns or DEFAULT_BLOCKED_PATTERNS)
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def _normalize_workspace_relative(self, path: str) -> Path:
        raw = Path(os.path.expanduser(path))
        if raw.is_absolute():
            parts = raw.parts
            if len(parts) > 2 and parts[1] == self.workspace_root.name:
                return Path(*parts[2:])
            return raw
        parts = raw.parts
        if len(parts) > 1 and parts[0] == self.workspace_root.name:
            return Path(*parts[1:])
        return raw

    def _resolve_write_path(self, path: str) -> Path:
        raw = self._normalize_workspace_relative(path)
        candidate = (self.workspace_root / raw) if not raw.is_absolute() else raw
        resolved = candidate.resolve()
        if not self._is_relative_to(resolved, self.workspace_root):
            raise PermissionError("Write path escapes workspace root")
        return resolved

    def _resolve_read_path(self, path: str) -> Path:
        raw = self._normalize_workspace_relative(path)
        candidate = (self.workspace_root / raw) if not raw.is_absolute() else raw
        resolved = candidate.resolve()

        # Block core system trees.
        for blocked_root in (Path("/etc"), Path("/var"), Path("/System"), Path("/bin"), Path("/usr")):
            if self._is_relative_to(resolved, blocked_root):
                raise PermissionError("Read path is in restricted system location")

        # Workspace reads must remain valid even when the workspace itself is
        # configured outside the user's home directory.
        if self._is_relative_to(resolved, self.workspace_root):
            return resolved

        if not self._is_relative_to(resolved, self.home_root):
            raise PermissionError("Read path must remain inside home directory")
        if self._is_secret_path(resolved):
            raise PermissionError("Read path matches sensitive file policy")
        return resolved

    def _is_secret_path(self, path: Path) -> bool:
        norm = str(path).lower()
        for part in path.parts:
            part_lower = part.lower()
            for pattern in self.blocked_patterns:
                p = pattern.lower()
                if fnmatch.fnmatch(part_lower, p) or p in part_lower:
                    return True
        if norm.endswith(".pem") or norm.endswith(".key"):
            return True
        return False

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def read_file(self, path: str) -> str:
        target = self._resolve_read_path(path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(str(target))
        return target.read_text(encoding="utf-8")

    def read_files(self, paths: Iterable[str], max_chars_per_file: int = MAX_TOOL_TEXT_BYTES) -> List[dict[str, Any]]:
        limit = max(1, int(max_chars_per_file))
        files: List[dict[str, Any]] = []
        for raw_path in paths:
            path = str(raw_path)
            content = self.read_file(path)
            truncated = content[:limit]
            files.append(
                {
                    "filepath": path,
                    "content": truncated,
                    "size_bytes": len(content.encode("utf-8")),
                    "line_count": content.count("\n") + (1 if content else 0),
                    "truncated": truncated != content,
                    "returned_chars": len(truncated),
                }
            )
        return files

    def create_file(self, filepath: str, content: str) -> str:
        target = self._resolve_write_path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return str(target)

    def create_directory(self, path: str) -> str:
        target = self._resolve_write_path(path)
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    def edit_file(self, filepath: str, content: str) -> str:
        target = self._resolve_write_path(filepath)
        if not target.exists():
            raise FileNotFoundError(str(target))
        target.write_text(content, encoding="utf-8")
        return str(target)

    def delete_path(self, path: str, recursive: bool = False) -> str:
        raw = self._normalize_workspace_relative(path)
        candidate = (self.workspace_root / raw) if not raw.is_absolute() else raw
        candidate = Path(os.path.abspath(str(candidate)))
        if not self._is_relative_to(candidate, self.workspace_root):
            raise PermissionError("Write path escapes workspace root")

        if candidate == self.workspace_root:
            raise PermissionError("Deleting the workspace root is not allowed")
        protected_dir = self.workspace_root / ".alphanus"
        if self._is_relative_to(candidate, protected_dir):
            raise PermissionError("Deleting .alphanus state is not allowed")
        if candidate.is_symlink():
            candidate.unlink()
            return str(candidate)

        target = self._resolve_write_path(path)
        if not target.exists():
            raise FileNotFoundError(str(target))
        if target.is_file():
            target.unlink()
            return str(target)
        if not target.is_dir():
            raise FileNotFoundError(str(target))
        if recursive:
            shutil.rmtree(target)
            return str(target)
        try:
            target.rmdir()
        except OSError as exc:
            raise OSError("Directory is not empty; set recursive=true to delete it") from exc
        return str(target)

    def list_files(self, path: str = ".") -> List[str]:
        target = self._resolve_read_path(path)
        if not target.exists() or not target.is_dir():
            raise FileNotFoundError(str(target))
        results = []
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            suffix = "/" if child.is_dir() else ""
            results.append(child.name + suffix)
        return results

    def workspace_tree(self, max_depth: int = 3) -> str:
        lines: List[str] = [f"{self.workspace_root.name}/"]

        def walk(path: Path, prefix: str, depth: int) -> None:
            if depth > max_depth:
                return
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            for idx, entry in enumerate(entries):
                last = idx == len(entries) - 1
                conn = "└── " if last else "├── "
                label = entry.name + ("/" if entry.is_dir() else "")
                lines.append(prefix + conn + label)
                if entry.is_dir():
                    walk(entry, prefix + ("    " if last else "│   "), depth + 1)

        walk(self.workspace_root, "", 1)
        return "\n".join(lines)

    @staticmethod
    def _clip_text(text: str, max_bytes: int = MAX_TOOL_TEXT_BYTES) -> tuple[str, bool]:
        encoded = text.encode("utf-8")
        if len(encoded) <= max_bytes:
            return text, False
        clipped = encoded[:max_bytes].decode("utf-8", errors="ignore")
        return clipped, True

    def _run_argv(
        self,
        argv: List[str],
        *,
        timeout_s: int = 30,
        cwd: Optional[Path] = None,
        max_output_bytes: int = MAX_TOOL_TEXT_BYTES,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        proc = subprocess.run(
            argv,
            shell=False,
            cwd=str(cwd or self.workspace_root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        stdout, stdout_truncated = self._clip_text(proc.stdout, max_output_bytes)
        stderr, stderr_truncated = self._clip_text(proc.stderr, max_output_bytes)
        return {
            "command": argv[0] if argv else "",
            "argv": argv,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "returncode": proc.returncode,
            "cwd": str(cwd or self.workspace_root),
            "duration_ms": int((time.perf_counter() - start) * 1000),
        }

    def _resolve_workspace_subpath(self, path: str) -> Path:
        return self._resolve_read_path(path or ".")

    def _is_searchable_path(self, path: Path) -> bool:
        try:
            resolved = self._resolve_read_path(str(path))
        except (PermissionError, FileNotFoundError):
            return False
        return resolved.exists() and ".git" not in resolved.parts

    def _iter_searchable_files(self, target: Path, glob: Optional[str]) -> Iterable[Path]:
        if target.is_file():
            if self._is_searchable_path(target) and (not glob or fnmatch.fnmatch(target.name, glob)):
                yield target
            return

        for root, dirnames, filenames in os.walk(target, topdown=True):
            root_path = Path(root)
            allowed_dirs = []
            for dirname in dirnames:
                candidate = root_path / dirname
                if self._is_searchable_path(candidate):
                    allowed_dirs.append(dirname)
            dirnames[:] = allowed_dirs

            for filename in filenames:
                candidate = root_path / filename
                if glob and not fnmatch.fnmatch(filename, glob):
                    continue
                if not self._is_searchable_path(candidate):
                    continue
                yield candidate

    def search_code(
        self,
        query: str,
        *,
        path: str = ".",
        glob: Optional[str] = None,
        max_results: int = 50,
        case_sensitive: bool = False,
        fixed_strings: bool = True,
    ) -> dict[str, Any]:
        needle = str(query)
        if not needle.strip():
            raise ValueError("search_code query must be non-empty")

        target = self._resolve_workspace_subpath(path)
        limit = max(1, int(max_results))
        searchable_files = list(self._iter_searchable_files(target, glob))
        rg_path = shutil.which("rg")
        if rg_path and searchable_files:
            results: List[dict[str, Any]] = []
            truncated = False
            chunk_size = 200
            for start in range(0, len(searchable_files), chunk_size):
                chunk = searchable_files[start : start + chunk_size]
                argv = [
                    rg_path,
                    "--json",
                    "--line-number",
                    "--column",
                    "--hidden",
                ]
                if not case_sensitive:
                    argv.append("-i")
                if fixed_strings:
                    argv.append("-F")
                argv.append(needle)
                argv.extend(str(item) for item in chunk)
                proc = subprocess.run(
                    argv,
                    shell=False,
                    cwd=str(self.workspace_root),
                    capture_output=True,
                    text=True,
                )
                if proc.returncode not in (0, 1):
                    raise RuntimeError(proc.stderr.strip() or "rg search failed")

                for line in proc.stdout.splitlines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if event.get("type") != "match":
                        continue
                    data = event.get("data") or {}
                    path_data = data.get("path") or {}
                    lines_data = data.get("lines") or {}
                    submatches = []
                    for match in data.get("submatches") or []:
                        text_data = match.get("match") or {}
                        submatches.append(
                            {
                                "match": str(text_data.get("text", "")),
                                "start": int(match.get("start", 0)),
                                "end": int(match.get("end", 0)),
                            }
                        )
                    results.append(
                        {
                            "filepath": str(path_data.get("text", "")),
                            "line_number": int(data.get("line_number", 0)),
                            "line": str(lines_data.get("text", "")).rstrip("\n"),
                            "submatches": submatches,
                        }
                    )
                    if len(results) >= limit:
                        truncated = True
                        break
                if truncated:
                    break

            return {
                "query": needle,
                "path": str(target),
                "glob": glob,
                "count": len(results),
                "results": results,
                "truncated": truncated,
                "backend": "rg",
            }

        pattern = needle if case_sensitive else needle.lower()
        results = []
        for file_path in searchable_files:
            try:
                content = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                haystack = line if case_sensitive else line.lower()
                if pattern in haystack:
                    start = haystack.find(pattern)
                    results.append(
                        {
                            "filepath": str(file_path),
                            "line_number": line_number,
                            "line": line,
                            "submatches": [{"match": needle, "start": start, "end": start + len(needle)}],
                        }
                    )
                    if len(results) >= limit:
                        return {
                            "query": needle,
                            "path": str(target),
                            "glob": glob,
                            "count": len(results),
                            "results": results,
                            "truncated": True,
                            "backend": "python",
                        }
        return {
            "query": needle,
            "path": str(target),
            "glob": glob,
            "count": len(results),
            "results": results,
            "truncated": False,
            "backend": "python",
        }

    def run_checks(
        self,
        command: str,
        *,
        args: Optional[Iterable[str]] = None,
        path: str = ".",
        timeout_s: int = 120,
    ) -> dict[str, Any]:
        executable = str(command).strip()
        if not executable:
            raise ValueError("run_checks command must be non-empty")
        if re.search(r"[\\/\s]", executable):
            raise ValueError("run_checks command must be a single executable name")

        argv = [executable]
        for item in args or []:
            value = str(item)
            if "\n" in value or "\r" in value:
                raise ValueError("run_checks args must not contain newlines")
            argv.append(value)

        if executable == "uv":
            if len(argv) < 3 or argv[1] != "run" or argv[2] not in SAFE_CHECK_RUNNERS:
                raise PermissionError("run_checks only allows 'uv run' with approved verification runners")
        elif executable not in SAFE_CHECK_RUNNERS:
            raise PermissionError("run_checks only supports approved verification runners")

        cwd = self._resolve_workspace_subpath(path)
        if cwd.is_file():
            cwd = cwd.parent
        run = self._run_argv(argv, timeout_s=max(1, int(timeout_s)), cwd=cwd)
        run["passed"] = run["returncode"] == 0
        return run

    def _validate_shell_command(self, command: str) -> None:
        trimmed = command.strip()
        if not trimmed:
            raise PermissionError("Empty command is not allowed")
        for token in METACHAR_BLOCKLIST:
            if token in trimmed:
                raise PermissionError(f"Command rejected by shell metacharacter policy: {token}")
        try:
            argv = shlex.split(trimmed, posix=True)
        except ValueError as exc:
            raise PermissionError(f"Command could not be parsed safely: {exc}") from exc
        if not argv:
            raise PermissionError("Empty command is not allowed")
        if any(part in {"&&", "||", ";", "|"} for part in argv):
            raise PermissionError("Command rejected by shell metacharacter policy")
        for pattern in DANGEROUS_SHELL_PATTERNS:
            if re.search(pattern, trimmed, flags=re.IGNORECASE):
                raise PermissionError("Command matches blocked dangerous pattern")

    def _parse_command_argv(self, command: str) -> List[str]:
        try:
            argv = shlex.split(command, posix=True)
        except ValueError as exc:
            raise PermissionError(f"Command could not be parsed safely: {exc}") from exc
        if not argv:
            raise PermissionError("Empty command is not allowed")
        return argv

    def run_shell_command(self, command: str, timeout_s: int = 30) -> dict:
        start = time.perf_counter()
        try:
            self._validate_shell_command(command)
            argv = self._parse_command_argv(command)
            run = self._run_argv(argv, timeout_s=timeout_s)
            if run["returncode"] != 0:
                detail = (run["stderr"] or run["stdout"] or "").strip()
                message = f"Command exited with code {run['returncode']}"
                if detail:
                    message += f": {detail}"
                return {
                    "ok": False,
                    "data": {
                        "command": command,
                        "argv": run["argv"],
                        "stdout": run["stdout"],
                        "stderr": run["stderr"],
                        "returncode": run["returncode"],
                        "cwd": run["cwd"],
                        "stdout_truncated": run["stdout_truncated"],
                        "stderr_truncated": run["stderr_truncated"],
                    },
                    "error": {"code": "E_SHELL", "message": message},
                    "meta": {"duration_ms": run["duration_ms"]},
                }
            return {
                "ok": True,
                "data": {
                    "command": command,
                    "argv": run["argv"],
                    "stdout": run["stdout"],
                    "stderr": run["stderr"],
                    "returncode": run["returncode"],
                    "cwd": run["cwd"],
                    "stdout_truncated": run["stdout_truncated"],
                    "stderr_truncated": run["stderr_truncated"],
                },
                "error": None,
                "meta": {"duration_ms": run["duration_ms"]},
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "data": None,
                "error": {"code": "E_TIMEOUT", "message": f"Command timed out after {timeout_s}s"},
                "meta": {"duration_ms": int((time.perf_counter() - start) * 1000)},
            }
        except PermissionError as exc:
            return {
                "ok": False,
                "data": None,
                "error": {"code": "E_POLICY", "message": str(exc)},
                "meta": {"duration_ms": int((time.perf_counter() - start) * 1000)},
            }
        except Exception as exc:
            return {
                "ok": False,
                "data": None,
                "error": {"code": "E_IO", "message": str(exc)},
                "meta": {"duration_ms": int((time.perf_counter() - start) * 1000)},
            }
