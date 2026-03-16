from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable, List, Optional

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

METACHAR_BLOCKLIST = ["&&", "||", ";", "`", "$(", "\n", "\r"]


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

    def _resolve_write_path(self, path: str) -> Path:
        raw = Path(os.path.expanduser(path))
        candidate = (self.workspace_root / raw) if not raw.is_absolute() else raw
        resolved = candidate.resolve()
        if not self._is_relative_to(resolved, self.workspace_root):
            raise PermissionError("Write path escapes workspace root")
        return resolved

    def _resolve_read_path(self, path: str) -> Path:
        raw = Path(os.path.expanduser(path))
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

    def create_file(self, filepath: str, content: str) -> str:
        target = self._resolve_write_path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return str(target)

    def edit_file(self, filepath: str, content: str) -> str:
        target = self._resolve_write_path(filepath)
        if not target.exists():
            raise FileNotFoundError(str(target))
        target.write_text(content, encoding="utf-8")
        return str(target)

    def delete_file(self, filepath: str) -> str:
        target = self._resolve_write_path(filepath)
        if not target.exists():
            raise FileNotFoundError(str(target))
        if target.is_dir():
            raise IsADirectoryError(str(target))
        target.unlink()
        return str(target)

    def delete_path(self, path: str, recursive: bool = False) -> str:
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

    def _validate_shell_command(self, command: str) -> None:
        trimmed = command.strip()
        if not trimmed:
            raise PermissionError("Empty command is not allowed")
        for token in METACHAR_BLOCKLIST:
            if token in trimmed:
                raise PermissionError(f"Command rejected by shell metacharacter policy: {token}")
        for pattern in DANGEROUS_SHELL_PATTERNS:
            if re.search(pattern, trimmed, flags=re.IGNORECASE):
                raise PermissionError("Command matches blocked dangerous pattern")

    def run_shell_command(self, command: str, timeout_s: int = 30) -> dict:
        start = time.perf_counter()
        try:
            self._validate_shell_command(command)
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return {
                "ok": True,
                "data": {
                    "command": command,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "returncode": proc.returncode,
                    "cwd": str(self.workspace_root),
                },
                "error": None,
                "meta": {"duration_ms": int((time.perf_counter() - start) * 1000)},
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

    def ensure_workspace_exists(self) -> None:
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def remove_workspace(self) -> None:
        shutil.rmtree(self.workspace_root)
