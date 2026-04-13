from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
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
SHELL_WRAPPER_BINARIES = {"bash", "sh", "zsh", "fish"}
MAX_TOOL_TEXT_BYTES = 20000
PROTECTED_STATE_TOKEN_RE = re.compile(r"(^|[^A-Za-z0-9._-])\.alphanus(?=$|[/\\\\]|[^A-Za-z0-9._-])", re.IGNORECASE)
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
PYTHON_MODULE_CHECK_RUNNERS = {"pytest", "ruff", "mypy", "pyright", "tox", "nox"}
READ_ONLY_SHELL_COMMANDS = {
    "ls",
    "pwd",
    "cat",
    "head",
    "tail",
    "grep",
    "rg",
    "find",
    "stat",
    "wc",
    "sort",
    "uniq",
    "cut",
}
READ_ONLY_GIT_SUBCOMMANDS = {"status", "diff", "show", "log", "rev-parse", "branch"}
MUTATING_SHELL_COMMANDS = {"touch", "mkdir", "mv", "cp", "rm", "chmod", "chown", "ln"}
MUTATING_GIT_SUBCOMMANDS = {"add", "rm", "mv", "restore", "checkout", "switch", "commit", "clean", "apply", "am", "stash"}


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
        self._filesystem_case_insensitive = self._detect_case_insensitive_filesystem()

    @property
    def protected_state_dir(self) -> Path:
        return self.workspace_root / ".alphanus"

    def _detect_case_insensitive_filesystem(self) -> bool:
        root = self.workspace_root
        alt_root_name = root.name.swapcase()
        if alt_root_name and alt_root_name != root.name:
            alt_root = root.parent / alt_root_name
            try:
                if alt_root.exists():
                    return alt_root.samefile(root)
            except OSError:
                pass

        fd, probe_text = tempfile.mkstemp(prefix=".alphanusCaseProbe", dir=str(root))
        os.close(fd)
        probe = Path(probe_text)
        try:
            alt_probe = probe.with_name(probe.name.swapcase())
            if alt_probe.name == probe.name:
                return False
            try:
                return alt_probe.exists() and alt_probe.samefile(probe)
            except OSError:
                return False
        finally:
            try:
                probe.unlink()
            except OSError:
                pass

    def _path_compare_parts(self, path: Path) -> tuple[str, ...]:
        parts = tuple(path.parts)
        if not self._filesystem_case_insensitive:
            return parts
        return tuple(part.casefold() for part in parts)

    def _is_path_equal_or_descendant(self, path: Path, root: Path) -> bool:
        path_parts = self._path_compare_parts(path)
        root_parts = self._path_compare_parts(root)
        return len(path_parts) >= len(root_parts) and path_parts[: len(root_parts)] == root_parts

    def _is_protected_state_path(self, path: Path) -> bool:
        try:
            candidate = path.resolve(strict=False)
        except OSError:
            candidate = Path(os.path.abspath(str(path)))
        try:
            protected = self.protected_state_dir.resolve(strict=False)
        except OSError:
            protected = Path(os.path.abspath(str(self.protected_state_dir)))
        return self._is_path_equal_or_descendant(candidate, protected)

    def _resolve_command_token_path(self, token: str, cwd: Path) -> Optional[Path]:
        value = str(token or "").strip()
        if not value or value == "-":
            return None
        pathish = (
            value.startswith((".", "/", "~"))
            or "/" in value
            or "\\" in value
        )
        if not pathish:
            candidate = cwd / value
            try:
                if not candidate.exists() and not candidate.is_symlink():
                    return None
            except OSError:
                return None
            pathish = True
        raw = Path(os.path.expanduser(value))
        candidate = raw if raw.is_absolute() else cwd / raw
        try:
            return candidate.resolve(strict=False)
        except OSError:
            return Path(os.path.abspath(str(candidate)))

    @staticmethod
    def _text_mentions_protected_state(text: str) -> bool:
        return bool(PROTECTED_STATE_TOKEN_RE.search(str(text)))

    def _command_touches_protected_state(self, argv: List[str], cwd: Path) -> bool:
        git_subcommand_index = self._git_subcommand_index(argv) if argv and argv[0] == "git" else -1
        for idx, part in enumerate(argv[1:], start=1):
            if self._text_mentions_protected_state(part):
                return True
            if idx == git_subcommand_index:
                continue
            resolved = self._resolve_command_token_path(part, cwd)
            if resolved is not None and self._is_protected_state_path(resolved):
                return True
        return False

    def workspace_state_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for root, dirnames, filenames in os.walk(self.workspace_root, topdown=True, followlinks=False):
            root_path = Path(root)
            dirnames[:] = sorted(dirname for dirname in dirnames if dirname != ".alphanus")
            rel_root = root_path.relative_to(self.workspace_root)
            for dirname in dirnames:
                candidate = root_path / dirname
                try:
                    stat = candidate.lstat()
                except OSError:
                    continue
                rel_path = (rel_root / dirname).as_posix()
                digest.update(f"d:{rel_path}:{stat.st_mode}:{stat.st_mtime_ns}\n".encode("utf-8"))
            for filename in sorted(filenames):
                candidate = root_path / filename
                try:
                    stat = candidate.lstat()
                except OSError:
                    continue
                rel_path = (rel_root / filename).as_posix()
                digest.update(f"f:{rel_path}:{stat.st_mode}:{stat.st_size}:{stat.st_mtime_ns}\n".encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _git_subcommand(argv: List[str]) -> str:
        if len(argv) < 2:
            return ""
        if argv[1] == "-C" and len(argv) >= 4:
            return argv[3]
        return argv[1]

    @staticmethod
    def _git_subcommand_index(argv: List[str]) -> int:
        if len(argv) < 2:
            return -1
        if argv[1] == "-C" and len(argv) >= 4:
            return 3
        return 1

    def _classify_shell_command(self, argv: List[str]) -> str:
        if not argv:
            return "ambiguous"
        executable = argv[0]
        if executable in READ_ONLY_SHELL_COMMANDS:
            return "readonly"
        if executable in MUTATING_SHELL_COMMANDS:
            return "mutating"
        if executable == "sed":
            return "readonly" if "-i" not in argv else "mutating"
        if executable != "git":
            return "ambiguous"

        subcommand = self._git_subcommand(argv)
        if subcommand == "branch":
            if "--show-current" in argv:
                return "readonly"
            return "ambiguous"
        if subcommand in READ_ONLY_GIT_SUBCOMMANDS:
            return "readonly"
        if subcommand in MUTATING_GIT_SUBCOMMANDS:
            return "mutating"
        return "ambiguous"

    def _git_status_snapshot(self) -> Optional[tuple[str, tuple[tuple[str, str, int, int], ...]]]:
        git_path = shutil.which("git")
        if not git_path:
            return None
        base_argv = [git_path, "-C", str(self.workspace_root)]
        repo_probe = subprocess.run(
            base_argv + ["rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if repo_probe.returncode != 0 or repo_probe.stdout.strip().lower() != "true":
            return None
        status = subprocess.run(
            base_argv + ["status", "--porcelain=v1", "--untracked-files=normal"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status.returncode != 0:
            return None
        untracked = subprocess.run(
            base_argv + ["ls-files", "--others", "--exclude-standard", "-z"],
            capture_output=True,
            text=False,
            timeout=5,
        )
        if untracked.returncode != 0:
            return None
        untracked_meta: List[tuple[str, str, int, int]] = []
        for raw_path in untracked.stdout.decode("utf-8", errors="ignore").split("\x00"):
            path_text = raw_path.strip()
            if not path_text:
                continue
            candidate = (self.workspace_root / path_text).resolve()
            if not self._is_relative_to(candidate, self.workspace_root):
                continue
            try:
                stat = candidate.lstat()
            except OSError:
                continue
            if candidate.is_dir():
                entry_type = "dir"
            elif candidate.is_symlink():
                entry_type = "symlink"
            else:
                entry_type = "file"
            untracked_meta.append((path_text, entry_type, int(stat.st_size), int(stat.st_mtime_ns)))
        untracked_meta.sort()
        return (status.stdout, tuple(untracked_meta))

    def _workspace_change_snapshot(self) -> tuple[str, Any]:
        git_snapshot = self._git_status_snapshot()
        if git_snapshot is not None:
            return ("git", git_snapshot)
        return ("fingerprint", self.workspace_state_fingerprint())

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
        if self._is_protected_state_path(resolved):
            raise PermissionError("Writing protected internal state is not allowed")
        return resolved

    def _resolve_read_path(self, path: str, extra_allowed_roots: Optional[Iterable[str]] = None) -> Path:
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
            if self._is_protected_state_path(resolved):
                raise PermissionError("Read path targets protected internal state")
            return resolved
        for root_text in extra_allowed_roots or []:
            try:
                allowed_root = Path(os.path.expanduser(str(root_text))).resolve()
            except Exception:
                continue
            if self._is_relative_to(resolved, allowed_root):
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

    def move_path(self, source_path: str, destination_path: str, overwrite: bool = False) -> str:
        source = self._resolve_write_path(source_path)
        destination = self._resolve_write_path(destination_path)

        if not source.exists():
            raise FileNotFoundError(str(source))
        if source == self.workspace_root:
            raise PermissionError("Moving the workspace root is not allowed")

        protected_dir = self.protected_state_dir
        if self._is_protected_state_path(source) or self._is_protected_state_path(destination):
            raise PermissionError("Moving protected internal state is not allowed")
        if source == destination:
            raise ValueError("Source and destination must be different")

        if destination.exists():
            if not overwrite:
                raise FileExistsError(str(destination))
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()

        destination.parent.mkdir(parents=True, exist_ok=True)
        moved = shutil.move(str(source), str(destination))
        return str(Path(moved).resolve())

    def delete_path(self, path: str, recursive: bool = False) -> str:
        raw = self._normalize_workspace_relative(path)
        candidate = (self.workspace_root / raw) if not raw.is_absolute() else raw
        candidate = Path(os.path.abspath(str(candidate)))
        if not self._is_relative_to(candidate, self.workspace_root):
            raise PermissionError("Write path escapes workspace root")

        if candidate == self.workspace_root:
            raise PermissionError("Deleting the workspace root is not allowed")
        if self._is_protected_state_path(candidate):
            raise PermissionError("Deleting protected internal state is not allowed")
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
            if self._is_protected_state_path(child):
                continue
            suffix = "/" if child.is_dir() else ""
            results.append(child.name + suffix)
        return results

    def workspace_tree(self, max_depth: int = 3) -> str:
        lines: List[str] = [f"{self.workspace_root.name}/"]

        def walk(path: Path, prefix: str, depth: int) -> None:
            if depth > max_depth:
                return
            entries = sorted(
                (entry for entry in path.iterdir() if not self._is_protected_state_path(entry)),
                key=lambda p: (p.is_file(), p.name.lower()),
            )
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

    def _search_result_filepath(self, path_text: str) -> str:
        raw = str(path_text or "").strip()
        if not raw:
            return ""
        candidate = Path(raw)
        if candidate.is_absolute():
            return str(candidate)
        try:
            return str((self.workspace_root / candidate).resolve())
        except OSError:
            return str(self.workspace_root / candidate)

    def _search_code_with_rg(
        self,
        target: Path,
        *,
        needle: str,
        glob: Optional[str],
        limit: int,
        case_sensitive: bool,
        fixed_strings: bool,
    ) -> Optional[dict[str, Any]]:
        if not self._is_relative_to(target, self.workspace_root):
            return None
        if not target.exists():
            return {
                "query": needle,
                "path": str(target),
                "glob": glob,
                "count": 0,
                "results": [],
                "truncated": False,
                "backend": "rg",
            }
        if target.is_file() and glob and not fnmatch.fnmatch(target.name, glob):
            return {
                "query": needle,
                "path": str(target),
                "glob": glob,
                "count": 0,
                "results": [],
                "truncated": False,
                "backend": "rg",
            }

        rg_path = shutil.which("rg")
        if not rg_path:
            return None

        argv = [
            rg_path,
            "--json",
            "--line-number",
            "--column",
            "--hidden",
            "--glob",
            "!**/.git/**",
            "--glob",
            "!**/.alphanus/**",
        ]
        if not case_sensitive:
            argv.append("-i")
        if fixed_strings:
            argv.append("-F")
        if glob:
            argv.extend(["--glob", glob])
        argv.extend([needle, str(target)])

        results: List[dict[str, Any]] = []
        truncated = False
        terminated_early = False
        proc = subprocess.Popen(
            argv,
            shell=False,
            cwd=str(self.workspace_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
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
                        "filepath": self._search_result_filepath(str(path_data.get("text", ""))),
                        "line_number": int(data.get("line_number", 0)),
                        "line": str(lines_data.get("text", "")).rstrip("\n"),
                        "submatches": submatches,
                    }
                )
                if len(results) >= limit:
                    truncated = True
                    terminated_early = True
                    proc.terminate()
                    break
            stderr_text = proc.stderr.read() if proc.stderr is not None else ""
            try:
                returncode = proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
                returncode = proc.wait(timeout=1)
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.stderr is not None:
                proc.stderr.close()

        if terminated_early:
            returncode = 0
        # ripgrep uses exit code 2 for IO errors such as unreadable paths.
        # Treat that like a partial search result rather than a hard failure.
        if returncode not in (0, 1, 2):
            raise RuntimeError(stderr_text.strip() or "rg search failed")
        return {
            "query": needle,
            "path": str(target),
            "glob": glob,
            "count": len(results),
            "results": results,
            "truncated": truncated,
            "backend": "rg",
        }

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
        if self._text_mentions_protected_state(needle):
            raise PermissionError("Queries mentioning protected internal state are not allowed")

        target = self._resolve_workspace_subpath(path)
        limit = max(1, int(max_results))
        rg_result = self._search_code_with_rg(
            target,
            needle=needle,
            glob=glob,
            limit=limit,
            case_sensitive=case_sensitive,
            fixed_strings=fixed_strings,
        )
        if rg_result is not None:
            return rg_result

        pattern = needle if case_sensitive else needle.lower()
        results = []
        for file_path in self._iter_searchable_files(target, glob):
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
        if any(self._text_mentions_protected_state(part) for part in argv):
            raise PermissionError("Commands touching protected internal state are not allowed")

        if executable == "uv":
            if len(argv) < 3 or argv[1] != "run" or argv[2] not in SAFE_CHECK_RUNNERS:
                raise PermissionError("run_checks only allows 'uv run' with approved verification runners")
        elif executable not in SAFE_CHECK_RUNNERS:
            raise PermissionError("run_checks only supports approved verification runners")

        # Prefer explicit executable paths, and fall back to `python -m` for
        # Python-native runners when the entrypoint is not on PATH.
        resolved = shutil.which(executable)
        if resolved:
            argv[0] = resolved
        elif executable in PYTHON_MODULE_CHECK_RUNNERS:
            argv = [sys.executable, "-m", executable, *argv[1:]]

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
        if argv[0] in SHELL_WRAPPER_BINARIES and len(argv) >= 3 and argv[1] in {"-c", "-lc"}:
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

    def run_shell_command(
        self,
        command: str,
        timeout_s: int = 30,
        cwd: Optional[str] = None,
        allowed_cwd_roots: Optional[Iterable[str]] = None,
    ) -> dict:
        start = time.perf_counter()
        try:
            self._validate_shell_command(command)
            argv = self._parse_command_argv(command)
            command_kind = self._classify_shell_command(argv)
            target_cwd = self._resolve_read_path(cwd, extra_allowed_roots=allowed_cwd_roots) if cwd else self.workspace_root
            if target_cwd.is_file():
                target_cwd = target_cwd.parent
            if self._is_protected_state_path(target_cwd):
                raise PermissionError("Commands touching protected internal state are not allowed")
            if self._command_touches_protected_state(argv, target_cwd):
                raise PermissionError("Commands touching protected internal state are not allowed")
            snapshot_before: Optional[tuple[str, Any]] = None
            fingerprint_before: Optional[str] = None
            if command_kind == "ambiguous":
                snapshot_before = self._workspace_change_snapshot()
                if snapshot_before[0] == "git":
                    fingerprint_before = self.workspace_state_fingerprint()
            run = self._run_argv(argv, timeout_s=timeout_s, cwd=target_cwd)
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
                    "meta": {"duration_ms": run["duration_ms"], "workspace_changed": False},
                }
            workspace_changed = False
            if command_kind == "mutating":
                workspace_changed = True
            elif command_kind == "ambiguous" and snapshot_before is not None:
                snapshot_after = self._workspace_change_snapshot()
                workspace_changed = snapshot_before != snapshot_after
                if not workspace_changed and snapshot_before[0] == "git" and fingerprint_before is not None:
                    workspace_changed = fingerprint_before != self.workspace_state_fingerprint()
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
                "meta": {"duration_ms": run["duration_ms"], "workspace_changed": workspace_changed},
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
