from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from core.project_command_policy import ProjectCommandPolicy, unwrap_shell_command
from core.project_command_policy import shell_has_boundary as shell_has_approval_boundary
from core.sandbox import SandboxCommand, shell_tokens

NESTED_SHELL_EXECUTABLES = {"sh", "bash", "zsh", "fish", "dash", "ksh", "pwsh", "powershell", "cmd"}
SHELL_WRAPPER_EXECUTABLES = {"env", "command", "builtin", "exec", "time", "nice", "nohup", "sudo"}
HEAVY_WALK_DIRS = frozenset(".alphanus .git .mypy_cache .pytest_cache .ruff_cache .tox .venv __pycache__ build dist node_modules".split())


class ProjectSearchShellMixin:
    project_root: Path
    permission_mode: str
    sandbox_config: Any
    sandbox_runner: Any

    def _resolve_read_path(self, path: str, extra_allowed_roots: Iterable[str] | None = None) -> Path: ...

    @staticmethod
    def _text_mentions_protected_state(text: str) -> bool: ...

    def _is_protected_project_path(self, path: Path) -> bool: ...

    def _shell_command_touches_protected_state(self, command: str, cwd: Path) -> bool: ...

    def _shell_command_touches_protected_project_path(self, command: str, cwd: Path) -> bool: ...

    def shell_cwd_requires_approval(self, cwd: str | None) -> bool: ...

    def shell_command_external_paths(self, command: str) -> tuple[Path, ...]: ...

    @staticmethod
    def shell_command_external_grant_roots(paths: Iterable[Path]) -> tuple[Path, ...]: ...

    def _run_shell_string(
        self,
        command: str,
        *,
        timeout_s: int = 30,
        cwd: Path | None = None,
        extra_roots: Iterable[Path] | None = None,
    ) -> dict[str, Any]:
        return self.sandbox_runner.run(
            SandboxCommand(
                command=command,
                cwd=cwd or self.project_root,
                project_root=self.project_root,
                timeout_s=timeout_s,
                config=self.sandbox_config,
                extra_roots=tuple(extra_roots or ()),
            )
        )

    def _is_searchable_path(self, path: Path) -> bool:
        try:
            resolved = self._resolve_read_path(str(path))
        except (PermissionError, FileNotFoundError):
            return False
        return resolved.exists() and ".git" not in resolved.parts

    def _iter_searchable_files(self, target: Path, glob: str | None) -> Iterable[Path]:
        if target.is_file():
            if self._is_searchable_path(target) and (not glob or fnmatch.fnmatch(target.name, glob)):
                yield target
            return

        for root, dirnames, filenames in os.walk(target, topdown=True):
            root_path = Path(root)
            allowed_dirs = []
            for dirname in dirnames:
                if dirname in HEAVY_WALK_DIRS:
                    continue
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
            return str((self.project_root / candidate).resolve())
        except OSError:
            return str(self.project_root / candidate)

    def _search_result_context(
        self,
        *,
        file_path: str,
        line_number: int,
        before_context: int,
        after_context: int,
        line_cache: dict[str, list[str] | None] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        before_n = max(0, int(before_context))
        after_n = max(0, int(after_context))
        if before_n == 0 and after_n == 0:
            return [], []
        cache_key = str(file_path)
        if line_cache is not None and cache_key in line_cache:
            cached_lines = line_cache[cache_key]
            if cached_lines is None:
                return [], []
            lines = cached_lines
        else:
            try:
                resolved = self._resolve_read_path(file_path)
                loaded_lines = resolved.read_text(encoding="utf-8").splitlines()
            except (PermissionError, FileNotFoundError, UnicodeDecodeError, OSError):
                if line_cache is not None:
                    line_cache[cache_key] = None
                return [], []
            if line_cache is not None:
                line_cache[cache_key] = loaded_lines
            lines = loaded_lines
        try:
            idx = max(0, int(line_number) - 1)
        except Exception:
            return [], []
        before_start = max(0, idx - before_n)
        after_end = min(len(lines), idx + 1 + after_n)
        before_rows = [{"line_number": line_idx + 1, "line": lines[line_idx]} for line_idx in range(before_start, min(idx, len(lines)))]
        after_rows = [{"line_number": line_idx + 1, "line": lines[line_idx]} for line_idx in range(max(idx + 1, 0), after_end)]
        return before_rows, after_rows

    @staticmethod
    def _search_response(
        needle: str,
        target: Path,
        glob: str | None,
        results: list[dict[str, Any]],
        truncated: bool,
        backend: str,
        before_context: int,
        after_context: int,
    ) -> dict[str, Any]:
        return {
            "query": needle,
            "path": str(target),
            "glob": glob,
            "count": len(results),
            "results": results,
            "truncated": truncated,
            "backend": backend,
            "before_context": max(0, int(before_context)),
            "after_context": max(0, int(after_context)),
        }

    def _search_code_with_rg(
        self,
        target: Path,
        *,
        needle: str,
        glob: str | None,
        limit: int,
        case_sensitive: bool,
        fixed_strings: bool,
        before_context: int,
        after_context: int,
    ) -> dict[str, Any] | None:
        if not target.is_relative_to(self.project_root):
            return None
        if not target.exists() or (target.is_file() and glob and not fnmatch.fnmatch(target.name, glob)):
            return self._search_response(needle, target, glob, [], False, "rg", before_context, after_context)

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

        results: list[dict[str, Any]] = []
        filepath_cache: dict[str, str] = {}
        context_line_cache: dict[str, list[str] | None] = {}
        truncated = False
        terminated_early = False
        proc = subprocess.Popen(
            argv,
            shell=False,
            cwd=str(self.project_root),
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
                raw_path = str(path_data.get("text", ""))
                filepath = filepath_cache.get(raw_path)
                if filepath is None:
                    filepath = self._search_result_filepath(raw_path)
                    filepath_cache[raw_path] = filepath
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
                        "filepath": filepath,
                        "line_number": int(data.get("line_number", 0)),
                        "line": str(lines_data.get("text", "")).rstrip("\n"),
                        "submatches": submatches,
                    }
                )
                if before_context > 0 or after_context > 0:
                    before_rows, after_rows = self._search_result_context(
                        file_path=str(results[-1]["filepath"]),
                        line_number=int(results[-1]["line_number"]),
                        before_context=before_context,
                        after_context=after_context,
                        line_cache=context_line_cache,
                    )
                    results[-1]["before_context"] = before_rows
                    results[-1]["after_context"] = after_rows
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
        return self._search_response(needle, target, glob, results, truncated, "rg", before_context, after_context)

    def search_code(
        self,
        query: str,
        *,
        path: str = ".",
        glob: str | None = None,
        max_results: int = 50,
        case_sensitive: bool = False,
        fixed_strings: bool = True,
        before_context: int = 0,
        after_context: int = 0,
    ) -> dict[str, Any]:
        needle = str(query)
        if not needle.strip():
            raise ValueError("search_code query must be non-empty")
        if self._text_mentions_protected_state(needle):
            raise PermissionError("Queries mentioning protected internal state are not allowed")

        target = self._resolve_read_path(path or ".")
        limit = max(1, int(max_results))
        rg_result = self._search_code_with_rg(
            target,
            needle=needle,
            glob=glob,
            limit=limit,
            case_sensitive=case_sensitive,
            fixed_strings=fixed_strings,
            before_context=before_context,
            after_context=after_context,
        )
        if rg_result is not None:
            return rg_result

        if fixed_strings:
            pattern = needle if case_sensitive else needle.lower()
            regex = None
        else:
            try:
                regex = re.compile(needle, 0 if case_sensitive else re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"Invalid search_code regex: {exc}") from exc
            pattern = ""
        results = []
        context_line_cache: dict[str, list[str] | None] = {}
        for file_path in self._iter_searchable_files(target, glob):
            try:
                content = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            content_lines = content.splitlines()
            context_line_cache[str(file_path)] = content_lines
            for line_number, line in enumerate(content_lines, start=1):
                if regex is None:
                    haystack = line if case_sensitive else line.lower()
                    start = haystack.find(pattern)
                    if start < 0:
                        continue
                    end = start + len(needle)
                    match_text = line[start:end]
                else:
                    match = regex.search(line)
                    if match is None:
                        continue
                    start = match.start()
                    end = match.end()
                    match_text = match.group(0)
                results.append(
                    {
                        "filepath": str(file_path),
                        "line_number": line_number,
                        "line": line,
                        "submatches": [{"match": match_text, "start": start, "end": end}],
                    }
                )
                if before_context > 0 or after_context > 0:
                    before_rows, after_rows = self._search_result_context(
                        file_path=str(file_path),
                        line_number=line_number,
                        before_context=before_context,
                        after_context=after_context,
                        line_cache=context_line_cache,
                    )
                    results[-1].update(before_context=before_rows, after_context=after_rows)
                if len(results) >= limit:
                    return self._search_response(needle, target, glob, results, True, "python", before_context, after_context)
        return self._search_response(needle, target, glob, results, False, "python", before_context, after_context)

    @staticmethod
    def shell_command_requires_approval(command: str) -> bool:
        command_text = str(command or "")
        if shell_has_approval_boundary(command_text):
            return True
        argv = shell_tokens(command)
        if not argv:
            return True
        executable, remaining = unwrap_shell_command(argv, SHELL_WRAPPER_EXECUTABLES)
        if not executable:
            return True
        if executable in NESTED_SHELL_EXECUTABLES and any(part == "/c" or (part.startswith("-") and "c" in part) for part in remaining):
            return True
        if executable in {"chmod", "chown"} and any(part == "-R" or part.startswith("-R") for part in remaining):
            return True
        if executable in {"rm", "trash"} and any(
            part in {"-r", "-R", "-rf", "-fr"} or ("r" in part and part.startswith("-")) for part in remaining
        ):
            return True
        if executable == "git":
            subcommand = ProjectCommandPolicy.git_subcommand([executable, *remaining])
            if subcommand == "clean":
                return True
            if subcommand == "reset" and "--hard" in remaining:
                return True
        package_managers = {"npm", "pnpm", "yarn", "pip", "pip3", "uv", "brew", "apt", "apt-get", "dnf", "yum", "cargo"}
        if executable in package_managers and any(part in {"install", "add", "update", "upgrade"} for part in remaining):
            return True
        return False

    def run_shell_command(
        self,
        command: str,
        timeout_s: int = 30,
        cwd: str | None = None,
        allowed_cwd_roots: Iterable[str] | None = None,
        approved: bool = False,
    ) -> dict:
        start = time.perf_counter()
        try:
            command_text = command.strip()
            if not command_text:
                raise PermissionError("Empty command is not allowed")
            target_cwd = self._resolve_read_path(cwd, extra_allowed_roots=allowed_cwd_roots) if cwd else self.project_root
            if target_cwd.is_file():
                target_cwd = target_cwd.parent
            if self._is_protected_project_path(target_cwd):
                raise PermissionError("Commands touching protected project paths are not allowed")
            if self._shell_command_touches_protected_state(command_text, target_cwd):
                raise PermissionError("Commands touching protected internal state are not allowed")
            if self._shell_command_touches_protected_project_path(command_text, target_cwd):
                raise PermissionError("Commands touching protected project paths are not allowed")
            if (
                self.permission_mode == "read-only"
                and ProjectCommandPolicy.classify_shell_command(shell_tokens(command_text)) != "readonly"
            ):
                raise PermissionError("Mutating shell commands are not allowed in read-only mode")
            external_cwd = self.shell_cwd_requires_approval(str(target_cwd))
            external_paths = () if self.permission_mode == "danger-full-access" else self.shell_command_external_paths(command_text)
            if self.permission_mode != "danger-full-access" and external_cwd and not approved:
                raise PermissionError("Shell command outside the project requires approval before execution")
            if self.permission_mode != "danger-full-access" and external_paths and not approved:
                raise PermissionError("Shell command accessing paths outside the project requires approval before execution")
            if self.permission_mode != "danger-full-access" and self.shell_command_requires_approval(command_text) and not approved:
                raise PermissionError("Shell command requires approval before execution")
            extra_roots = [target_cwd] if external_cwd else []
            extra_roots.extend(self.shell_command_external_grant_roots(external_paths))
            run = self._run_shell_string(command_text, timeout_s=timeout_s, cwd=target_cwd, extra_roots=extra_roots)
            sandbox_error = run.get("sandbox_error")
            if isinstance(sandbox_error, dict):
                return {
                    "ok": False,
                    "data": None,
                    "error": {
                        "code": str(sandbox_error.get("code", "E_SANDBOX_SETUP")),
                        "message": str(sandbox_error.get("message", "Sandbox setup failed")),
                    },
                    "meta": {"duration_ms": int(run.get("duration_ms", 0)), "project_changed": False},
                }
            # Shell commands do not provide a reliable list of touched paths.
            # Avoid O(repository-size) before/after fingerprints and report the
            # conservative mutation classification to downstream auditing.
            project_changed = ProjectCommandPolicy.classify_shell_command(
                shell_tokens(command_text)
            ) == "mutating" or shell_has_approval_boundary(command_text)
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
                    "meta": {"duration_ms": run["duration_ms"], "project_changed": project_changed},
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
                "meta": {"duration_ms": run["duration_ms"], "project_changed": project_changed},
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "data": None,
                "error": {"code": "E_TIMEOUT", "message": f"Command did not finish within {timeout_s}s"},
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
