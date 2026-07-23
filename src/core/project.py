from __future__ import annotations

import fnmatch
import glob as globlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from core.project_command_policy import ProjectCommandPolicy, shell_has_boundary, unwrap_shell_command
from core.sandbox import SandboxCommand, SandboxConfig, SandboxRunner, shell_tokens

DEFAULT_BLOCKED_PATTERNS = ".alphanus .ssh .aws .gnupg id_rsa id_ed25519 .env .bash_history .zsh_history".split()

MAX_TOOL_TEXT_BYTES = 20000
PROTECTED_STATE_TOKEN_RE = re.compile(r"(^|[^A-Za-z0-9._-])\.alphanus(?=$|[/\\\\]|[^A-Za-z0-9._-])", re.IGNORECASE)
NESTED_SHELL_EXECUTABLES = {"sh", "bash", "zsh", "fish", "dash", "ksh", "pwsh", "powershell", "cmd"}
SHELL_WRAPPER_EXECUTABLES = {"env", "command", "builtin", "exec", "time", "nice", "nohup", "sudo"}
HEAVY_WALK_DIRS = frozenset(".alphanus .git .mypy_cache .pytest_cache .ruff_cache .tox .venv __pycache__ build dist node_modules".split())
RESTRICTED_SYSTEM_ROOTS = tuple(map(Path, ("/etc", "/var", "/System", "/bin", "/usr")))
TRUSTED_RUNTIME_ROOTS = tuple(map(Path, ("/bin", "/sbin", "/usr", "/System", "/Library", "/opt", "/lib")))


shell_has_approval_boundary = shell_has_boundary


class ProjectRuntime:
    def __init__(
        self,
        project_root: str,
        blocked_patterns: Iterable[str] | None = None,
        *,
        permission_mode: str = "project-write",
        network_access: bool = False,
        sandbox_backend: str = "auto",
        sandbox_fail_closed: bool = True,
    ) -> None:
        self.project_root = Path(os.path.expanduser(project_root)).resolve()
        if not self.project_root.exists():
            raise FileNotFoundError(f"Project root does not exist: {self.project_root}")
        if not self.project_root.is_dir():
            raise NotADirectoryError(f"Project root is not a directory: {self.project_root}")
        self.blocked_patterns = list(blocked_patterns or DEFAULT_BLOCKED_PATTERNS)
        self.permission_mode = (
            permission_mode if permission_mode in {"read-only", "project-write", "danger-full-access"} else "project-write"
        )
        self.sandbox_config = SandboxConfig(
            mode=self.permission_mode,
            network=bool(network_access),
            backend=sandbox_backend or "auto",
            fail_closed=bool(sandbox_fail_closed),
        )
        self.sandbox_runner = SandboxRunner()
        self._filesystem_case_insensitive = self._detect_case_insensitive_filesystem()

    @property
    def protected_state_dir(self) -> Path:
        return self.project_root / ".alphanus"

    def _detect_case_insensitive_filesystem(self) -> bool:
        root = self.project_root
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

    @staticmethod
    def _resolve_loose(path: Path) -> Path:
        try:
            return path.resolve(strict=False)
        except OSError:
            return Path(os.path.abspath(str(path)))

    def _is_outside_project(self, path: Path) -> bool:
        candidate = self._resolve_loose(path)
        return not self._is_path_equal_or_descendant(candidate, self.project_root)

    def _is_protected_state_path(self, path: Path) -> bool:
        candidate = self._resolve_loose(path)
        if any(part.lower() == ".alphanus" for part in candidate.parts):
            return True
        protected = self._resolve_loose(self.protected_state_dir)
        return self._is_path_equal_or_descendant(candidate, protected)

    def _is_protected_project_path(self, path: Path) -> bool:
        if self._is_protected_state_path(path):
            return True
        git_dir = self.project_root / ".git"
        candidate = self._resolve_loose(path)
        resolved_git = self._resolve_loose(git_dir)
        return (
            self._is_path_equal_or_descendant(candidate, resolved_git)
            or any(part.lower() == ".git" for part in candidate.parts)
            or self._is_secret_path(candidate)
        )

    def sandbox_preflight(self) -> dict[str, Any]:
        return self.sandbox_runner.preflight(self.sandbox_config)

    def _ensure_can_write_project(self) -> None:
        if self.permission_mode == "read-only":
            raise PermissionError("Project is read-only in the active permission mode")

    def _resolve_command_token_path(self, token: str, cwd: Path) -> Path | None:
        value = str(token or "").strip()
        if not value or value == "-":
            return None
        pathish = value.startswith((".", "/", "~")) or "/" in value or "\\" in value
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
        return self._resolve_loose(candidate)

    def _resolve_globbed_command_token_paths(self, token: str, cwd: Path) -> list[Path]:
        value = str(token or "").strip()
        if not value or not globlib.has_magic(value):
            return []
        raw = Path(os.path.expanduser(value))
        pattern = raw if raw.is_absolute() else cwd / raw
        matches = globlib.glob(str(pattern), recursive=True)
        resolved: list[Path] = []
        for match in matches:
            resolved.append(self._resolve_loose(Path(match)))
        return resolved

    @staticmethod
    def _text_mentions_protected_state(text: str) -> bool:
        return bool(PROTECTED_STATE_TOKEN_RE.search(str(text)))

    def _command_touches_path(self, argv: list[str], cwd: Path, predicate, *, inspect_text: bool = False) -> bool:
        git_subcommand_index = ProjectCommandPolicy.git_subcommand_index(argv) if argv and Path(argv[0]).name == "git" else -1
        for idx, part in enumerate(argv[1:], start=1):
            if inspect_text and self._text_mentions_protected_state(part):
                return True
            if idx == git_subcommand_index:
                continue
            for expanded in self._resolve_globbed_command_token_paths(part, cwd):
                if predicate(expanded):
                    return True
            resolved = self._resolve_command_token_path(part, cwd)
            if resolved is not None and predicate(resolved):
                return True
        return False

    def _command_touches_protected_state(self, argv: list[str], cwd: Path) -> bool:
        return self._command_touches_path(argv, cwd, self._is_protected_state_path, inspect_text=True)

    def _command_touches_protected_project_path(self, argv: list[str], cwd: Path) -> bool:
        return self._command_touches_path(argv, cwd, self._is_protected_project_path)

    def _shell_command_touches_protected_state(self, command: str, cwd: Path) -> bool:
        if self._text_mentions_protected_state(command):
            return True
        try:
            argv = shlex.split(command, posix=True)
        except ValueError:
            return False
        return self._command_touches_protected_state(argv, cwd)

    def _shell_command_touches_protected_project_path(self, command: str, cwd: Path) -> bool:
        try:
            argv = shlex.split(command, posix=True)
        except ValueError:
            return False
        return self._command_touches_protected_project_path(argv, cwd)

    def _normalize_project_relative(self, path: str) -> Path:
        raw = Path(os.path.expanduser(path))
        if raw.is_absolute():
            parts = raw.parts
            if len(parts) > 2 and parts[1] == self.project_root.name:
                return Path(*parts[2:])
            return raw
        parts = raw.parts
        if len(parts) > 1 and parts[0] == self.project_root.name:
            return Path(*parts[1:])
        return raw

    def _resolve_write_path(self, path: str) -> Path:
        self._ensure_can_write_project()
        raw = self._normalize_project_relative(path)
        candidate = (self.project_root / raw) if not raw.is_absolute() else raw
        lexical_candidate = Path(os.path.abspath(str(candidate)))
        resolved = candidate.resolve()
        explicit_external = raw.is_absolute() and not self._is_path_equal_or_descendant(lexical_candidate, self.project_root)
        if not resolved.is_relative_to(self.project_root) and not explicit_external:
            raise PermissionError("Write path escapes project root")
        if self._is_protected_project_path(resolved):
            raise PermissionError("Writing protected internal state or project paths is not allowed")
        return resolved

    def write_path_requires_approval(self, path: str, *, overwrite: bool = False) -> bool:
        target = self._resolve_write_path(path)
        return self._is_outside_project(target)

    def delete_path_requires_approval(self, path: str, *, recursive: bool = False) -> bool:
        target = self._resolve_write_path(path)
        if target == self.project_root or target == Path(target.anchor):
            return False
        return bool(recursive or self._is_outside_project(target))

    def move_path_requires_approval(self, source_path: str, destination_path: str, *, overwrite: bool = False) -> bool:
        source = self._resolve_write_path(source_path)
        destination = self._resolve_write_path(destination_path)
        return bool(overwrite or self._is_outside_project(source) or self._is_outside_project(destination))

    def shell_cwd_requires_approval(self, cwd: str | None) -> bool:
        if not cwd:
            return False
        target = self._resolve_read_path(cwd, extra_allowed_roots=[cwd])
        if target.is_file():
            target = target.parent
        return self._is_outside_project(target)

    def shell_command_external_paths(self, command: str) -> tuple[Path, ...]:
        # Find approval-boundary paths, retaining non-existent creation targets.
        external_paths: list[Path] = []
        seen: set[str] = set()
        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()")
            lexer.whitespace_split = True
            lexer.commenters = ""
            tokens = list(lexer)
        except ValueError:
            tokens = shell_tokens(command)
        for token in tokens:
            if token and all(character in ";&|<>()" for character in token):
                continue
            candidate_text = token.split("=", 1)[1] if "=" in token else token
            if candidate_text.startswith("~/"):
                candidate_text = os.path.expanduser(candidate_text)
            candidate = Path(candidate_text)
            if not candidate.is_absolute():
                continue
            resolved = candidate.resolve(strict=False)
            if resolved.is_relative_to(self.project_root):
                continue
            if any(resolved.is_relative_to(root) for root in TRUSTED_RUNTIME_ROOTS):
                continue
            if resolved.exists():
                # Reuse normal read validation so protected and restricted paths
                # do not become accessible merely because they appeared in a command.
                resolved = self._resolve_read_path(str(resolved), extra_allowed_roots=[str(resolved)])
            else:
                if any(resolved.is_relative_to(root) for root in RESTRICTED_SYSTEM_ROOTS):
                    raise PermissionError("Shell command path is in restricted system location")
                if self._is_protected_project_path(resolved):
                    raise PermissionError("Shell command path targets protected internal state or project paths")
            key = str(resolved)
            if key not in seen:
                seen.add(key)
                external_paths.append(resolved)
        return tuple(external_paths)

    @staticmethod
    def shell_command_external_grant_roots(paths: Iterable[Path]) -> tuple[Path, ...]:
        # Missing targets bind through their nearest existing parent.
        roots: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            target = Path(path).resolve(strict=False)
            grant = target
            while not grant.exists() and grant != grant.parent:
                grant = grant.parent
            if not grant.exists():
                continue
            key = str(grant.resolve())
            if key not in seen:
                seen.add(key)
                roots.append(Path(key))
        return tuple(roots)

    def _resolve_read_path(self, path: str, extra_allowed_roots: Iterable[str] | None = None) -> Path:
        raw = self._normalize_project_relative(path)
        candidate = (self.project_root / raw) if not raw.is_absolute() else raw
        lexical_candidate = Path(os.path.abspath(str(candidate)))
        resolved = candidate.resolve()

        # Block core system trees.
        if any(resolved.is_relative_to(root) for root in RESTRICTED_SYSTEM_ROOTS):
            raise PermissionError("Read path is in restricted system location")

        if resolved.is_relative_to(self.project_root):
            if self._is_protected_project_path(resolved):
                raise PermissionError("Read path targets protected internal state or project paths")
            return resolved
        explicit_external = raw.is_absolute() and not self._is_path_equal_or_descendant(lexical_candidate, self.project_root)
        if explicit_external:
            if self._is_protected_project_path(resolved):
                raise PermissionError("Read path targets protected internal state or project paths")
            return resolved
        for root_text in extra_allowed_roots or []:
            try:
                allowed_root = Path(os.path.expanduser(str(root_text))).resolve()
            except Exception:
                continue
            if resolved.is_relative_to(allowed_root):
                return resolved
        raise PermissionError("Read path escapes project root")

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

    def read_file(self, path: str) -> str:
        target = self._resolve_read_path(path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(str(target))
        return target.read_text(encoding="utf-8")

    @staticmethod
    def _line_count(text: str) -> int:
        return text.count("\n") + (1 if text else 0)

    @staticmethod
    def _find_unique_anchor_line(lines: list[str], anchor: str, *, field_name: str) -> int:
        needle = str(anchor)
        if not needle:
            raise ValueError(f"{field_name} must be non-empty when provided")
        matches = [idx + 1 for idx, line in enumerate(lines) if needle in line]
        if not matches:
            raise ValueError(f"{field_name} was not found in the file")
        if len(matches) > 1:
            raise ValueError(f"{field_name} matched multiple lines; provide a more specific anchor")
        return matches[0]

    def resolve_text_section(
        self,
        content: str,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
        before_anchor: str = "",
        after_anchor: str = "",
    ) -> dict[str, Any]:
        lines_plain = content.split("\n") if content else []
        total_line_count = self._line_count(content)
        line_start_offsets: list[int] = []
        if content:
            line_start_offsets.append(0)
            for idx, ch in enumerate(content):
                if ch == "\n":
                    line_start_offsets.append(idx + 1)
            # Keep resolver line semantics aligned with tool metadata line_count.
            # This includes the trailing empty logical line for newline-terminated files.
            if len(line_start_offsets) != total_line_count:
                raise RuntimeError("Failed to resolve logical line offsets")
        selectors_applied = start_line is not None or end_line is not None or bool(before_anchor) or bool(after_anchor)

        if total_line_count == 0:
            if selectors_applied:
                raise ValueError("Section selectors cannot be resolved because the file is empty")
            return {
                "content": "",
                "start_line": 0,
                "end_line": 0,
                "total_line_count": 0,
                "start_offset": 0,
                "end_offset": 0,
                "before_anchor_line": 0,
                "after_anchor_line": 0,
                "selectors_applied": False,
            }

        resolved_start = int(start_line) if start_line is not None else 1
        resolved_end = int(end_line) if end_line is not None else total_line_count

        before_anchor_line = 0
        after_anchor_line = 0
        if before_anchor:
            before_anchor_line = self._find_unique_anchor_line(lines_plain, before_anchor, field_name="before_anchor")
            if start_line is None:
                resolved_start = before_anchor_line + 1
            elif resolved_start <= before_anchor_line:
                raise ValueError("start_line must be after before_anchor")
        if after_anchor:
            after_anchor_line = self._find_unique_anchor_line(lines_plain, after_anchor, field_name="after_anchor")
            if end_line is None:
                resolved_end = after_anchor_line - 1
            elif resolved_end >= after_anchor_line:
                raise ValueError("end_line must be before after_anchor")

        if before_anchor_line and after_anchor_line and before_anchor_line >= after_anchor_line:
            raise ValueError("before_anchor must appear before after_anchor")
        if resolved_start < 1 or resolved_start > total_line_count:
            raise ValueError("start_line is out of range")
        if resolved_end < 1 or resolved_end > total_line_count:
            raise ValueError("end_line is out of range")
        if resolved_start > resolved_end:
            raise ValueError("Resolved section is empty; adjust line bounds or anchors")

        start_offset = line_start_offsets[resolved_start - 1]
        if resolved_end < total_line_count:
            end_offset = line_start_offsets[resolved_end]
        else:
            end_offset = len(content)
        section_content = content[start_offset:end_offset]
        return {
            "content": section_content,
            "start_line": resolved_start,
            "end_line": resolved_end,
            "total_line_count": total_line_count,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "before_anchor_line": before_anchor_line,
            "after_anchor_line": after_anchor_line,
            "selectors_applied": selectors_applied,
        }

    def create_file(self, filepath: str, content: str, *, approved: bool = False) -> str:
        target = self._resolve_write_path(filepath)
        if self.write_path_requires_approval(filepath, overwrite=True) and self.permission_mode != "danger-full-access" and not approved:
            raise PermissionError("External file overwrite requires approval before execution")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return str(target)

    def create_directory(self, path: str, *, approved: bool = False) -> str:
        target = self._resolve_write_path(path)
        if self.write_path_requires_approval(path) and self.permission_mode != "danger-full-access" and not approved:
            raise PermissionError("External directory creation requires approval before execution")
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    def edit_file(self, filepath: str, content: str, *, approved: bool = False) -> str:
        target = self._resolve_write_path(filepath)
        if self.write_path_requires_approval(filepath, overwrite=True) and self.permission_mode != "danger-full-access" and not approved:
            raise PermissionError("External file edit requires approval before execution")
        if not target.exists():
            raise FileNotFoundError(str(target))
        target.write_text(content, encoding="utf-8")
        return str(target)

    def move_path(self, source_path: str, destination_path: str, overwrite: bool = False, *, approved: bool = False) -> str:
        source = self._resolve_write_path(source_path)
        destination = self._resolve_write_path(destination_path)

        if not source.exists():
            raise FileNotFoundError(str(source))
        if source == self.project_root:
            raise PermissionError("Moving the project root is not allowed")

        if self._is_protected_project_path(source) or self._is_protected_project_path(destination):
            raise PermissionError("Moving protected internal state or project paths is not allowed")
        if source == destination:
            raise ValueError("Source and destination must be different")
        if (
            self.move_path_requires_approval(source_path, destination_path, overwrite=overwrite)
            and self.permission_mode != "danger-full-access"
            and not approved
        ):
            raise PermissionError("Move crosses the project-write approval boundary")

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

    def delete_path(self, path: str, recursive: bool = False, *, approved: bool = False) -> str:
        target = self._resolve_write_path(path)
        if target == self.project_root:
            raise PermissionError("Deleting the project root is not allowed")
        if target == Path(target.anchor):
            raise PermissionError("Deleting the filesystem root is not allowed")
        if self.delete_path_requires_approval(path, recursive=recursive) and self.permission_mode != "danger-full-access" and not approved:
            raise PermissionError("Delete crosses the project-write approval boundary")
        if target.is_symlink():
            target.unlink()
            return str(target)

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

    def delete_path_metadata(self, path: str) -> dict[str, Any]:
        target = self._resolve_write_path(path)
        if target == self.project_root:
            raise PermissionError("Deleting the project root is not allowed")
        if target == Path(target.anchor):
            raise PermissionError("Deleting the filesystem root is not allowed")

        if target.is_symlink():
            stat = target.lstat()
            return {"is_dir": False, "size_bytes": stat.st_size, "file_count": 1}

        if not target.exists():
            raise FileNotFoundError(str(target))
        if not target.is_dir():
            return {"is_dir": False, "size_bytes": target.stat().st_size, "file_count": 1}

        size_bytes = 0
        file_count = 0
        for child in target.rglob("*"):
            try:
                if child.is_symlink():
                    size_bytes += child.lstat().st_size
                    file_count += 1
                elif child.is_file():
                    size_bytes += child.stat().st_size
                    file_count += 1
            except OSError:
                continue
        return {"is_dir": True, "size_bytes": size_bytes, "file_count": file_count}

    def list_files(self, path: str = ".") -> list[str]:
        target = self._resolve_read_path(path)
        if not target.exists() or not target.is_dir():
            raise FileNotFoundError(str(target))
        results = []
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if self._is_protected_state_path(child):
                continue
            suffix = "/" if child.is_dir() and not child.is_symlink() else ""
            results.append(child.name + suffix)
        return results

    def _display_path_label(self, path: Path) -> str:
        try:
            if path.is_relative_to(self.project_root):
                rel = path.relative_to(self.project_root)
                if str(rel) == ".":
                    return f"{self.project_root.name}/"
                return str(rel).rstrip("/") + ("/" if path.is_dir() and not path.is_symlink() else "")
        except ValueError:
            pass
        return path.name + ("/" if path.is_dir() and not path.is_symlink() else "")

    def _is_project_tree_visible_path(self, path: Path) -> bool:
        if self._is_protected_project_path(path):
            return False
        try:
            resolved = path.resolve(strict=False)
        except OSError:
            return False
        if resolved.is_relative_to(self.project_root):
            return resolved.exists()
        lexical = Path(os.path.abspath(str(path)))
        return path.is_absolute() and not self._is_path_equal_or_descendant(lexical, self.project_root) and resolved.exists()

    def project_tree(self, path: str = ".", max_depth: int = 3, max_entries: int = 500) -> dict[str, Any]:
        target = self._resolve_read_path(path or ".")
        if not target.exists():
            raise FileNotFoundError(str(target))
        if not target.is_dir():
            raise NotADirectoryError(str(target))
        max_depth = max(1, int(max_depth))
        max_entries = max(1, int(max_entries))
        lines: list[str] = [self._display_path_label(target)]
        emitted = 0
        omitted = 0
        truncated = False

        def walk(current: Path, prefix: str, depth: int) -> None:
            nonlocal emitted, omitted, truncated
            if depth > max_depth:
                return
            if emitted >= max_entries:
                truncated = True
                return
            entries = sorted(
                (
                    entry
                    for entry in current.iterdir()
                    if self._is_project_tree_visible_path(entry)
                    and (not entry.is_dir() or entry.is_symlink() or entry.name not in HEAVY_WALK_DIRS)
                ),
                key=lambda p: (p.is_file(), p.name.lower()),
            )
            for idx, entry in enumerate(entries):
                if emitted >= max_entries:
                    truncated = True
                    omitted += len(entries) - idx
                    break
                last = idx == len(entries) - 1
                conn = "└── " if last else "├── "
                is_walkable_dir = entry.is_dir() and not entry.is_symlink()
                label = entry.name + ("/" if is_walkable_dir else "")
                lines.append(prefix + conn + label)
                emitted += 1
                if is_walkable_dir:
                    walk(entry, prefix + ("    " if last else "│   "), depth + 1)

        walk(target, "", 1)
        return {
            "path": str(target),
            "tree": "\n".join(lines),
            "max_depth": max_depth,
            "max_entries": max_entries,
            "truncated": truncated,
            "omitted_entries": omitted,
        }

    def _project_relative_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return ""

    def find_files(
        self,
        *,
        path: str = ".",
        name: str = "",
        glob: str = "",
        include_dirs: bool = False,
        case_sensitive: bool = False,
        max_results: int = 50,
    ) -> dict[str, Any]:
        if not str(name or "").strip() and not str(glob or "").strip():
            raise ValueError("find_files requires name or glob")
        target = self._resolve_read_path(path or ".")
        if not target.exists():
            raise FileNotFoundError(str(target))
        limit = max(1, int(max_results))
        name_query = str(name or "").strip()
        glob_query = str(glob or "").strip()
        glob_cmp = glob_query if case_sensitive else glob_query.lower()
        cmp_name = name_query if case_sensitive else name_query.lower()
        results: list[dict[str, Any]] = []
        truncated = False

        def matches(candidate: Path) -> bool:
            rel = self._project_relative_path(candidate) or str(candidate)
            rel_cmp = rel if case_sensitive else rel.lower()
            basename = candidate.name if case_sensitive else candidate.name.lower()
            if cmp_name and cmp_name not in {basename, rel_cmp} and cmp_name not in rel_cmp:
                return False
            glob_rel = rel if case_sensitive else rel.lower()
            glob_name = candidate.name if case_sensitive else candidate.name.lower()
            if glob_cmp and not fnmatch.fnmatch(glob_rel, glob_cmp) and not fnmatch.fnmatch(glob_name, glob_cmp):
                return False
            return True

        def add_candidate(candidate: Path) -> bool:
            nonlocal truncated
            try:
                if not self._is_searchable_path(candidate):
                    return False
            except OSError:
                return False
            is_dir = candidate.is_dir() and not candidate.is_symlink()
            if is_dir and not include_dirs:
                return False
            if not matches(candidate):
                return False
            if len(results) >= limit:
                truncated = True
                return True
            results.append(
                {
                    "filepath": str(candidate),
                    "relative_path": self._project_relative_path(candidate),
                    "basename": candidate.name,
                    "parent": str(candidate.parent),
                    "kind": "directory" if is_dir else "file",
                }
            )
            return False

        if target.is_file():
            add_candidate(target)
        else:
            for root, dirnames, filenames in os.walk(target, topdown=True):
                root_path = Path(root)
                allowed_dirs = []
                for dirname in dirnames:
                    if dirname in HEAVY_WALK_DIRS:
                        continue
                    candidate_dir = root_path / dirname
                    if not self._is_searchable_path(candidate_dir):
                        continue
                    if include_dirs and add_candidate(candidate_dir):
                        break
                    allowed_dirs.append(dirname)
                else:
                    dirnames[:] = allowed_dirs
                    for filename in filenames:
                        if add_candidate(root_path / filename):
                            break
                    if truncated:
                        break
                    continue
                truncated = True
                dirnames[:] = []
                break
        return {
            "path": str(target),
            "query": name_query,
            "glob": glob_query or None,
            "include_dirs": bool(include_dirs),
            "case_sensitive": bool(case_sensitive),
            "count": len(results),
            "results": results,
            "truncated": truncated,
        }

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

    def _resolve_project_subpath(self, path: str) -> Path:
        return self._resolve_read_path(path or ".")

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

        target = self._resolve_project_subpath(path)
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

    def _validate_shell_command(self, command: str) -> str:
        trimmed = command.strip()
        if not trimmed:
            raise PermissionError("Empty command is not allowed")
        return trimmed

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
            command_text = self._validate_shell_command(command)
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
