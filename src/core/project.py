from __future__ import annotations

import fnmatch
import glob as globlib
import os
import re
import shlex
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from core.project_command_policy import ProjectCommandPolicy
from core.project_search import HEAVY_WALK_DIRS, ProjectSearchShellMixin
from core.sandbox import SandboxConfig, SandboxRunner, shell_tokens

DEFAULT_BLOCKED_PATTERNS = ".alphanus .ssh .aws .gnupg id_rsa id_ed25519 .env .bash_history .zsh_history".split()

MAX_TOOL_TEXT_BYTES = 20000
PROTECTED_STATE_TOKEN_RE = re.compile(r"(^|[^A-Za-z0-9._-])\.alphanus(?=$|[/\\\\]|[^A-Za-z0-9._-])", re.IGNORECASE)
RESTRICTED_SYSTEM_ROOTS = tuple(map(Path, ("/etc", "/var", "/System", "/bin", "/usr")))
TRUSTED_RUNTIME_ROOTS = tuple(map(Path, ("/bin", "/sbin", "/usr", "/System", "/Library", "/opt", "/lib")))


class ProjectRuntime(ProjectSearchShellMixin):
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
