from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.markup import escape as esc

WriteFn = Callable[[str], None]
WriteIndentedFn = Callable[[str, int], None]
WriteCodeFn = Callable[[list[str], str | None, int], None]
UpdatePreviewFn = Callable[[list[str], str | None], None]
ClearPreviewFn = Callable[[], None]

LANGUAGE_BY_SUFFIX = {
    ".bash": "bash",
    ".c": "c",
    ".cc": "cpp",
    ".cjs": "javascript",
    ".cpp": "cpp",
    ".css": "css",
    ".cxx": "cpp",
    ".h": "c",
    ".hh": "cpp",
    ".htm": "html",
    ".html": "html",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".js": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".mjs": "javascript",
    ".py": "python",
    ".sh": "bash",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".zsh": "bash",
}


@dataclass(slots=True)
class LivePreviewState:
    name: str
    filepath: str = ""
    item_index: int = 0
    opened: bool = False
    closed: bool = False
    flushed_item_count: int = 0
    line_buf: str = ""
    preview_lines: list[str] = field(default_factory=list)
    rendered_filepaths: set[str] = field(default_factory=set)
    raw_len: int = 0


class LiveToolPreviewManager:
    def __init__(
        self,
        streamed_file_tools: set[str] | None = None,
        max_static_preview_chars: int = 8000,
        max_static_preview_lines: int = 140,
    ) -> None:
        self.streamed_file_tools = set(streamed_file_tools or {"create_file", "edit_file"})
        self.draft_preview_tools = set(self.streamed_file_tools)
        self.max_static_preview_chars = int(max_static_preview_chars)
        self.max_static_preview_lines = int(max_static_preview_lines)
        self._streams: dict[str, LivePreviewState] = {}
        self._label_color = "dim"
        self._muted_color = "dim"

    @staticmethod
    def canonical_preview_tool_name(tool_name: str) -> str:
        name = str(tool_name or "").strip()
        suffix = name.split(":")[-1].split(".")[-1]
        if suffix == "write_file":
            return "create_file"
        if suffix in {"create_file", "edit_file"}:
            return suffix
        return suffix or name

    def supports_draft_preview(self, tool_name: str) -> bool:
        return self.canonical_preview_tool_name(tool_name) in self.draft_preview_tools

    def supports_result_preview(self, tool_name: str, result: Any) -> bool:
        if self.canonical_preview_tool_name(tool_name) != "shell_command":
            return False
        if not isinstance(result, dict):
            return False
        data = result.get("data")
        return isinstance(data, dict) and any(key in data for key in ("command", "stdout", "stderr", "returncode"))

    def set_theme_colors(self, *, label_color: str = "", muted_color: str = "") -> None:
        self._label_color = str(label_color or "dim")
        self._muted_color = str(muted_color or self._label_color or "dim")

    def reset(self) -> None:
        self._streams = {}

    def _label_markup(self, label: str, value: str) -> str:
        return f"[{self._label_color}]  · {label}: {esc(value)}[/{self._label_color}]"

    def _muted_markup(self, text: str) -> str:
        return f"[{self._muted_color}]{esc(text)}[/{self._muted_color}]"

    @staticmethod
    def _content_is_compacted_history(content: str) -> bool:
        return "\n...[history excerpt;" in content or "\n...[compacted]" in content

    def _clipped_preview_lines(self, content: str) -> tuple[list[str], bool]:
        clipped = content[: self.max_static_preview_chars]
        split_lines = clipped.splitlines()
        lines = split_lines[: self.max_static_preview_lines]
        clipped_for_display = len(content) > self.max_static_preview_chars or len(split_lines) > self.max_static_preview_lines
        return lines, clipped_for_display

    def _read_workspace_preview(self, workspace_root: str | Path | None, filepath: str) -> str | None:
        if workspace_root is None or not filepath:
            return None
        root = Path(workspace_root).expanduser().resolve()
        raw_path = Path(filepath).expanduser()
        candidate = raw_path if raw_path.is_absolute() else root / raw_path
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            return None
        if not resolved.is_file():
            return None
        try:
            with resolved.open("r", encoding="utf-8", errors="replace") as handle:
                return handle.read(self.max_static_preview_chars + 1)
        except OSError:
            return None

    @staticmethod
    def _reset_stream_state(state: LivePreviewState) -> None:
        state.filepath = ""
        state.item_index = 0
        state.opened = False
        state.closed = False
        state.flushed_item_count = 0
        state.line_buf = ""
        state.preview_lines = []
        state.raw_len = 0

    def _set_preview_content(
        self,
        state: LivePreviewState,
        content: str,
        write: WriteFn,
        update_preview: UpdatePreviewFn,
    ) -> bool:
        if not content:
            return False

        if not state.opened and state.filepath:
            state.opened = True
            write(self._label_markup("file draft", state.filepath))

        lines = content.split("\n")
        if content.endswith("\n"):
            state.preview_lines = lines[:-1]
            state.line_buf = ""
        else:
            state.preview_lines = lines[:-1]
            state.line_buf = lines[-1] if lines else ""

        preview_text = "\n".join(state.preview_lines)
        if state.line_buf:
            preview_text = f"{preview_text}\n{state.line_buf}" if preview_text else state.line_buf
        preview_lines, _ = self._clipped_preview_lines(preview_text)
        if preview_lines:
            update_preview(preview_lines, self._guess_language(state.filepath))
            return True
        return False

    def _preview_single_file(
        self,
        state: LivePreviewState,
        filepath: str,
        content: str,
        write: WriteFn,
        update_preview: UpdatePreviewFn,
    ) -> bool:
        state.filepath = filepath or state.filepath
        return self._set_preview_content(state, content, write, update_preview)

    def compact_tool_args(self, tool_name: str, args: Any) -> str:
        if not isinstance(args, dict):
            return str(args)

        canonical_name = self.canonical_preview_tool_name(tool_name)
        if canonical_name == "edit_file":
            filepath = str(args.get("filepath", ""))
            if "content" in args:
                content = args.get("content", "")
                n_chars = len(content) if isinstance(content, str) else 0
                return f'filepath="{filepath}", content={n_chars} chars'
            if "old_string" in args and "new_string" in args:
                replace_all = bool(args.get("replace_all", False))
                mode = "replace_all" if replace_all else "replace_one"
                return f'filepath="{filepath}", mode={mode}'
            return f'filepath="{filepath}"'

        if canonical_name in self.streamed_file_tools:
            filepath = str(args.get("filepath", ""))
            content = args.get("content", "")
            n_chars = len(content) if isinstance(content, str) else 0
            return f'filepath="{filepath}", content={n_chars} chars'

        parts = []
        for key in sorted(args.keys()):
            value = args[key]
            text = str(value)
            if len(text) > 60:
                text = text[:57] + "..."
            parts.append(f"{key}={text}")
        return ", ".join(parts)

    def update(
        self,
        stream_id: str,
        name: str,
        raw_arguments: str,
        write: WriteFn,
        update_preview: UpdatePreviewFn,
        write_indented: WriteIndentedFn | None = None,
        write_code: WriteCodeFn | None = None,
        clear_preview: ClearPreviewFn | None = None,
    ) -> bool:
        _ = (write_indented, write_code, clear_preview)
        canonical_name = self.canonical_preview_tool_name(name)
        if canonical_name not in self.draft_preview_tools:
            return False
        if not raw_arguments:
            return False

        state = self._streams.get(stream_id)
        if state is None:
            state = LivePreviewState(name=canonical_name)
            self._streams[stream_id] = state
        else:
            state.name = canonical_name
        if len(raw_arguments) < state.raw_len:
            self._reset_stream_state(state)
        state.raw_len = len(raw_arguments)

        try:
            parsed_args = json.loads(raw_arguments)
        except json.JSONDecodeError:
            parsed_args = None

        if isinstance(parsed_args, dict):
            filepath = str(parsed_args.get("filepath", ""))
            content = parsed_args.get("content")
            if isinstance(content, str):
                return self._preview_single_file(state, filepath, content, write, update_preview)

        filepath, _ = self._extract_partial_json_string_field(raw_arguments, "filepath")
        content, _ = self._extract_partial_json_string_field(raw_arguments, "content")
        if content is None:
            return False
        return self._preview_single_file(state, filepath or "", content, write, update_preview)

    def close(
        self,
        stream_id: str,
        write_indented: WriteIndentedFn,
        write_code: WriteCodeFn,
        clear_preview: ClearPreviewFn,
        *,
        retain_partial: bool = False,
    ) -> bool:
        state = self._streams.get(stream_id)
        if not state or (not state.opened and not state.preview_lines and not state.line_buf):
            return False
        if not state.closed:
            self._flush_state_preview(state, write_indented, write_code, clear_preview, retain_partial=retain_partial)
            state.closed = True
        self._streams.pop(stream_id, None)
        return True

    def close_all(self, write_indented: WriteIndentedFn, write_code: WriteCodeFn, clear_preview: ClearPreviewFn) -> None:
        for stream_id in list(self._streams.keys()):
            self.close(stream_id, write_indented, write_code, clear_preview)

    def write_static_preview(
        self,
        tool_name: str,
        args: Any,
        write: WriteFn,
        write_indented: WriteIndentedFn,
        write_code: WriteCodeFn,
        *,
        workspace_root: str | Path | None = None,
    ) -> None:
        if not isinstance(args, dict):
            return
        if self.canonical_preview_tool_name(tool_name) not in self.draft_preview_tools:
            return
        content = args.get("content")
        filepath = str(args.get("filepath", ""))
        if not isinstance(content, str) or not content.strip():
            return
        if self._content_is_compacted_history(content):
            restored_content = self._read_workspace_preview(workspace_root, filepath)
            if restored_content is None or not restored_content.strip():
                label = filepath or "file"
                write(self._label_markup("file draft", label))
                write_indented(self._muted_markup("preview unavailable; compacted history no longer contains file contents"), 2)
                return
            self._write_file_preview(filepath, restored_content, write, write_indented, write_code)
            write_indented(self._muted_markup("preview restored from current workspace file"), 2)
            return

        lines, clipped_for_display = self._clipped_preview_lines(content)

        write(self._label_markup("file draft", filepath))
        write_code(lines, self._guess_language(filepath), 2)
        if clipped_for_display:
            write_indented(self._muted_markup("... (preview clipped; file write still uses full content) ..."), 2)

    def write_shell_running_preview(self, tool_name: str, args: Any, write: WriteFn) -> None:
        if self.canonical_preview_tool_name(tool_name) != "shell_command" or not isinstance(args, dict):
            return
        command = str(args.get("command") or "").strip() or "shell command"
        timeout = args.get("timeout_s")
        timeout_text = ""
        try:
            timeout_int = int(timeout) if timeout is not None else 600
        except (TypeError, ValueError):
            timeout_int = 600
        if timeout_int > 0:
            timeout_text = f", timeout {timeout_int}s"
        write(self._label_markup("shell running", f"{command} (waiting for completion{timeout_text})"))

    def _write_file_preview(
        self,
        filepath: str,
        content: str,
        write: WriteFn,
        write_indented: WriteIndentedFn,
        write_code: WriteCodeFn,
    ) -> None:
        lines, clipped_for_display = self._clipped_preview_lines(content)
        write(self._label_markup("file draft", filepath))
        write_code(lines, self._guess_language(filepath), 2)
        if clipped_for_display:
            write_indented(self._muted_markup("... (preview clipped; file write still uses full content) ..."), 2)

    def write_result_preview(
        self,
        tool_name: str,
        result: Any,
        write: WriteFn,
        write_indented: WriteIndentedFn,
        write_code: WriteCodeFn,
    ) -> None:
        canonical_name = self.canonical_preview_tool_name(tool_name)
        if canonical_name == "shell_command":
            self._write_shell_result_preview(result, write, write_indented, write_code)
            return
        if canonical_name != "edit_file":
            return
        if not isinstance(result, dict):
            return
        data = result.get("data")
        if not isinstance(data, dict):
            return
        diff_text = data.get("diff")
        filepath = str(data.get("filepath", ""))
        if not isinstance(diff_text, str) or not diff_text.strip():
            return

        lines, truncated = self._clipped_preview_lines(diff_text)
        label = filepath or "edited file"
        write(self._label_markup("edit diff", label))
        write_code(lines, "diff", 2)
        if truncated:
            write_indented(self._muted_markup("... (diff truncated) ..."), 2)

    def _write_shell_result_preview(
        self,
        result: Any,
        write: WriteFn,
        write_indented: WriteIndentedFn,
        write_code: WriteCodeFn,
    ) -> None:
        if not isinstance(result, dict):
            return
        data = result.get("data")
        if not isinstance(data, dict):
            return
        if not any(key in data for key in ("command", "stdout", "stderr", "returncode")):
            return

        command = str(data.get("command") or "").strip()
        returncode = data.get("returncode")
        duration = ""
        meta = result.get("meta")
        if isinstance(meta, dict) and isinstance(meta.get("duration_ms"), int):
            duration = f", {int(meta['duration_ms'])}ms"
        status = f"exit {returncode}" if returncode is not None else "completed"
        label = command or "shell command"
        write(self._label_markup("shell", f"{label} ({status}{duration})"))

        stdout = str(data.get("stdout") or "")
        stderr = str(data.get("stderr") or "")
        stdout_truncated = bool(data.get("stdout_truncated"))
        stderr_truncated = bool(data.get("stderr_truncated"))
        wrote_output = False
        if stdout:
            self._write_shell_output_block(
                "stdout" if stderr else "",
                stdout,
                stdout_truncated,
                write_indented,
                write_code,
            )
            wrote_output = True
        if stderr:
            self._write_shell_output_block("stderr", stderr, stderr_truncated, write_indented, write_code)
            wrote_output = True
        if not wrote_output:
            write_indented(self._muted_markup("no stdout or stderr"), 2)

    def _write_shell_output_block(
        self,
        stream_name: str,
        output: str,
        command_truncated: bool,
        write_indented: WriteIndentedFn,
        write_code: WriteCodeFn,
    ) -> None:
        lines, display_truncated = self._clipped_preview_lines(output.rstrip("\n"))
        if stream_name:
            write_indented(self._muted_markup(f"{stream_name}:"), 2)
        write_code(lines or [""], "text", 2)
        if command_truncated or display_truncated:
            write_indented(self._muted_markup(f"... ({stream_name} truncated) ..."), 2)

    def rendered_filepaths(self, stream_id: str) -> set[str]:
        state = self._streams.get(stream_id)
        if state is None:
            return set()
        rendered = set(state.rendered_filepaths)
        if state.filepath:
            rendered.add(state.filepath)
        return rendered

    def mark_rendered_filepaths(self, stream_id: str, filepaths: set[str]) -> None:
        state = self._streams.get(stream_id)
        if state is None:
            return
        state.rendered_filepaths.update(filepaths)

    def apply_final_arguments(self, stream_id: str, name: str, args: Any) -> None:
        state = self._streams.get(stream_id)
        if state is None or not isinstance(args, dict):
            return
        canonical_name = self.canonical_preview_tool_name(name)
        if canonical_name not in self.draft_preview_tools:
            return
        filepath = str(args.get("filepath", "")).strip()
        if filepath:
            state.filepath = filepath
        content = args.get("content")
        if isinstance(content, str) and content:
            lines = content.split("\n")
            if content.endswith("\n"):
                state.preview_lines = lines[:-1]
                state.line_buf = ""
            else:
                state.preview_lines = lines[:-1]
                state.line_buf = lines[-1] if lines else ""

    def _flush_state_preview(
        self,
        state: LivePreviewState,
        write_indented: WriteIndentedFn,
        write_code: WriteCodeFn,
        clear_preview: ClearPreviewFn,
        *,
        retain_partial: bool = False,
    ) -> None:
        tail = state.line_buf
        if tail:
            state.preview_lines.append(tail)
            state.line_buf = ""
        if state.preview_lines:
            if not state.opened:
                path_label = state.filepath or "(pending filepath)"
                write_indented(self._label_markup("file draft", path_label), 0)
                state.opened = True
            preview_text = "\n".join(state.preview_lines)
            lines, clipped_for_display = self._clipped_preview_lines(preview_text)
            write_code(lines, self._guess_language(state.filepath), 2)
            if clipped_for_display:
                write_indented(self._muted_markup("... (preview clipped; file write still uses full content) ..."), 2)
            if state.filepath:
                state.rendered_filepaths.add(state.filepath)
            state.flushed_item_count = max(state.flushed_item_count, state.item_index + 1)
        state.opened = False
        state.preview_lines = []
        state.line_buf = ""
        if not retain_partial:
            clear_preview()

    @staticmethod
    def _guess_language(filepath: str) -> str | None:
        return LANGUAGE_BY_SUFFIX.get(Path(filepath).suffix.lower())

    @staticmethod
    def _extract_partial_json_string_field(raw: str, key: str) -> tuple[str | None, bool]:
        value, complete, _ = LiveToolPreviewManager._extract_partial_json_string_field_from(raw, key, 0)
        return value, complete

    @staticmethod
    def _extract_partial_json_string_field_from(raw: str, key: str, start_at: int) -> tuple[str | None, bool, int]:
        marker = f'"{key}"'
        start = raw.find(marker, start_at)
        if start < 0:
            return None, False, -1
        colon = raw.find(":", start + len(marker))
        if colon < 0:
            return None, False, -1
        i = colon + 1
        while i < len(raw) and raw[i].isspace():
            i += 1
        if i >= len(raw) or raw[i] != '"':
            return None, False, -1
        i += 1
        out: list[str] = []
        escaped = False
        while i < len(raw):
            ch = raw[i]
            if escaped:
                if ch == "n":
                    out.append("\n")
                elif ch == "r":
                    out.append("\r")
                elif ch == "t":
                    out.append("\t")
                elif ch in {'"', "\\", "/"}:
                    out.append(ch)
                elif ch == "b":
                    out.append("\b")
                elif ch == "f":
                    out.append("\f")
                elif ch == "u":
                    if i + 4 < len(raw):
                        hex_part = raw[i + 1 : i + 5]
                        try:
                            out.append(chr(int(hex_part, 16)))
                            i += 4
                        except ValueError:
                            out.append("u")
                    else:
                        return "".join(out), False, i
                else:
                    out.append(ch)
                escaped = False
            else:
                if ch == "\\":
                    escaped = True
                elif ch == '"':
                    return "".join(out), True, i
                else:
                    out.append(ch)
            i += 1
        return "".join(out), False, i
