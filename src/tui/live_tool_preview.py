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
        self.draft_preview_tools = {name for name in self.streamed_file_tools if name != "edit_file"}
        self.max_static_preview_chars = int(max_static_preview_chars)
        self.max_static_preview_lines = int(max_static_preview_lines)
        self._streams: dict[str, LivePreviewState] = {}

    def reset(self) -> None:
        self._streams = {}

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

        if not state.opened:
            state.opened = True
            path_label = state.filepath or "(pending filepath)"
            write(f"[dim]  · file draft: {esc(path_label)}[/dim]")

        lines = content.split("\n")
        if content.endswith("\n"):
            state.preview_lines = lines[:-1]
            state.line_buf = ""
        else:
            state.preview_lines = lines[:-1]
            state.line_buf = lines[-1] if lines else ""

        preview_lines = list(state.preview_lines[-self.max_static_preview_lines :])
        if state.line_buf:
            preview_lines.append(state.line_buf)
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

        if tool_name == "edit_file":
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

        if tool_name in self.streamed_file_tools:
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
        if name not in self.draft_preview_tools:
            return False
        if not raw_arguments:
            return False

        state = self._streams.get(stream_id)
        if state is None:
            state = LivePreviewState(name=name)
            self._streams[stream_id] = state
        else:
            state.name = name
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
        if not state or not state.opened:
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
    ) -> None:
        if not isinstance(args, dict):
            return
        if tool_name not in self.draft_preview_tools:
            return
        content = args.get("content")
        filepath = str(args.get("filepath", ""))
        if not isinstance(content, str) or not content.strip():
            return

        clipped = content[: self.max_static_preview_chars]
        lines = clipped.splitlines()
        truncated = len(content) > self.max_static_preview_chars or len(lines) > self.max_static_preview_lines
        lines = lines[: self.max_static_preview_lines]

        write(f"[dim]  · file draft: {esc(filepath)}[/dim]")
        write_code(lines, self._guess_language(filepath), 2)
        if truncated:
            write_indented("[dim]... (preview truncated) ...[/dim]", 2)

    def _write_file_preview(
        self,
        filepath: str,
        content: str,
        write: WriteFn,
        write_indented: WriteIndentedFn,
        write_code: WriteCodeFn,
    ) -> None:
        clipped = content[: self.max_static_preview_chars]
        lines = clipped.splitlines()
        truncated = len(content) > self.max_static_preview_chars or len(lines) > self.max_static_preview_lines
        lines = lines[: self.max_static_preview_lines]
        write(f"[dim]  · file draft: {esc(filepath)}[/dim]")
        write_code(lines, self._guess_language(filepath), 2)
        if truncated:
            write_indented("[dim]... (preview truncated) ...[/dim]", 2)

    def write_result_preview(
        self,
        tool_name: str,
        result: Any,
        write: WriteFn,
        write_indented: WriteIndentedFn,
        write_code: WriteCodeFn,
    ) -> None:
        if tool_name != "edit_file":
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

        clipped = diff_text[: self.max_static_preview_chars]
        lines = clipped.splitlines()
        truncated = len(diff_text) > self.max_static_preview_chars or len(lines) > self.max_static_preview_lines
        lines = lines[: self.max_static_preview_lines]
        label = filepath or "edited file"
        write(f"[dim]  · edit diff: {esc(label)}[/dim]")
        write_code(lines, "diff", 2)
        if truncated:
            write_indented("[dim]... (diff truncated) ...[/dim]", 2)

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
            write_code(state.preview_lines, self._guess_language(state.filepath), 2)
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
        suffix = Path(filepath).suffix.lower()
        if suffix == ".py":
            return "python"
        if suffix in {".c", ".h"}:
            return "c"
        if suffix in {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"}:
            return "cpp"
        if suffix in {".js", ".mjs", ".cjs"}:
            return "javascript"
        if suffix in {".ts", ".tsx"}:
            return "typescript"
        if suffix in {".json"}:
            return "json"
        if suffix in {".html", ".htm"}:
            return "html"
        if suffix in {".css"}:
            return "css"
        if suffix in {".md"}:
            return "markdown"
        if suffix in {".sh", ".bash", ".zsh"}:
            return "bash"
        return None

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
