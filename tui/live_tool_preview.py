from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from rich.markup import escape as esc


WriteFn = Callable[[str], None]
WriteIndentedFn = Callable[[str, int], None]
WriteCodeFn = Callable[[List[str], Optional[str], int], None]
UpdatePreviewFn = Callable[[List[str], Optional[str]], None]
ClearPreviewFn = Callable[[], None]


@dataclass(slots=True)
class LivePreviewState:
    name: str
    filepath: str = ""
    item_index: int = 0
    opened: bool = False
    closed: bool = False
    printed_len: int = 0
    line_buf: str = ""
    rendered_chars: int = 0
    rendered_lines: int = 0
    truncated: bool = False
    preview_lines: List[str] = field(default_factory=list)
    flushed_item_count: int = 0
    rendered_filepaths: Set[str] = field(default_factory=set)


class LiveToolPreviewManager:
    def __init__(
        self,
        streamed_file_tools: Optional[Set[str]] = None,
        max_live_preview_chars: int = 12000,
        max_live_preview_lines: int = 180,
        max_static_preview_chars: int = 8000,
        max_static_preview_lines: int = 140,
    ) -> None:
        self.streamed_file_tools = set(streamed_file_tools or {"create_file", "edit_file", "create_files"})
        self.draft_preview_tools = {name for name in self.streamed_file_tools if name != "edit_file"}
        self.max_live_preview_chars = int(max_live_preview_chars)
        self.max_live_preview_lines = int(max_live_preview_lines)
        self.max_static_preview_chars = int(max_static_preview_chars)
        self.max_static_preview_lines = int(max_static_preview_lines)
        self._streams: Dict[str, LivePreviewState] = {}

    def reset(self) -> None:
        self._streams = {}

    def compact_tool_args(self, tool_name: str, args: Any) -> str:
        if not isinstance(args, dict):
            return str(args)

        if tool_name == "create_files":
            files = args.get("files")
            if isinstance(files, list):
                return f"{len(files)} files"
            return "files=?"

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
        write_indented: Optional[WriteIndentedFn] = None,
        write_code: Optional[WriteCodeFn] = None,
        clear_preview: Optional[ClearPreviewFn] = None,
    ) -> None:
        if name not in self.draft_preview_tools:
            return
        if not raw_arguments:
            return

        state = self._streams.get(stream_id)
        if state is None:
            state = LivePreviewState(name=name)
            self._streams[stream_id] = state
        else:
            state.name = name

        if name == "create_files":
            entries = self._extract_partial_file_entries(raw_arguments)
            if not entries:
                return

            completed_count = max(0, len(entries) - 1)
            if write_indented and write_code:
                for idx in range(state.flushed_item_count, completed_count):
                    filepath_i, content_i, _complete_i = entries[idx]
                    self._write_file_preview(filepath_i, content_i, write, write_indented, write_code)
                    state.rendered_filepaths.add(filepath_i)
                state.flushed_item_count = max(state.flushed_item_count, completed_count)

            current_index = len(entries) - 1
            filepath, content, _content_complete = entries[current_index]
        else:
            current_index = 0
            filepath, _ = self._extract_partial_json_string_field(raw_arguments, "filepath")
            content, _ = self._extract_partial_json_string_field(raw_arguments, "content")

        if filepath and (filepath != state.filepath or current_index != state.item_index):
            if state.opened and write_indented and write_code and clear_preview:
                self._flush_state_preview(state, write_indented, write_code, clear_preview)
                state.printed_len = 0
                state.line_buf = ""
                state.preview_lines = []
                state.rendered_chars = 0
                state.rendered_lines = 0
                state.truncated = False
                state.opened = False
            state.filepath = filepath
            state.item_index = current_index

        if content is None:
            return

        if not state.opened:
            state.opened = True
            path_label = state.filepath or "(pending filepath)"
            write(f"[dim]  · file draft: {esc(path_label)}[/dim]")

        if len(content) < state.printed_len:
            state.printed_len = 0
            state.line_buf = ""
            state.preview_lines = []
            state.rendered_chars = 0
            state.rendered_lines = 0
            state.truncated = False

        delta = content[state.printed_len :]
        state.printed_len = len(content)

        if delta and not state.truncated:
            state.line_buf += delta
            while "\n" in state.line_buf:
                line, state.line_buf = state.line_buf.split("\n", 1)
                if (
                    state.rendered_lines >= self.max_live_preview_lines
                    or state.rendered_chars >= self.max_live_preview_chars
                ):
                    state.truncated = True
                    state.line_buf = ""
                    break

                remaining_chars = self.max_live_preview_chars - state.rendered_chars
                out_line = line[:remaining_chars]
                state.preview_lines.append(out_line)
                state.rendered_chars += len(out_line) + 1
                state.rendered_lines += 1

                if len(line) > remaining_chars:
                    state.truncated = True
                    state.line_buf = ""
                    break

        preview_lines = list(state.preview_lines)
        if state.line_buf and not state.truncated:
            preview_lines.append(state.line_buf)
        if preview_lines:
            update_preview(preview_lines, self._guess_language(state.filepath))

    def close(
        self,
        stream_id: str,
        write_indented: WriteIndentedFn,
        write_code: WriteCodeFn,
        clear_preview: ClearPreviewFn,
    ) -> bool:
        state = self._streams.get(stream_id)
        if not state or not state.opened:
            return False
        if not state.closed:
            self._flush_state_preview(state, write_indented, write_code, clear_preview)
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
        if tool_name == "create_files":
            files = args.get("files")
            if not isinstance(files, list):
                return
            for item in files:
                if not isinstance(item, dict):
                    continue
                filepath = str(item.get("filepath", ""))
                content = item.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue
                self._write_file_preview(filepath, content, write, write_indented, write_code)
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

    def rendered_filepaths(self, stream_id: str) -> Set[str]:
        state = self._streams.get(stream_id)
        if state is None:
            return set()
        rendered = set(state.rendered_filepaths)
        if state.filepath:
            rendered.add(state.filepath)
        return rendered

    def mark_rendered_filepaths(self, stream_id: str, filepaths: Set[str]) -> None:
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
    ) -> None:
        tail = state.line_buf
        if tail and not state.truncated:
            remaining_chars = self.max_live_preview_chars - state.rendered_chars
            if remaining_chars > 0:
                state.preview_lines.append(tail[:remaining_chars])
                state.rendered_chars += min(len(tail), remaining_chars)
            if len(tail) > remaining_chars:
                state.truncated = True
            state.line_buf = ""
        if state.preview_lines:
            write_code(state.preview_lines, self._guess_language(state.filepath), 2)
            if state.filepath:
                state.rendered_filepaths.add(state.filepath)
        if state.truncated:
            write_indented("[dim]... (live preview truncated) ...[/dim]", 2)
        clear_preview()

    @staticmethod
    def _guess_language(filepath: str) -> Optional[str]:
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
    def _extract_partial_json_string_field(raw: str, key: str) -> Tuple[Optional[str], bool]:
        value, complete, _ = LiveToolPreviewManager._extract_partial_json_string_field_from(raw, key, 0)
        return value, complete

    @staticmethod
    def _extract_partial_json_string_field_from(raw: str, key: str, start_at: int) -> Tuple[Optional[str], bool, int]:
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
        out: List[str] = []
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

    @classmethod
    def _extract_partial_file_entries(cls, raw: str) -> List[Tuple[str, str, bool]]:
        pos = 0
        entries: List[Tuple[str, str, bool]] = []
        while True:
            filepath, _complete, filepath_end = cls._extract_partial_json_string_field_from(raw, "filepath", pos)
            if filepath is None:
                break
            content, content_complete, content_end = cls._extract_partial_json_string_field_from(raw, "content", filepath_end)
            if content is None:
                break
            entries.append((filepath, content, content_complete))
            pos = max(content_end, filepath_end) + 1
        return entries
