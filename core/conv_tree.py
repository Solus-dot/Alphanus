from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = "1.0.0"
_COMPACTED_MARKER = "\n...[compacted]"


def _major(version: str) -> int:
    return int((version or "0").split(".", 1)[0])


@dataclass
class Turn:
    id: str
    user_content: Any
    assistant_content: Optional[str]
    parent: Optional[str]
    children: List[str]
    label: str = ""
    branch_root: bool = False
    skill_exchanges: List[dict] = field(default_factory=list)
    assistant_state: str = "pending"

    @staticmethod
    def _strip_attachment_blocks(text: str) -> str:
        """Strip inline attachment payload blocks from display text.

        Attachment text is injected as:
        [File: name]
        ```ext
        ...
        ```

        We remove those blocks for UI preview/readability while preserving
        the original `user_content` used for model context.
        """
        lines = text.splitlines()
        kept: List[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("[Attachments: ") and line.endswith("]"):
                i += 1
                while i < len(lines) and not lines[i].strip():
                    i += 1
                continue
            if line.startswith("[File: ") and line.endswith("]"):
                i += 1
                if i < len(lines) and lines[i].startswith("```"):
                    i += 1
                    while i < len(lines) and not lines[i].startswith("```"):
                        i += 1
                    if i < len(lines) and lines[i].startswith("```"):
                        i += 1
                while i < len(lines) and not lines[i].strip():
                    i += 1
                continue
            kept.append(line)
            i += 1
        return "\n".join(kept).strip()

    def user_text(self) -> str:
        if isinstance(self.user_content, list):
            chunks: List[str] = []
            for part in self.user_content:
                if part.get("type") == "text":
                    chunks.append(part.get("text", ""))
            combined = "\n".join(c for c in chunks if c).strip()
            return self._strip_attachment_blocks(combined)
        return str(self.user_content or "")

    def attachment_summary(self) -> str:
        def _extract(text: str) -> str:
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("[Attachments: ") and stripped.endswith("]"):
                    return stripped[len("[Attachments: ") : -1].strip()
            return ""

        if isinstance(self.user_content, list):
            for part in self.user_content:
                if part.get("type") == "text":
                    summary = _extract(str(part.get("text", "")))
                    if summary:
                        return summary
            return ""
        return _extract(str(self.user_content or ""))

    def short(self, max_len: int = 45) -> str:
        text = self.user_text().replace("\n", " ").strip()
        if len(text) <= max_len:
            return text
        return text[:max_len] + "…"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_content": self.user_content,
            "assistant_content": self.assistant_content,
            "parent": self.parent,
            "children": self.children,
            "label": self.label,
            "branch_root": self.branch_root,
            "skill_exchanges": self.skill_exchanges,
            "assistant_state": self.assistant_state,
        }

    @staticmethod
    def from_dict(data: dict) -> "Turn":
        assistant_content = data.get("assistant_content")
        assistant_state = str(data.get("assistant_state") or "").strip()
        if not assistant_state:
            content = str(assistant_content or "")
            if "[interrupted]" in content:
                assistant_state = "cancelled"
            elif assistant_content:
                assistant_state = "done"
            else:
                assistant_state = "pending"
        return Turn(
            id=data["id"],
            user_content=data.get("user_content", ""),
            assistant_content=assistant_content,
            parent=data.get("parent"),
            children=list(data.get("children", [])),
            label=data.get("label", ""),
            branch_root=bool(data.get("branch_root", False)),
            skill_exchanges=list(data.get("skill_exchanges", [])),
            assistant_state=assistant_state,
        )


class ConvTree:
    def __init__(
        self,
        compact_inactive_branches: bool = True,
        inactive_assistant_char_limit: int = 12000,
        inactive_tool_argument_char_limit: int = 5000,
        inactive_tool_content_char_limit: int = 8000,
    ) -> None:
        self.nodes: Dict[str, Turn] = {
            "root": Turn(
                id="root",
                user_content="",
                assistant_content="",
                parent=None,
                children=[],
            )
        }
        self.current_id = "root"
        self._pending_branch = False
        self._pending_branch_label = ""
        self._compact_inactive_branches = bool(compact_inactive_branches)
        self._inactive_assistant_char_limit = max(0, int(inactive_assistant_char_limit))
        self._inactive_tool_argument_char_limit = max(0, int(inactive_tool_argument_char_limit))
        self._inactive_tool_content_char_limit = max(0, int(inactive_tool_content_char_limit))
        self._version = 0
        self._active_path_cache_id = ""
        self._active_path_cache: List[Turn] = []
        self._history_messages_cache_key: tuple[str, int] | None = None
        self._history_messages_cache: List[dict] = []
        self._render_tree_cache: Dict[tuple[int, str, int], List[Tuple[str, str, bool]]] = {}

    def _invalidate_caches(self) -> None:
        self._active_path_cache_id = ""
        self._active_path_cache = []
        self._history_messages_cache_key = None
        self._history_messages_cache = []
        self._render_tree_cache = {}

    def _mark_changed(self) -> None:
        self._version += 1
        self._invalidate_caches()

    def set_compaction_policy(
        self,
        *,
        enabled: bool,
        inactive_assistant_char_limit: int,
        inactive_tool_argument_char_limit: int,
        inactive_tool_content_char_limit: int,
    ) -> None:
        self._compact_inactive_branches = bool(enabled)
        self._inactive_assistant_char_limit = max(0, int(inactive_assistant_char_limit))
        self._inactive_tool_argument_char_limit = max(0, int(inactive_tool_argument_char_limit))
        self._inactive_tool_content_char_limit = max(0, int(inactive_tool_content_char_limit))
        self.compact_inactive_branches()
        self._invalidate_caches()

    @property
    def current(self) -> Turn:
        return self.nodes[self.current_id]

    def arm_branch(self, label: str = "") -> None:
        self._pending_branch = True
        self._pending_branch_label = label.strip() or f"branch-{self.turn_count() + 1}"

    def clear_pending_branch(self) -> None:
        self._pending_branch = False
        self._pending_branch_label = ""

    def path_to(self, node_id: str) -> List[Turn]:
        path: List[Turn] = []
        cursor: Optional[str] = node_id
        while cursor is not None:
            node = self.nodes[cursor]
            path.append(node)
            cursor = node.parent
        path.reverse()
        return path

    @property
    def active_path(self) -> List[Turn]:
        if self._active_path_cache_id != self.current_id:
            self._active_path_cache = self.path_to(self.current_id)
            self._active_path_cache_id = self.current_id
        return list(self._active_path_cache)

    def turn_count(self) -> int:
        return max(0, len(self.active_path) - 1)

    def add_turn(self, user_content: Any) -> Turn:
        is_branch = self._pending_branch
        label = self._pending_branch_label
        self.clear_pending_branch()

        turn_id = str(uuid.uuid4())[:8]
        turn = Turn(
            id=turn_id,
            user_content=user_content,
            assistant_content=None,
            parent=self.current_id,
            children=[],
            label=label,
            branch_root=is_branch,
            assistant_state="pending",
        )
        self.nodes[turn_id] = turn
        self.current.children.append(turn_id)
        self.current_id = turn_id
        self._mark_changed()
        return turn

    def complete_turn(self, turn_id: str, reply: str) -> None:
        if turn_id in self.nodes:
            self.nodes[turn_id].assistant_content = reply
            self.nodes[turn_id].assistant_state = "done"
            self.compact_inactive_branches()
            self._mark_changed()

    def cancel_turn(self, turn_id: str, partial: str) -> None:
        if turn_id not in self.nodes:
            return
        partial = (partial or "").rstrip()
        self.nodes[turn_id].assistant_content = f"{partial}\n[interrupted]" if partial else "[interrupted]"
        self.nodes[turn_id].assistant_state = "cancelled"
        self.compact_inactive_branches()
        self._mark_changed()

    def fail_turn(self, turn_id: str, partial: str) -> None:
        if turn_id not in self.nodes:
            return
        self.nodes[turn_id].assistant_content = (partial or "").rstrip()
        self.nodes[turn_id].assistant_state = "error"
        self.compact_inactive_branches()
        self._mark_changed()

    def append_skill_exchange(self, turn_id: str, message: dict) -> None:
        if turn_id in self.nodes:
            self.nodes[turn_id].skill_exchanges.append(message)
            self._mark_changed()

    def history_messages(self) -> List[dict]:
        cache_key = (self.current_id, self._version)
        if self._history_messages_cache_key == cache_key:
            return list(self._history_messages_cache)
        messages: List[dict] = []
        for turn in self.active_path:
            if turn.id == "root":
                continue
            messages.append({"role": "user", "content": turn.user_content})
            messages.extend(turn.skill_exchanges)
            if turn.assistant_content:
                messages.append({"role": "assistant", "content": turn.assistant_content})
        self._history_messages_cache_key = cache_key
        self._history_messages_cache = list(messages)
        return messages

    def unbranch(self) -> Optional[str]:
        node = self.current
        while node.parent is not None:
            if node.branch_root:
                self.current_id = node.parent
                self.compact_inactive_branches()
                self._invalidate_caches()
                return self.current_id
            node = self.nodes[node.parent]
        return None

    def switch_child(self, idx: int) -> Optional[Turn]:
        children = self.current.children
        if idx < 0 or idx >= len(children):
            return None
        self.current_id = children[idx]
        self.compact_inactive_branches()
        self._invalidate_caches()
        return self.current

    def _status_marker(self, node: Turn) -> str:
        state = str(getattr(node, "assistant_state", "pending") or "pending")
        if state == "pending":
            return "…"
        if state == "cancelled":
            return "✖"
        if state == "error":
            return "!"
        return "✓"

    def render_tree(self, width: int = 80) -> List[Tuple[str, str, bool]]:
        cache_key = (width, self.current_id, self._version)
        cached = self._render_tree_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        active_ids = {t.id for t in self.active_path}
        rows: List[Tuple[str, str, bool]] = []

        def dot(node_id: str) -> str:
            if node_id == self.current_id:
                return "●"
            if node_id in active_ids:
                return "○"
            return "·"

        def node_line(node_id: str, depth: int) -> str:
            node = self.nodes[node_id]
            label = f" [{node.label}]" if node.label else (" [branch]" if node.branch_root else "")
            branch = " ⎇" if node.branch_root else ""
            indent = "  " * max(0, depth)
            text = node.short(max_len=max(8, width - len(indent) - 10))
            return f"{indent}{dot(node_id)}{label}{branch} {self._status_marker(node)}  {text}"

        def walk(node_id: str, depth: int) -> None:
            for child_id in self.nodes[node_id].children:
                child = self.nodes[child_id]
                child_depth = depth + (1 if child.branch_root else 0)
                rows.append((node_line(child_id, child_depth), child_id, child_id in active_ids))
                walk(child_id, child_depth)

        rows.append(("● [root]", "root", True))
        walk("root", 0)
        if len(rows) == 1:
            rows.append(("(empty)", "sub", False))
        self._render_tree_cache[cache_key] = list(rows)
        return rows

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "current_id": self.current_id,
            "pending_branch": self._pending_branch,
            "pending_branch_label": self._pending_branch_label,
            "nodes": {key: node.to_dict() for key, node in self.nodes.items()},
        }

    @staticmethod
    def from_dict(data: dict) -> "ConvTree":
        version = data.get("schema_version", "1.0.0")
        if _major(version) != _major(SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported tree schema version {version}; expected major {SCHEMA_VERSION}"
            )

        tree = ConvTree.__new__(ConvTree)
        tree.nodes = {key: Turn.from_dict(value) for key, value in data["nodes"].items()}
        tree.current_id = data["current_id"]
        tree._pending_branch = bool(data.get("pending_branch", False))
        tree._pending_branch_label = str(data.get("pending_branch_label") or "")
        tree._compact_inactive_branches = True
        tree._inactive_assistant_char_limit = 12000
        tree._inactive_tool_argument_char_limit = 5000
        tree._inactive_tool_content_char_limit = 8000
        tree._version = 0
        tree._active_path_cache_id = ""
        tree._active_path_cache = []
        tree._history_messages_cache_key = None
        tree._history_messages_cache = []
        tree._render_tree_cache = {}
        tree.compact_inactive_branches()
        return tree

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        if not isinstance(text, str):
            return text
        if limit <= 0:
            return _COMPACTED_MARKER.lstrip("\n")
        if len(text) <= limit or text.endswith(_COMPACTED_MARKER):
            return text
        keep = max(0, limit - len(_COMPACTED_MARKER))
        return text[:keep] + _COMPACTED_MARKER

    def _compact_skill_message(self, message: dict) -> dict:
        compacted = dict(message)
        role = str(compacted.get("role", ""))

        if role == "assistant":
            tool_calls = compacted.get("tool_calls")
            if isinstance(tool_calls, list):
                compacted_calls = []
                for call in tool_calls:
                    if not isinstance(call, dict):
                        compacted_calls.append(call)
                        continue
                    compacted_call = dict(call)
                    fn = compacted_call.get("function")
                    if isinstance(fn, dict):
                        compacted_fn = dict(fn)
                        args = compacted_fn.get("arguments")
                        if isinstance(args, str):
                            compacted_fn["arguments"] = self._truncate_text(
                                args,
                                self._inactive_tool_argument_char_limit,
                            )
                        compacted_call["function"] = compacted_fn
                    compacted_calls.append(compacted_call)
                compacted["tool_calls"] = compacted_calls

        if role == "tool":
            content = compacted.get("content")
            if isinstance(content, str):
                compacted["content"] = self._truncate_text(
                    content,
                    self._inactive_tool_content_char_limit,
                )

        return compacted

    def compact_inactive_branches(self) -> None:
        if not self._compact_inactive_branches:
            return
        active_ids = {turn.id for turn in self.active_path}
        for node_id, node in self.nodes.items():
            if node_id == "root" or node_id in active_ids:
                continue
            if isinstance(node.assistant_content, str):
                node.assistant_content = self._truncate_text(
                    node.assistant_content,
                    self._inactive_assistant_char_limit,
                )
            if node.skill_exchanges:
                node.skill_exchanges = [self._compact_skill_message(msg) for msg in node.skill_exchanges]

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        fd, tmp_path = tempfile.mkstemp(prefix=target.name + ".", dir=str(target.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp_path, target)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def load(path: str) -> "ConvTree":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return ConvTree.from_dict(data)
