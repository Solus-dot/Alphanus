from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import cast

from core.message_types import ChatMessage, JSONValue, MessageContentPart, ToolCallDelta, ToolFunctionCall

SCHEMA_VERSION = "1.0.0"
_COMPACTED_MARKER = "\n...[compacted]"


def _major(version: str) -> int:
    return int((version or "0").split(".", 1)[0])


@dataclass
class Turn:
    id: str
    user_content: JSONValue | list[MessageContentPart]
    assistant_content: str | None
    parent: str | None
    children: list[str]
    label: str = ""
    branch_root: bool = False
    skill_exchanges: list[ChatMessage] = field(default_factory=list)
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
        kept: list[str] = []
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
            chunks: list[str] = []
            for part in self.user_content:
                if not isinstance(part, dict):
                    continue
                if str(part.get("type", "")).strip() == "text":
                    chunks.append(str(part.get("text", "")))
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
                if not isinstance(part, dict):
                    continue
                if str(part.get("type", "")).strip() == "text":
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

    def to_dict(self) -> dict[str, object]:
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
    def from_dict(data: dict[str, object]) -> Turn:
        assistant_content_raw = data.get("assistant_content")
        assistant_content = str(assistant_content_raw) if assistant_content_raw is not None else None
        assistant_state = str(data.get("assistant_state") or "").strip()
        if not assistant_state:
            content = str(assistant_content or "")
            if "[interrupted]" in content:
                assistant_state = "cancelled"
            elif assistant_content:
                assistant_state = "done"
            else:
                assistant_state = "pending"
        user_content_raw = data.get("user_content", "")
        if isinstance(user_content_raw, list):
            user_content = cast(list[MessageContentPart], user_content_raw)
        elif isinstance(user_content_raw, (str, int, float, bool, dict)) or user_content_raw is None:
            user_content = cast(JSONValue, user_content_raw)
        else:
            user_content = ""
        parent_raw = data.get("parent")
        parent = str(parent_raw) if isinstance(parent_raw, str) else None
        children_raw = data.get("children", [])
        children = [str(item) for item in children_raw] if isinstance(children_raw, list) else []
        exchanges_raw = data.get("skill_exchanges", [])
        exchanges: list[ChatMessage] = []
        if isinstance(exchanges_raw, list):
            for item in exchanges_raw:
                if isinstance(item, dict):
                    exchanges.append(cast(ChatMessage, item))
        return Turn(
            id=str(data.get("id", "")),
            user_content=user_content,
            assistant_content=assistant_content,
            parent=parent,
            children=children,
            label=str(data.get("label", "")),
            branch_root=bool(data.get("branch_root", False)),
            skill_exchanges=exchanges,
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
        self.nodes: dict[str, Turn] = {
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
        self._history_version = 0
        self._active_path_cache_id = ""
        self._active_path_cache: list[Turn] = []
        self._history_messages_cache_key: tuple[str, int] | None = None
        self._history_messages_cache: list[ChatMessage] = []

    def _invalidate_active_path_cache(self) -> None:
        self._active_path_cache_id = ""
        self._active_path_cache = []

    def _invalidate_history_cache(self) -> None:
        self._history_messages_cache_key = None
        self._history_messages_cache = []

    def _mark_history_changed(self) -> None:
        self._history_version += 1
        self._invalidate_history_cache()

    def _mark_path_changed(self) -> None:
        self._invalidate_active_path_cache()

    def _mark_structure_changed(self) -> None:
        self._mark_path_changed()
        self._mark_history_changed()

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
        self._invalidate_active_path_cache()
        self._invalidate_history_cache()

    @property
    def current(self) -> Turn:
        return self.nodes[self.current_id]

    def arm_branch(self, label: str = "") -> None:
        self._pending_branch = True
        self._pending_branch_label = label.strip() or f"branch-{self.turn_count() + 1}"

    def clear_pending_branch(self) -> None:
        self._pending_branch = False
        self._pending_branch_label = ""

    def path_to(self, node_id: str) -> list[Turn]:
        path: list[Turn] = []
        cursor: str | None = node_id
        while cursor is not None:
            node = self.nodes[cursor]
            path.append(node)
            cursor = node.parent
        path.reverse()
        return path

    @property
    def active_path(self) -> list[Turn]:
        if self._active_path_cache_id != self.current_id:
            self._active_path_cache = self.path_to(self.current_id)
            self._active_path_cache_id = self.current_id
        return list(self._active_path_cache)

    def turn_count(self) -> int:
        return max(0, len(self.active_path) - 1)

    def add_turn(self, user_content: JSONValue | list[MessageContentPart]) -> Turn:
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
        self._mark_structure_changed()
        return turn

    def complete_turn(self, turn_id: str, reply: str) -> None:
        if turn_id in self.nodes:
            self.nodes[turn_id].assistant_content = reply
            self.nodes[turn_id].assistant_state = "done"
            self.compact_inactive_branches()
            self._mark_history_changed()

    def cancel_turn(self, turn_id: str, partial: str) -> None:
        if turn_id not in self.nodes:
            return
        partial = (partial or "").rstrip()
        self.nodes[turn_id].assistant_content = f"{partial}\n[interrupted]" if partial else "[interrupted]"
        self.nodes[turn_id].assistant_state = "cancelled"
        self.compact_inactive_branches()
        self._mark_history_changed()

    def fail_turn(self, turn_id: str, partial: str) -> None:
        if turn_id not in self.nodes:
            return
        self.nodes[turn_id].assistant_content = (partial or "").rstrip()
        self.nodes[turn_id].assistant_state = "error"
        self.compact_inactive_branches()
        self._mark_history_changed()

    def append_skill_exchange(self, turn_id: str, message: ChatMessage) -> None:
        if turn_id in self.nodes:
            self.nodes[turn_id].skill_exchanges.append(message)
            self._mark_history_changed()

    def history_messages(self) -> list[ChatMessage]:
        cache_key = (self.current_id, self._history_version)
        if self._history_messages_cache_key == cache_key:
            return list(self._history_messages_cache)
        messages: list[ChatMessage] = []
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

    def unbranch(self) -> str | None:
        node = self.current
        while node.parent is not None:
            if node.branch_root:
                self.current_id = node.parent
                self.compact_inactive_branches()
                self._mark_structure_changed()
                return self.current_id
            node = self.nodes[node.parent]
        return None

    def switch_child(self, idx: int) -> Turn | None:
        children = self.current.children
        if idx < 0 or idx >= len(children):
            return None
        self.current_id = children[idx]
        self.compact_inactive_branches()
        self._mark_structure_changed()
        return self.current

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "current_id": self.current_id,
            "pending_branch": self._pending_branch,
            "pending_branch_label": self._pending_branch_label,
            "nodes": {key: node.to_dict() for key, node in self.nodes.items()},
        }

    @staticmethod
    def from_dict(data: dict[str, object]) -> ConvTree:
        version = str(data.get("schema_version", "1.0.0"))
        if _major(version) != _major(SCHEMA_VERSION):
            raise ValueError(f"Unsupported tree schema version {version}; expected major {SCHEMA_VERSION}")

        tree = ConvTree.__new__(ConvTree)
        nodes_raw = data.get("nodes", {})
        parsed_nodes: dict[str, Turn] = {}
        if isinstance(nodes_raw, dict):
            for key, value in nodes_raw.items():
                if isinstance(value, dict):
                    parsed_nodes[str(key)] = Turn.from_dict(cast(dict[str, object], value))
        tree.nodes = parsed_nodes or {
            "root": Turn(id="root", user_content="", assistant_content="", parent=None, children=[]),
        }
        current_id = str(data.get("current_id", "root"))
        tree.current_id = current_id if current_id in tree.nodes else "root"
        tree._pending_branch = bool(data.get("pending_branch", False))
        tree._pending_branch_label = str(data.get("pending_branch_label") or "")
        tree._compact_inactive_branches = True
        tree._inactive_assistant_char_limit = 12000
        tree._inactive_tool_argument_char_limit = 5000
        tree._inactive_tool_content_char_limit = 8000
        tree._history_version = 0
        tree._active_path_cache_id = ""
        tree._active_path_cache = []
        tree._history_messages_cache_key = None
        tree._history_messages_cache = []
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

    def _compact_skill_message(self, message: ChatMessage) -> ChatMessage:
        compacted = cast(ChatMessage, dict(message))
        role = str(compacted.get("role", ""))

        if role == "assistant":
            tool_calls = compacted.get("tool_calls")
            if isinstance(tool_calls, list):
                compacted_calls: list[ToolCallDelta] = []
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    compacted_call = cast(ToolCallDelta, dict(call))
                    fn = compacted_call.get("function")
                    if isinstance(fn, dict):
                        compacted_fn = dict(fn)
                        args = compacted_fn.get("arguments")
                        if isinstance(args, str):
                            compacted_fn["arguments"] = self._truncate_text(
                                args,
                                self._inactive_tool_argument_char_limit,
                            )
                        compacted_call["function"] = cast(ToolFunctionCall, compacted_fn)
                    compacted_calls.append(compacted_call)
                compacted["tool_calls"] = compacted_calls

        if role == "tool":
            content = compacted.get("content")
            if isinstance(content, str):
                compacted["content"] = self._truncate_text(
                    content,
                    self._inactive_tool_content_char_limit,
                )

        return cast(ChatMessage, compacted)

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
