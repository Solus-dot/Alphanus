from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = "1.0.0"


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
        }

    @staticmethod
    def from_dict(data: dict) -> "Turn":
        return Turn(
            id=data["id"],
            user_content=data.get("user_content", ""),
            assistant_content=data.get("assistant_content"),
            parent=data.get("parent"),
            children=list(data.get("children", [])),
            label=data.get("label", ""),
            branch_root=bool(data.get("branch_root", False)),
            skill_exchanges=list(data.get("skill_exchanges", [])),
        )


class ConvTree:
    def __init__(self) -> None:
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

    @property
    def current(self) -> Turn:
        return self.nodes[self.current_id]

    def arm_branch(self, label: str = "") -> None:
        self._pending_branch = True
        self._pending_branch_label = label

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
        return self.path_to(self.current_id)

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
        )
        self.nodes[turn_id] = turn
        self.current.children.append(turn_id)
        self.current_id = turn_id
        return turn

    def complete_turn(self, turn_id: str, reply: str) -> None:
        if turn_id in self.nodes:
            self.nodes[turn_id].assistant_content = reply

    def cancel_turn(self, turn_id: str, partial: str) -> None:
        if turn_id not in self.nodes:
            return
        partial = (partial or "").rstrip()
        self.nodes[turn_id].assistant_content = (
            f"{partial}\n[interrupted]" if partial else "[interrupted]"
        )

    def append_skill_exchange(self, turn_id: str, message: dict) -> None:
        if turn_id in self.nodes:
            self.nodes[turn_id].skill_exchanges.append(message)

    def history_messages(self) -> List[dict]:
        messages: List[dict] = []
        for turn in self.active_path:
            if turn.id == "root":
                continue
            messages.append({"role": "user", "content": turn.user_content})
            messages.extend(turn.skill_exchanges)
            if turn.assistant_content:
                messages.append({"role": "assistant", "content": turn.assistant_content})
        return messages

    def unbranch(self) -> Optional[str]:
        node = self.current
        while node.parent is not None:
            if node.branch_root:
                self.current_id = node.parent
                return self.current_id
            node = self.nodes[node.parent]
        return None

    def switch_child(self, idx: int) -> Optional[Turn]:
        children = self.current.children
        if idx < 0 or idx >= len(children):
            return None
        self.current_id = children[idx]
        return self.current

    def _status_marker(self, node: Turn) -> str:
        content = node.assistant_content
        if content is None:
            return "…"
        if "[interrupted]" in content:
            return "✗"
        return "✓"

    def render_tree(self, width: int = 80) -> List[Tuple[str, str, bool]]:
        active_ids = {t.id for t in self.active_path}
        rows: List[Tuple[str, str, bool]] = []

        def dot(node_id: str) -> str:
            if node_id == self.current_id:
                return "●"
            if node_id in active_ids:
                return "○"
            return "·"

        def node_line(node_id: str, prefix: str) -> str:
            node = self.nodes[node_id]
            label = f" [{node.label}]" if node.label else (" [branch]" if node.branch_root else "")
            branch = " ⎇" if node.branch_root else ""
            text = node.short(max_len=max(8, width - len(prefix) - 12))
            return f"{prefix}{dot(node_id)}{label}{branch} {self._status_marker(node)}  {text}"

        def walk(node_id: str, prefix: str) -> None:
            children = self.nodes[node_id].children
            for idx, child_id in enumerate(children):
                last = idx == len(children) - 1
                connector = "└─" if last else "├─"
                if node_id == "root":
                    row = node_line(child_id, "")
                else:
                    row = f"{prefix}{connector} {node_line(child_id, '')}"
                rows.append((row, child_id, child_id in active_ids))
                walk(child_id, prefix + ("   " if last else "│  "))

        rows.append(("● [root]", "root", True))
        walk("root", "")
        if len(rows) == 1:
            rows.append(("(empty)", "sub", False))
        return rows

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "current_id": self.current_id,
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
        tree._pending_branch = False
        tree._pending_branch_label = ""
        return tree

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
