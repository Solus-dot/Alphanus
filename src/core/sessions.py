from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.conv_tree import ConvTree

DEFAULT_SESSIONS_DIRNAME = "sessions"


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _search_preview(text: str, tokens: list[str], *, max_len: int = 96) -> str:
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= max_len:
        return collapsed
    lower = collapsed.casefold()
    positions = [lower.find(token) for token in tokens if lower.find(token) >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - max_len // 3)
    end = min(len(collapsed), start + max_len)
    start = max(0, end - max_len)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(collapsed) else ""
    return f"{prefix}{collapsed[start:end].strip()}{suffix}"


@dataclass
class ChatSession:
    id: str
    title: str
    created_at: str
    updated_at: str
    tree: ConvTree
    loaded_skill_ids: list[str] = field(default_factory=list)
    collaboration_mode: str = "execute"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tree": self.tree.to_dict(),
            "loaded_skill_ids": list(self.loaded_skill_ids),
            "collaboration_mode": str(self.collaboration_mode or "execute").strip().lower() or "execute",
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ChatSession:
        return ChatSession(
            id=str(data["id"]),
            title=str(data.get("title") or "Untitled Session"),
            created_at=str(data.get("created_at") or _utc_now_iso()),
            updated_at=str(data.get("updated_at") or data.get("created_at") or _utc_now_iso()),
            tree=ConvTree.from_dict(data["tree"]),
            loaded_skill_ids=[str(item).strip() for item in (data.get("loaded_skill_ids") or []) if str(item).strip()],
            collaboration_mode=(
                "plan" if str(data.get("collaboration_mode", "execute") or "execute").strip().lower() == "plan" else "execute"
            ),
        )


@dataclass(frozen=True)
class SessionSummary:
    id: str
    title: str
    created_at: str
    updated_at: str
    turn_count: int
    branch_count: int
    is_active: bool = False


@dataclass(frozen=True)
class SessionSearchResult:
    id: str
    session_id: str
    turn_id: str
    title: str
    kind: str
    preview: str
    score: int
    updated_at: str
    turn_count: int
    branch_count: int
    is_active: bool = False


class SessionStore:
    def __init__(self, workspace_root: str | Path, storage_dir: str | Path | None = None) -> None:
        self.workspace_root = Path(workspace_root)
        self.storage_dir = Path(storage_dir) if storage_dir is not None else self.workspace_root / DEFAULT_SESSIONS_DIRNAME
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.storage_dir / "manifest.json"

    def bootstrap(self) -> ChatSession:
        manifest = self._load_manifest()
        active_id = str(manifest.get("active_session_id") or "").strip()
        if active_id:
            try:
                return self.load_session(active_id, activate=True)
            except (FileNotFoundError, ValueError, KeyError):
                pass

        summaries = self.list_sessions()
        for summary in summaries:
            try:
                return self.load_session(summary.id, activate=True)
            except (FileNotFoundError, ValueError, KeyError):
                continue

        return self.create_session()

    def list_sessions(self) -> list[SessionSummary]:
        manifest = self._load_manifest()
        active_id = str(manifest.get("active_session_id") or "").strip()
        raw_sessions = manifest.get("sessions", {})
        if not isinstance(raw_sessions, dict):
            raw_sessions = {}

        summaries: list[SessionSummary] = []
        for session_id, raw_meta in raw_sessions.items():
            if not isinstance(raw_meta, dict):
                continue
            if not self._session_path(str(session_id)).exists():
                continue
            summaries.append(
                SessionSummary(
                    id=str(session_id),
                    title=str(raw_meta.get("title") or "Untitled Session"),
                    created_at=str(raw_meta.get("created_at") or _utc_now_iso()),
                    updated_at=str(raw_meta.get("updated_at") or raw_meta.get("created_at") or _utc_now_iso()),
                    turn_count=max(0, int(raw_meta.get("turn_count", 0))),
                    branch_count=max(0, int(raw_meta.get("branch_count", 0))),
                    is_active=str(session_id) == active_id,
                )
            )
        summaries.sort(key=lambda item: (item.updated_at, item.created_at, item.title.casefold()), reverse=True)
        return summaries

    def search_sessions(self, query: str, *, limit: int = 80) -> list[SessionSearchResult]:
        needle = " ".join(str(query or "").casefold().split())
        if not needle:
            return []
        tokens = [part for part in needle.split() if part]
        if not tokens:
            return []
        summaries = {summary.id: summary for summary in self.list_sessions()}
        results: list[SessionSearchResult] = []
        for summary in summaries.values():
            try:
                session = self.load_session(summary.id, activate=False)
            except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
                continue
            candidates: list[tuple[str, str, str, int]] = [("title", "", session.title, 0)]
            for turn in session.tree.nodes.values():
                if turn.id == "root":
                    continue
                candidates.extend(
                    [
                        ("user", turn.id, turn.user_text(), 2),
                        ("assistant", turn.id, str(turn.assistant_content or ""), 3),
                        ("branch", turn.id, turn.label, 1),
                    ]
                )
                tool_names = []
                for exchange in turn.skill_exchanges:
                    name = str(exchange.get("name") or "").strip() if isinstance(exchange, dict) else ""
                    if name and name not in tool_names:
                        tool_names.append(name)
                if tool_names:
                    candidates.append(("tool", turn.id, ", ".join(tool_names), 4))
            for kind, turn_id, text, rank in candidates:
                normalized = " ".join(str(text or "").casefold().split())
                if not normalized or any(token not in normalized for token in tokens):
                    continue
                score = rank * 100
                if normalized.startswith(needle):
                    score -= 20
                elif needle in normalized:
                    score -= 10
                preview = _search_preview(str(text or ""), tokens)
                results.append(
                    SessionSearchResult(
                        id=f"{summary.id}:{turn_id}:{kind}:{len(results)}",
                        session_id=summary.id,
                        turn_id=turn_id,
                        title=summary.title,
                        kind=kind,
                        preview=preview,
                        score=score,
                        updated_at=summary.updated_at,
                        turn_count=summary.turn_count,
                        branch_count=summary.branch_count,
                        is_active=summary.is_active,
                    )
                )
        results.sort(key=lambda item: (item.updated_at, item.title.casefold()), reverse=True)
        results.sort(key=lambda item: item.score)
        return results[: max(1, int(limit))]

    def create_session(self, title: str = "", tree: ConvTree | None = None, *, activate: bool = True) -> ChatSession:
        manifest = self._load_manifest()
        session_id = self._new_session_id(manifest)
        now = _utc_now_iso()
        session = ChatSession(
            id=session_id,
            title=title.strip() or self._next_default_title(manifest),
            created_at=now,
            updated_at=now,
            tree=tree or ConvTree(),
            loaded_skill_ids=[],
            collaboration_mode="execute",
        )
        self._write_session(session)
        self._update_manifest_for_session(manifest, session, activate=activate)
        return session

    def save_tree(
        self,
        session_id: str,
        title: str,
        tree: ConvTree,
        loaded_skill_ids: list[str] | None = None,
        collaboration_mode: str | None = None,
        *,
        created_at: str | None = None,
        activate: bool = True,
    ) -> ChatSession:
        manifest = self._load_manifest()
        raw_meta = manifest.get("sessions", {}).get(session_id, {}) if isinstance(manifest.get("sessions"), dict) else {}
        session = ChatSession(
            id=session_id,
            title=title.strip() or self._next_default_title(manifest),
            created_at=str(created_at or raw_meta.get("created_at") or _utc_now_iso()),
            updated_at=_utc_now_iso(),
            tree=tree,
            loaded_skill_ids=[
                str(item).strip() for item in (loaded_skill_ids or raw_meta.get("loaded_skill_ids") or []) if str(item).strip()
            ],
            collaboration_mode=(
                "plan"
                if str(collaboration_mode or raw_meta.get("collaboration_mode") or "execute").strip().lower() == "plan"
                else "execute"
            ),
        )
        self._write_session(session)
        self._update_manifest_for_session(manifest, session, activate=activate)
        return session

    def load_session(self, selector: str, *, activate: bool = True) -> ChatSession:
        session_id = self.resolve_session_id(selector)
        path = self._session_path(session_id)
        with open(path, encoding="utf-8") as handle:
            session = ChatSession.from_dict(json.load(handle))
        if activate:
            manifest = self._load_manifest()
            if not isinstance(manifest.get("sessions"), dict):
                manifest["sessions"] = {}
            manifest["active_session_id"] = session.id
            raw_meta = manifest["sessions"].get(session.id, {})
            manifest["sessions"][session.id] = {
                **(raw_meta if isinstance(raw_meta, dict) else {}),
                "title": session.title,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "turn_count": max(0, len(session.tree.nodes) - 1),
                "branch_count": sum(1 for node in session.tree.nodes.values() if node.branch_root),
            }
            self._write_manifest(manifest)
        return session

    def delete_session(self, session_id: str) -> None:
        target_id = str(session_id or "").strip()
        if not target_id:
            raise ValueError("session id is required")

        manifest = self._load_manifest()
        sessions = manifest.get("sessions")
        if not isinstance(sessions, dict):
            sessions = {}
            manifest["sessions"] = sessions

        sessions.pop(target_id, None)
        if str(manifest.get("active_session_id") or "") == target_id:
            manifest["active_session_id"] = ""

        path = self._session_path(target_id)
        if path.exists():
            path.unlink()

        self._write_manifest(manifest)

    def resolve_session_id(self, selector: str) -> str:
        value = str(selector or "").strip()
        if not value:
            raise ValueError("session selector is required")

        summaries = self.list_sessions()
        if value.isdigit():
            idx = int(value) - 1
            if 0 <= idx < len(summaries):
                return summaries[idx].id

        for summary in summaries:
            if summary.id == value:
                return summary.id

        title_matches = [summary.id for summary in summaries if summary.title.casefold() == value.casefold()]
        if len(title_matches) == 1:
            return title_matches[0]
        if len(title_matches) > 1:
            raise ValueError(f"Multiple sessions match '{value}'. Use the numeric index or session id.")
        raise ValueError(f"No saved session matches '{value}'.")

    def _new_session_id(self, manifest: dict[str, Any]) -> str:
        seen = set(manifest.get("sessions", {}).keys()) if isinstance(manifest.get("sessions"), dict) else set()
        while True:
            session_id = str(uuid.uuid4())[:8]
            if session_id not in seen:
                return session_id

    def _next_default_title(self, manifest: dict[str, Any]) -> str:
        raw_sessions = manifest.get("sessions", {})
        existing = set()
        if isinstance(raw_sessions, dict):
            for raw_meta in raw_sessions.values():
                if isinstance(raw_meta, dict):
                    title = str(raw_meta.get("title") or "").strip()
                    if title:
                        existing.add(title.casefold())
        idx = 1
        while True:
            title = f"Session {idx}"
            if title.casefold() not in existing:
                return title
            idx += 1

    def _session_path(self, session_id: str) -> Path:
        return self.storage_dir / f"{session_id}.json"

    def _write_session(self, session: ChatSession) -> None:
        _write_json_atomic(self._session_path(session.id), session.to_dict())

    def _update_manifest_for_session(self, manifest: dict[str, Any], session: ChatSession, *, activate: bool) -> None:
        sessions = manifest.get("sessions")
        if not isinstance(sessions, dict):
            sessions = {}
            manifest["sessions"] = sessions
        sessions[session.id] = {
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "turn_count": max(0, len(session.tree.nodes) - 1),
            "branch_count": sum(1 for node in session.tree.nodes.values() if node.branch_root),
            "loaded_skill_ids": list(session.loaded_skill_ids),
            "collaboration_mode": str(session.collaboration_mode or "execute").strip().lower() or "execute",
        }
        if activate:
            manifest["active_session_id"] = session.id
        self._write_manifest(manifest)

    def _load_manifest(self) -> dict[str, Any]:
        if not self._manifest_path.exists():
            manifest = {
                "active_session_id": "",
                "sessions": {},
            }
            self._write_manifest(manifest)
            return manifest

        with open(self._manifest_path, encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data.get("sessions"), dict):
            data["sessions"] = {}
        if "active_session_id" not in data:
            data["active_session_id"] = ""
        return data

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        payload = {
            "active_session_id": str(manifest.get("active_session_id") or ""),
            "sessions": manifest.get("sessions", {}),
        }
        _write_json_atomic(self._manifest_path, payload)
