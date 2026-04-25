from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.conv_tree import ConvTree

MANIFEST_SCHEMA_VERSION = "1.0.0"
SESSION_SCHEMA_VERSION = "1.0.0"
DEFAULT_SESSIONS_DIRNAME = "sessions"


def _major(version: str) -> int:
    return int((version or "0").split(".", 1)[0])


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
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


@dataclass
class ChatSession:
    id: str
    title: str
    created_at: str
    updated_at: str
    tree: ConvTree
    loaded_skill_ids: List[str] = field(default_factory=list)
    collaboration_mode: str = "execute"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SESSION_SCHEMA_VERSION,
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tree": self.tree.to_dict(),
            "loaded_skill_ids": list(self.loaded_skill_ids),
            "collaboration_mode": str(self.collaboration_mode or "execute").strip().lower() or "execute",
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ChatSession":
        version = data.get("schema_version", "1.0.0")
        if _major(version) != _major(SESSION_SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported session schema version {version}; expected major {SESSION_SCHEMA_VERSION}"
            )

        return ChatSession(
            id=str(data["id"]),
            title=str(data.get("title") or "Untitled Session"),
            created_at=str(data.get("created_at") or _utc_now_iso()),
            updated_at=str(data.get("updated_at") or data.get("created_at") or _utc_now_iso()),
            tree=ConvTree.from_dict(data["tree"]),
            loaded_skill_ids=[
                str(item).strip()
                for item in (data.get("loaded_skill_ids") or [])
                if str(item).strip()
            ],
            collaboration_mode=(
                "plan"
                if str(data.get("collaboration_mode", "execute") or "execute").strip().lower() == "plan"
                else "execute"
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


class SessionStore:
    def __init__(self, workspace_root: str | Path, storage_dir: str | Path | None = None) -> None:
        self.workspace_root = Path(workspace_root)
        self.storage_dir = (
            Path(storage_dir)
            if storage_dir is not None
            else self.workspace_root / DEFAULT_SESSIONS_DIRNAME
        )
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

    def list_sessions(self) -> List[SessionSummary]:
        manifest = self._load_manifest()
        active_id = str(manifest.get("active_session_id") or "").strip()
        raw_sessions = manifest.get("sessions", {})
        if not isinstance(raw_sessions, dict):
            raw_sessions = {}

        summaries: List[SessionSummary] = []
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

    def create_session(self, title: str = "", tree: Optional[ConvTree] = None, *, activate: bool = True) -> ChatSession:
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
        loaded_skill_ids: Optional[List[str]] = None,
        collaboration_mode: Optional[str] = None,
        *,
        created_at: Optional[str] = None,
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
                str(item).strip()
                for item in (loaded_skill_ids or raw_meta.get("loaded_skill_ids") or [])
                if str(item).strip()
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
        with open(path, "r", encoding="utf-8") as handle:
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

    def _new_session_id(self, manifest: Dict[str, Any]) -> str:
        seen = set(manifest.get("sessions", {}).keys()) if isinstance(manifest.get("sessions"), dict) else set()
        while True:
            session_id = str(uuid.uuid4())[:8]
            if session_id not in seen:
                return session_id

    def _next_default_title(self, manifest: Dict[str, Any]) -> str:
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

    def _update_manifest_for_session(self, manifest: Dict[str, Any], session: ChatSession, *, activate: bool) -> None:
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

    def _load_manifest(self) -> Dict[str, Any]:
        if not self._manifest_path.exists():
            manifest = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "active_session_id": "",
                "sessions": {},
            }
            self._write_manifest(manifest)
            return manifest

        with open(self._manifest_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        version = data.get("schema_version", "1.0.0")
        if _major(version) != _major(MANIFEST_SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported manifest schema version {version}; expected major {MANIFEST_SCHEMA_VERSION}"
            )
        if not isinstance(data.get("sessions"), dict):
            data["sessions"] = {}
        if "active_session_id" not in data:
            data["active_session_id"] = ""
        return data

    def _write_manifest(self, manifest: Dict[str, Any]) -> None:
        payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "active_session_id": str(manifest.get("active_session_id") or ""),
            "sessions": manifest.get("sessions", {}),
        }
        _write_json_atomic(self._manifest_path, payload)
