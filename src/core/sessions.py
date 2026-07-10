from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
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


@dataclass
class ChatSession:
    id: str
    title: str
    created_at: str
    updated_at: str
    tree: ConvTree
    loaded_skill_ids: list[str] = field(default_factory=list)
    collaboration_mode: str = "execute"
    context_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tree": self.tree.to_dict(),
            "loaded_skill_ids": list(self.loaded_skill_ids),
            "collaboration_mode": str(self.collaboration_mode or "execute").strip().lower() or "execute",
            "context_summary": str(self.context_summary or ""),
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
            context_summary=str(data.get("context_summary") or ""),
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


class LegacySessionStore:
    def __init__(self, project_root: str | Path, storage_dir: str | Path | None = None) -> None:
        self.project_root = Path(project_root)
        self.storage_dir = Path(storage_dir) if storage_dir is not None else self.project_root / DEFAULT_SESSIONS_DIRNAME
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
        results: list[SessionSearchResult] = []
        for summary in self.list_sessions():
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
                        ("branch", turn.id, turn.label, 1),
                    ]
                )
                if turn.assistant_state != "error":
                    candidates.append(("assistant", turn.id, str(turn.assistant_content or ""), 3))
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
                collapsed = " ".join(str(text or "").split())
                if len(collapsed) <= 96:
                    preview = collapsed
                else:
                    lower_text = collapsed.casefold()
                    positions = [lower_text.find(token) for token in tokens if lower_text.find(token) >= 0]
                    center = min(positions) if positions else 0
                    start = max(0, center - 32)
                    end = min(len(collapsed), start + 96)
                    start = max(0, end - 96)
                    prefix = "…" if start > 0 else ""
                    suffix = "…" if end < len(collapsed) else ""
                    preview = f"{prefix}{collapsed[start:end].strip()}{suffix}"
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
        seen_session_ids = set(manifest.get("sessions", {}).keys()) if isinstance(manifest.get("sessions"), dict) else set()
        while True:
            session_id = str(uuid.uuid4())[:8]
            if session_id not in seen_session_ids:
                break
        now = _utc_now_iso()
        session = ChatSession(
            id=session_id,
            title=title.strip() or self._next_default_title(manifest),
            created_at=now,
            updated_at=now,
            tree=tree or ConvTree(),
            loaded_skill_ids=[],
            collaboration_mode="execute",
            context_summary="",
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
        context_summary: str | None = None,
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
            context_summary=str(context_summary if context_summary is not None else raw_meta.get("context_summary") or ""),
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
            "context_summary": str(session.context_summary or ""),
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


class SQLiteSessionStore:
    """Transactional v1 session store with indexed, bounded search."""

    SCHEMA_VERSION = 1

    def __init__(self, project_root: str | Path, storage_dir: str | Path | None = None) -> None:
        self.project_root = Path(project_root)
        self.storage_dir = Path(storage_dir) if storage_dir is not None else self.project_root / DEFAULT_SESSIONS_DIRNAME
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        legacy = [self.storage_dir / "manifest.json", *self.storage_dir.glob("*.json")]
        if any(path.exists() for path in legacy):
            raise ValueError(
                f"Legacy unversioned sessions were found in {self.storage_dir}. "
                "Alphanus v1 does not migrate them; export or remove that directory before continuing."
            )
        self.database_path = self.storage_dir / "sessions.db"
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.database_path, timeout=5.0, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._migrate()

    def _migrate(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    tree_json TEXT NOT NULL,
                    loaded_skill_ids_json TEXT NOT NULL DEFAULT '[]',
                    collaboration_mode TEXT NOT NULL DEFAULT 'execute',
                    context_summary TEXT NOT NULL DEFAULT '',
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    branch_count INTEGER NOT NULL DEFAULT 0,
                    deleted_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(deleted_at, updated_at DESC);
                CREATE TABLE IF NOT EXISTS session_search (
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    turn_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    body TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    PRIMARY KEY(session_id, turn_id, kind)
                );
                CREATE INDEX IF NOT EXISTS idx_session_search_body ON session_search(body);
                CREATE TABLE IF NOT EXISTS app_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                """
            )
            self._connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (self.SCHEMA_VERSION, _utc_now_iso()),
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteSessionStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _active_id(self) -> str:
        row = self._connection.execute("SELECT value FROM app_state WHERE key='active_session_id'").fetchone()
        return str(row[0]) if row else ""

    def _activate(self, session_id: str) -> None:
        self._connection.execute(
            "INSERT INTO app_state(key, value) VALUES ('active_session_id', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (session_id,),
        )

    @staticmethod
    def _counts(tree: ConvTree) -> tuple[int, int]:
        return max(0, len(tree.nodes) - 1), sum(1 for node in tree.nodes.values() if node.branch_root)

    def bootstrap(self) -> ChatSession:
        active_id = self._active_id()
        if active_id:
            try:
                return self.load_session(active_id)
            except (FileNotFoundError, ValueError):
                pass
        sessions = self.list_sessions()
        return self.load_session(sessions[0].id) if sessions else self.create_session()

    def list_sessions(self, *, limit: int = 1000, offset: int = 0) -> list[SessionSummary]:
        active_id = self._active_id()
        rows = self._connection.execute(
            "SELECT id,title,created_at,updated_at,turn_count,branch_count FROM sessions "
            "WHERE deleted_at IS NULL ORDER BY updated_at DESC, created_at DESC LIMIT ? OFFSET ?",
            (max(1, min(int(limit), 1000)), max(0, int(offset))),
        ).fetchall()
        return [
            SessionSummary(
                id=row["id"], title=row["title"], created_at=row["created_at"], updated_at=row["updated_at"],
                turn_count=row["turn_count"], branch_count=row["branch_count"], is_active=row["id"] == active_id,
            )
            for row in rows
        ]

    def _index_session(self, session: ChatSession) -> None:
        self._connection.execute("DELETE FROM session_search WHERE session_id=?", (session.id,))
        docs: list[tuple[str, str, str, str, int]] = [(session.id, "", "title", session.title.casefold(), 0)]
        for turn in session.tree.nodes.values():
            if turn.id == "root":
                continue
            docs.extend(
                [
                    (session.id, turn.id, "user", turn.user_text().casefold(), 2),
                    (session.id, turn.id, "branch", str(turn.label).casefold(), 1),
                ]
            )
            if turn.assistant_state != "error":
                docs.append((session.id, turn.id, "assistant", str(turn.assistant_content or "").casefold(), 3))
            tool_names = {
                str(exchange.get("name") or "").strip()
                for exchange in turn.skill_exchanges
                if isinstance(exchange, dict) and str(exchange.get("name") or "").strip()
            }
            if tool_names:
                docs.append((session.id, turn.id, "tool", " ".join(sorted(tool_names)).casefold(), 4))
        self._connection.executemany(
            "INSERT INTO session_search(session_id,turn_id,kind,body,rank) VALUES (?,?,?,?,?)", docs
        )

    def search_sessions(self, query: str, *, limit: int = 80) -> list[SessionSearchResult]:
        needle = " ".join(str(query or "").casefold().split())
        if not needle:
            return []
        escaped_tokens = [token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") for token in needle.split()]
        pattern = "%" + "%".join(escaped_tokens) + "%"
        rows = self._connection.execute(
            "SELECT d.session_id,d.turn_id,d.kind,d.body,d.rank,s.title,s.updated_at,s.turn_count,s.branch_count "
            "FROM session_search d JOIN sessions s ON s.id=d.session_id "
            "WHERE s.deleted_at IS NULL AND d.body LIKE ? ESCAPE '\\' "
            "ORDER BY d.rank ASC,s.updated_at DESC LIMIT ?",
            (pattern, max(1, min(int(limit), 200))),
        ).fetchall()
        active_id = self._active_id()
        return [
            SessionSearchResult(
                id=f"{row['session_id']}:{row['turn_id']}:{row['kind']}:{idx}", session_id=row["session_id"],
                turn_id=row["turn_id"], title=row["title"], kind=row["kind"], preview=row["body"][:96],
                score=row["rank"] * 100, updated_at=row["updated_at"], turn_count=row["turn_count"],
                branch_count=row["branch_count"], is_active=row["session_id"] == active_id,
            )
            for idx, row in enumerate(rows)
        ]

    def create_session(self, title: str = "", tree: ConvTree | None = None, *, activate: bool = True) -> ChatSession:
        existing = {summary.id for summary in self.list_sessions()}
        session_id = str(uuid.uuid4())[:8]
        while session_id in existing:
            session_id = str(uuid.uuid4())[:8]
        now = _utc_now_iso()
        session = ChatSession(session_id, title.strip() or f"Session {len(existing) + 1}", now, now, tree or ConvTree())
        self._save(session, activate=activate)
        return session

    def _save(self, session: ChatSession, *, activate: bool) -> None:
        turn_count, branch_count = self._counts(session.tree)
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO sessions(id,title,created_at,updated_at,tree_json,loaded_skill_ids_json,collaboration_mode,context_summary,turn_count,branch_count,deleted_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,NULL) ON CONFLICT(id) DO UPDATE SET title=excluded.title,updated_at=excluded.updated_at,"
                "tree_json=excluded.tree_json,loaded_skill_ids_json=excluded.loaded_skill_ids_json,collaboration_mode=excluded.collaboration_mode,"
                "context_summary=excluded.context_summary,turn_count=excluded.turn_count,branch_count=excluded.branch_count,deleted_at=NULL",
                (session.id, session.title, session.created_at, session.updated_at, json.dumps(session.tree.to_dict(), ensure_ascii=False),
                 json.dumps(session.loaded_skill_ids), session.collaboration_mode, session.context_summary, turn_count, branch_count),
            )
            self._index_session(session)
            if activate:
                self._activate(session.id)

    def save_tree(self, session_id: str, title: str, tree: ConvTree, loaded_skill_ids: list[str] | None = None,
                  collaboration_mode: str | None = None, context_summary: str | None = None, *,
                  created_at: str | None = None, activate: bool = True) -> ChatSession:
        try:
            previous = self.load_session(session_id, activate=False)
        except (FileNotFoundError, ValueError):
            previous = None
        session = ChatSession(
            session_id, title.strip() or (previous.title if previous else "Untitled Session"),
            created_at or (previous.created_at if previous else _utc_now_iso()), _utc_now_iso(), tree,
            list(loaded_skill_ids if loaded_skill_ids is not None else (previous.loaded_skill_ids if previous else [])),
            "plan" if str(collaboration_mode or (previous.collaboration_mode if previous else "execute")).lower() == "plan" else "execute",
            context_summary if context_summary is not None else (previous.context_summary if previous else ""),
        )
        self._save(session, activate=activate)
        return session

    def load_session(self, selector: str, *, activate: bool = True) -> ChatSession:
        session_id = self.resolve_session_id(selector)
        row = self._connection.execute("SELECT * FROM sessions WHERE id=? AND deleted_at IS NULL", (session_id,)).fetchone()
        if row is None:
            raise FileNotFoundError(session_id)
        session = ChatSession(
            row["id"], row["title"], row["created_at"], row["updated_at"], ConvTree.from_dict(json.loads(row["tree_json"])),
            json.loads(row["loaded_skill_ids_json"]), row["collaboration_mode"], row["context_summary"],
        )
        if activate:
            with self._connection:
                self._activate(session.id)
        return session

    def delete_session(self, session_id: str) -> None:
        now = _utc_now_iso()
        with self._connection:
            cursor = self._connection.execute("UPDATE sessions SET deleted_at=? WHERE id=? AND deleted_at IS NULL", (now, session_id))
            if cursor.rowcount == 0:
                raise FileNotFoundError(session_id)
            if self._active_id() == session_id:
                self._activate("")

    def compact(self, *, deleted_before: str) -> int:
        with self._connection:
            cursor = self._connection.execute("DELETE FROM sessions WHERE deleted_at IS NOT NULL AND deleted_at < ?", (deleted_before,))
        self._connection.execute("PRAGMA wal_checkpoint(PASSIVE)")
        return max(0, cursor.rowcount)

    def resolve_session_id(self, selector: str) -> str:
        value = str(selector or "").strip()
        if not value:
            raise ValueError("session selector is required")
        if value.isdigit() and int(value) > 0:
            row = self._connection.execute(
                "SELECT id FROM sessions WHERE deleted_at IS NULL ORDER BY updated_at DESC, created_at DESC LIMIT 1 OFFSET ?",
                (int(value) - 1,),
            ).fetchone()
            if row is not None:
                return str(row["id"])
        row = self._connection.execute("SELECT id FROM sessions WHERE id=? AND deleted_at IS NULL", (value,)).fetchone()
        if row is not None:
            return str(row["id"])
        title_rows = self._connection.execute(
            "SELECT id FROM sessions WHERE deleted_at IS NULL AND title=? COLLATE NOCASE LIMIT 2", (value,)
        ).fetchall()
        title_ids = [str(row["id"]) for row in title_rows]
        if len(title_ids) == 1:
            return title_ids[0]
        if len(title_ids) > 1:
            raise ValueError(f"Multiple sessions match '{value}'. Use the numeric index or session id.")
        raise ValueError(f"No saved session matches '{value}'.")


# V1 runtime alias. The legacy class remains above solely to make the reset
# boundary explicit and can be removed after the v1 release branch stabilizes.
SessionStore = SQLiteSessionStore
