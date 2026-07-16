from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from core.conv_tree import ConvTree

DEFAULT_SESSIONS_DIRNAME = "sessions"


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


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

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _active_id(self) -> str:
        row = self._connection.execute("SELECT value FROM app_state WHERE key='active_session_id'").fetchone()
        return str(row[0]) if row else ""

    def _activate(self, session_id: str) -> None:
        self._connection.execute(
            "INSERT INTO app_state(key, value) VALUES ('active_session_id', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
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
                id=row["id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                turn_count=row["turn_count"],
                branch_count=row["branch_count"],
                is_active=row["id"] == active_id,
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
        self._connection.executemany("INSERT INTO session_search(session_id,turn_id,kind,body,rank) VALUES (?,?,?,?,?)", docs)

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
                id=f"{row['session_id']}:{row['turn_id']}:{row['kind']}:{idx}",
                session_id=row["session_id"],
                turn_id=row["turn_id"],
                title=row["title"],
                kind=row["kind"],
                preview=row["body"][:96],
                score=row["rank"] * 100,
                updated_at=row["updated_at"],
                turn_count=row["turn_count"],
                branch_count=row["branch_count"],
                is_active=row["session_id"] == active_id,
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
                (
                    session.id,
                    session.title,
                    session.created_at,
                    session.updated_at,
                    json.dumps(session.tree.to_dict(), ensure_ascii=False),
                    json.dumps(session.loaded_skill_ids),
                    session.collaboration_mode,
                    session.context_summary,
                    turn_count,
                    branch_count,
                ),
            )
            self._index_session(session)
            if activate:
                self._activate(session.id)

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
        try:
            previous = self.load_session(session_id, activate=False)
        except (FileNotFoundError, ValueError):
            previous = None
        session = ChatSession(
            session_id,
            title.strip() or (previous.title if previous else "Untitled Session"),
            created_at or (previous.created_at if previous else _utc_now_iso()),
            _utc_now_iso(),
            tree,
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
            row["id"],
            row["title"],
            row["created_at"],
            row["updated_at"],
            ConvTree.from_dict(json.loads(row["tree_json"])),
            json.loads(row["loaded_skill_ids_json"]),
            row["collaboration_mode"],
            row["context_summary"],
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
