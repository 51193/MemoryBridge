"""Session database — pure SQLite I/O, no business logic.

Owns the sqlite3.Connection and its lifecycle.
SessionStore (business logic) takes an injected SessionDatabase.
"""

import asyncio
import logging
import sqlite3
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)

SCHEMA: str = """\
CREATE TABLE sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id   TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(agent_id, session_id)
);

CREATE TABLE messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id   TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_messages_lookup ON messages(agent_id, session_id, id);
"""


class SessionDatabaseError(Exception):
    """Raised when the session database encounters an I/O error."""


class SessionDatabase:
    """Thin SQLite wrapper for sessions and messages tables."""

    @classmethod
    def initialize(cls, db_path: str) -> None:
        """Create the database file and schema. Called only during --init."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn: sqlite3.Connection = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        conn.commit()
        conn.close()
        logger.info("session database initialized: %s", db_path)

    def __init__(self, db_path: str) -> None:
        if not Path(db_path).is_file():
            raise SessionDatabaseError(
                f"Session database not found: {db_path}. "
                "Run: python memorybridge.pyz --init"
            )
        self._conn: sqlite3.Connection = sqlite3.connect(
            db_path, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._validate_schema()

    def _validate_schema(self) -> None:
        try:
            self._conn.execute(
                "SELECT id, agent_id, session_id, created_at FROM sessions LIMIT 0"
            )
            self._conn.execute(
                "SELECT id, agent_id, session_id, role, content, created_at FROM messages LIMIT 0"
            )
        except sqlite3.OperationalError as e:
            raise SessionDatabaseError(
                f"Session database schema is invalid: {e}. "
                "Re-run: python memorybridge.pyz --init"
            ) from e

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def commit(self) -> None:
        self._conn.commit()

    async def close(self) -> None:
        await asyncio.to_thread(self._conn.close)
