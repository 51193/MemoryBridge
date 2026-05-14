"""Token database — pure SQLite I/O, no business logic.

Owns the sqlite3.Connection and its lifecycle.
TokenStore (business logic) takes an injected TokenDatabase.
"""

import asyncio
import logging
import sqlite3
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)

SCHEMA: str = """\
CREATE TABLE tokens (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    token      TEXT NOT NULL UNIQUE,
    label      TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
);
"""


class TokenDatabaseError(Exception):
    """Raised when the token database encounters an I/O error."""


class TokenDatabase:
    """Thin SQLite wrapper for the tokens table.

    Owns the sqlite3.Connection. No business logic — just execute/commit.
    """

    @classmethod
    def initialize(cls, db_path: str) -> None:
        """Create the database file and schema. Called only during --init."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn: sqlite3.Connection = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(SCHEMA)
        conn.commit()
        conn.close()
        logger.info("token database initialized: %s", db_path)

    def __init__(self, db_path: str) -> None:
        if not Path(db_path).is_file():
            raise TokenDatabaseError(
                f"Token database not found: {db_path}. "
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
                "SELECT id, token, label, created_at FROM tokens LIMIT 0"
            )
        except sqlite3.OperationalError as e:
            raise TokenDatabaseError(
                f"Token database schema is invalid: {e}. "
                "Re-run: python memorybridge.pyz --init"
            ) from e

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def commit(self) -> None:
        self._conn.commit()

    async def close(self) -> None:
        await asyncio.to_thread(self._conn.close)
