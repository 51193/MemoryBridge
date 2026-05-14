"""Token store — SQLite-backed API token management."""

import asyncio
import logging
import secrets
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


class TokenStoreError(Exception):
    """Raised when token database is not initialized."""


class TokenRecord:
    def __init__(self, id: int, token: str, label: str, created_at: str) -> None:
        self.id: int = id
        self.token: str = token
        self.label: str = label
        self.created_at: str = created_at


class TokenStore:
    """SQLite-backed API token storage."""

    @classmethod
    def initialize(cls, db_path: str) -> None:
        """Create the database and table. Called only during --init."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn: sqlite3.Connection = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(SCHEMA)
        conn.commit()
        conn.close()
        logger.info("token database initialized: %s", db_path)

    def __init__(self, db_path: str) -> None:
        if not Path(db_path).is_file():
            raise TokenStoreError(
                f"Token database not found: {db_path}. "
                "Run: python memorybridge.pyz --init"
            )
        self._conn: sqlite3.Connection = sqlite3.connect(
            db_path, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")

    def is_initialized(self) -> bool:
        row: tuple[int] | None = self._conn.execute(
            "SELECT COUNT(*) FROM tokens"
        ).fetchone()
        return row is not None and row[0] > 0

    async def validate(self, token: str) -> bool:
        try:
            row: tuple[int] | None = await asyncio.to_thread(
                lambda: self._conn.execute(
                    "SELECT 1 FROM tokens WHERE token = ?", (token,)
                ).fetchone()
            )
            return row is not None
        except Exception:
            logger.exception("token validation failed")
            return False

    def create(self, label: str = "") -> str:
        token: str = secrets.token_hex(16)
        self._conn.execute(
            "INSERT INTO tokens (token, label) VALUES (?, ?)", (token, label)
        )
        self._conn.commit()
        logger.info("token created label=%s", label)
        return token

    def list_all(self) -> list[TokenRecord]:
        rows: list[sqlite3.Row] = self._conn.execute(
            "SELECT id, token, label, created_at FROM tokens ORDER BY id"
        ).fetchall()
        return [TokenRecord(r[0], r[1], r[2], r[3]) for r in rows]

    def delete(self, token: str) -> None:
        self._conn.execute("DELETE FROM tokens WHERE token = ?", (token,))
        self._conn.commit()

    async def close(self) -> None:
        """Close the SQLite connection."""
        await asyncio.to_thread(self._conn.close)
