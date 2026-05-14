"""Token store — API token management (pure business logic).

Takes an injected TokenDatabase for all I/O.
"""

import asyncio
import logging
import secrets
import sqlite3

from .token_database import TokenDatabase

logger: logging.Logger = logging.getLogger(__name__)


class TokenRecord:
    def __init__(self, id: int, token: str, label: str, created_at: str) -> None:
        self.id: int = id
        self.token: str = token
        self.label: str = label
        self.created_at: str = created_at


class TokenStore:
    """API token management backed by an injected TokenDatabase."""

    def __init__(self, database: TokenDatabase) -> None:
        self._db: TokenDatabase = database

    def is_initialized(self) -> bool:
        row: tuple[int] | None = self._db.execute(
            "SELECT COUNT(*) FROM tokens"
        ).fetchone()
        return row is not None and row[0] > 0

    async def validate(self, token: str) -> bool:
        try:
            row: tuple[int] | None = await asyncio.to_thread(
                lambda: self._db.execute(
                    "SELECT 1 FROM tokens WHERE token = ?", (token,)
                ).fetchone()
            )
            return row is not None
        except Exception:
            logger.exception("token validation failed")
            return False

    def create(self, label: str = "") -> str:
        token: str = secrets.token_hex(16)
        self._db.execute(
            "INSERT INTO tokens (token, label) VALUES (?, ?)", (token, label)
        )
        self._db.commit()
        logger.info("token created label=%s", label)
        return token

    def list_all(self) -> list[TokenRecord]:
        rows: list[sqlite3.Row] = self._db.execute(
            "SELECT id, token, label, created_at FROM tokens ORDER BY id"
        ).fetchall()
        return [TokenRecord(r[0], r[1], r[2], r[3]) for r in rows]

    def delete(self, token: str) -> None:
        self._db.execute("DELETE FROM tokens WHERE token = ?", (token,))
        self._db.commit()

    async def close(self) -> None:
        await self._db.close()
