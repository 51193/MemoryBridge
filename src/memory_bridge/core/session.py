"""Session store — persistent conversation history via SQLite.

Each message is stored as a single row in the messages table.
System messages are filtered out on write — only user, assistant, and tool
messages are persisted. On read, the most recent N messages (window_size)
are retrieved and ordered chronologically.
"""

import logging
import sqlite3
import uuid

from ..logfmt import structured_debug, structured_info
from .session_database import SessionDatabase

logger: logging.Logger = logging.getLogger(__name__)

_SESSION_ROLES: frozenset[str] = frozenset({"user", "assistant", "tool"})


def _filter_system(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return messages with system-role entries removed."""
    return [m for m in messages if m.get("role") in _SESSION_ROLES]


class SessionExistsError(Exception):
    """Raised when creating a session that already exists."""


class SessionNotFoundError(Exception):
    """Raised when accessing a non-existent session."""


class SessionStore:
    """Persistent session history store backed by SQLite.

    Sessions are keyed by (agent_id, session_id) tuples.
    Each read returns at most window_size messages from the tail of the history.
    System messages are filtered on write — only user/assistant/tool are stored.
    """

    def __init__(self, db: SessionDatabase, window_size: int = 50) -> None:
        self._db: SessionDatabase = db
        self._window_size: int = window_size

    def create(self, agent_id: str, session_id: str | None = None) -> str:
        sid: str = session_id or uuid.uuid4().hex[:12]
        try:
            self._db.execute(
                "INSERT INTO sessions (agent_id, session_id) VALUES (?, ?)",
                (agent_id, sid),
            )
            self._db.commit()
        except sqlite3.IntegrityError:
            raise SessionExistsError(
                f"SESSION_EXISTS: session {sid} already exists for agent {agent_id}"
            )
        structured_info(
            logger,
            "session created",
            agent_id=agent_id,
            session_id=sid,
        )
        return sid

    def get(self, agent_id: str, session_id: str) -> list[dict[str, object]]:
        cursor = self._db.execute(
            "SELECT 1 FROM sessions WHERE agent_id = ? AND session_id = ?",
            (agent_id, session_id),
        )
        if not cursor.fetchone():
            structured_debug(
                logger,
                "session get → not found",
                agent_id=agent_id,
                session_id=session_id,
            )
            raise SessionNotFoundError(
                f"SESSION_NOT_FOUND: session {session_id} for agent {agent_id}"
            )
        cursor = self._db.execute(
            "SELECT role, content FROM messages "
            "WHERE agent_id = ? AND session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (agent_id, session_id, self._window_size),
        )
        rows: list[tuple[str, str]] = cursor.fetchall()
        history: list[dict[str, object]] = [
            {"role": role, "content": content}
            for role, content in reversed(rows)
        ]
        structured_debug(
            logger,
            "session history retrieved",
            agent_id=agent_id,
            session_id=session_id,
            messages_count=len(history),
        )
        return history

    def append(
        self,
        agent_id: str,
        session_id: str,
        messages: list[dict[str, object]],
    ) -> None:
        cursor = self._db.execute(
            "SELECT 1 FROM sessions WHERE agent_id = ? AND session_id = ?",
            (agent_id, session_id),
        )
        if not cursor.fetchone():
            raise SessionNotFoundError(
                f"SESSION_NOT_FOUND: session {session_id} for agent {agent_id}"
            )
        filtered: list[dict[str, object]] = _filter_system(messages)
        for msg in filtered:
            self._db.execute(
                "INSERT INTO messages (agent_id, session_id, role, content) "
                "VALUES (?, ?, ?, ?)",
                (
                    agent_id,
                    session_id,
                    str(msg["role"]),
                    str(msg["content"]),
                ),
            )
        self._db.commit()
        structured_debug(
            logger,
            "session appended",
            agent_id=agent_id,
            session_id=session_id,
            added=len(filtered),
        )
