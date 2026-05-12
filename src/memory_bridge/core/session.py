"""Session store — pure in-memory conversation history management."""

import logging
from collections import deque

from .logging import structured_debug, structured_info

logger: logging.Logger = logging.getLogger(__name__)


class SessionExistsError(Exception):
    """Raised when creating a session that already exists."""


class SessionNotFoundError(Exception):
    """Raised when accessing a non-existent session."""


class SessionStore:
    """In-memory session history store.

    Sessions are keyed by (agent_id, session_id) tuples.
    Each session holds at most max_history messages.
    All data is lost on process restart.
    """

    def __init__(self, max_history: int = 50) -> None:
        self._sessions: dict[tuple[str, str], deque[dict[str, object]]] = {}
        self._max_history: int = max_history

    def create(
        self,
        agent_id: str,
        session_id: str,
        messages: list[dict[str, object]] | None = None,
    ) -> None:
        key: tuple[str, str] = (agent_id, session_id)
        if key in self._sessions:
            structured_info(
                logger,
                "session create conflict",
                agent_id=agent_id,
                session_id=session_id,
            )
            raise SessionExistsError(
                f"SESSION_EXISTS: session {session_id} already exists"
                f" for agent {agent_id}"
            )
        dq: deque[dict[str, object]] = deque[dict[str, object]](
            maxlen=self._max_history
        )
        msg_count: int = 0
        if messages:
            dq.extend(messages)
            msg_count = len(messages)
        self._sessions[key] = dq
        structured_info(
            logger,
            "session created",
            agent_id=agent_id,
            session_id=session_id,
            messages_count=msg_count,
        )

    def exists(self, agent_id: str, session_id: str) -> bool:
        result: bool = (agent_id, session_id) in self._sessions
        if not result:
            structured_debug(
                logger,
                "session check → not found",
                agent_id=agent_id,
                session_id=session_id,
            )
        return result

    def get(self, agent_id: str, session_id: str) -> list[dict[str, object]]:
        key: tuple[str, str] = (agent_id, session_id)
        if key not in self._sessions:
            structured_debug(
                logger,
                "session get → not found",
                agent_id=agent_id,
                session_id=session_id,
            )
            raise SessionNotFoundError(
                f"SESSION_NOT_FOUND: session {session_id} for agent {agent_id}"
            )
        history: list[dict[str, object]] = list(self._sessions[key])
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
        key: tuple[str, str] = (agent_id, session_id)
        if key not in self._sessions:
            self._sessions[key] = deque[dict[str, object]](maxlen=self._max_history)
        self._sessions[key].extend(messages)
        structured_debug(
            logger,
            "session appended",
            agent_id=agent_id,
            session_id=session_id,
            added=len(messages),
            total=len(self._sessions[key]),
        )

    def clear(self, agent_id: str, session_id: str) -> None:
        key: tuple[str, str] = (agent_id, session_id)
        existed: bool = key in self._sessions
        self._sessions.pop(key, None)
        if existed:
            structured_debug(
                logger,
                "session cleared",
                agent_id=agent_id,
                session_id=session_id,
            )
