"""Session store — pure in-memory conversation history management.

System messages are filtered out on write — only user, assistant, and tool
messages are persisted in the session deque. The system prompt is injected by
ContextBuilder at the top of each request, never stored in history.
"""

import logging
from collections import deque

from ..logfmt import structured_debug, structured_info

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
    """In-memory session history store.

    Sessions are keyed by (agent_id, session_id) tuples.
    Each session holds at most max_history messages.
    System messages are filtered on write — only user/assistant/tool are stored.
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
            filtered: list[dict[str, object]] = _filter_system(messages)
            dq.extend(filtered)
            msg_count = len(filtered)
        self._sessions[key] = dq
        structured_info(
            logger,
            "session created",
            agent_id=agent_id,
            session_id=session_id,
            messages_count=msg_count,
        )

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
        filtered: list[dict[str, object]] = _filter_system(messages)
        self._sessions[key].extend(filtered)
        structured_debug(
            logger,
            "session appended",
            agent_id=agent_id,
            session_id=session_id,
            added=len(filtered),
            total=len(self._sessions[key]),
        )
