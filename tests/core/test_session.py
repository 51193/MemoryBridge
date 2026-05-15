"""Tests for session module (persistent SQLite-backed)."""

import os
import tempfile

import pytest

from memory_bridge.core.session import (
    SessionExistsError,
    SessionNotFoundError,
    SessionStore,
    _filter_system,
)
from memory_bridge.core.session_database import SessionDatabase


@pytest.fixture
def session_db_path() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    SessionDatabase.initialize(path)
    yield path
    os.unlink(path)


@pytest.fixture
def session_db(session_db_path: str) -> SessionDatabase:
    db: SessionDatabase = SessionDatabase(session_db_path)
    yield db
    db._conn.close()


@pytest.fixture
def session_store(session_db: SessionDatabase) -> SessionStore:
    return SessionStore(db=session_db, window_size=3)


class TestFilterSystem:
    def test_filters_system_messages(self) -> None:
        messages: list[dict[str, object]] = [
            {"role": "system", "content": "you are a helper"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result: list[dict[str, object]] = _filter_system(messages)
        assert len(result) == 2
        assert all(m["role"] != "system" for m in result)

    def test_preserves_tool_messages(self) -> None:
        messages: list[dict[str, object]] = [
            {"role": "tool", "content": "result", "tool_call_id": "t1"},
        ]
        result: list[dict[str, object]] = _filter_system(messages)
        assert len(result) == 1

    def test_returns_empty_for_all_system(self) -> None:
        messages: list[dict[str, object]] = [
            {"role": "system", "content": "a"},
            {"role": "system", "content": "b"},
        ]
        result: list[dict[str, object]] = _filter_system(messages)
        assert result == []


class TestSessionStore:
    def test_create_session(self, session_store: SessionStore) -> None:
        sid: str = session_store.create("agent-1", "sess-1")
        assert sid == "sess-1"
        history: list[dict[str, object]] = session_store.get("agent-1", "sess-1")
        assert history == []

    def test_create_auto_generates_id(self, session_store: SessionStore) -> None:
        sid: str = session_store.create("agent-1")
        assert len(sid) == 12
        history: list[dict[str, object]] = session_store.get("agent-1", sid)
        assert history == []

    def test_create_returns_session_id(self, session_store: SessionStore) -> None:
        sid: str = session_store.create("agent-1", "my-session")
        assert sid == "my-session"

    def test_create_duplicate_raises(self, session_store: SessionStore) -> None:
        session_store.create("agent-1", "sess-1")
        with pytest.raises(SessionExistsError, match="SESSION_EXISTS"):
            session_store.create("agent-1", "sess-1")

    def test_get_raises_for_missing(self, session_store: SessionStore) -> None:
        with pytest.raises(SessionNotFoundError, match="SESSION_NOT_FOUND"):
            session_store.get("agent-1", "nonexistent")

    def test_get_returns_empty_for_new_session(self, session_store: SessionStore) -> None:
        session_store.create("agent-1", "sess-1")
        history: list[dict[str, object]] = session_store.get("agent-1", "sess-1")
        assert history == []

    def test_append_adds_messages(self, session_store: SessionStore) -> None:
        session_store.create("agent-1", "sess-1")
        session_store.append(
            "agent-1", "sess-1", [{"role": "user", "content": "hello"}]
        )
        history: list[dict[str, object]] = session_store.get("agent-1", "sess-1")
        assert len(history) == 1
        assert history[0]["content"] == "hello"

    def test_append_multiple(self, session_store: SessionStore) -> None:
        session_store.create("agent-1", "sess-1")
        session_store.append(
            "agent-1",
            "sess-1",
            [
                {"role": "user", "content": "m1"},
                {"role": "assistant", "content": "m2"},
                {"role": "user", "content": "m3"},
            ],
        )
        history: list[dict[str, object]] = session_store.get("agent-1", "sess-1")
        assert len(history) == 3

    def test_get_returns_chronological_order(self, session_store: SessionStore) -> None:
        session_store.create("agent-1", "sess-1")
        session_store.append(
            "agent-1", "sess-1", [{"role": "user", "content": "first"}]
        )
        session_store.append(
            "agent-1", "sess-1", [{"role": "assistant", "content": "second"}]
        )
        session_store.append(
            "agent-1", "sess-1", [{"role": "user", "content": "third"}]
        )
        history: list[dict[str, object]] = session_store.get("agent-1", "sess-1")
        assert history[0]["content"] == "first"
        assert history[1]["content"] == "second"
        assert history[2]["content"] == "third"

    def test_window_size_limits_results(self, session_store: SessionStore) -> None:
        session_store.create("agent-1", "sess-1")
        for i in range(5):
            session_store.append(
                "agent-1", "sess-1", [{"role": "user", "content": f"m{i}"}]
            )
        history: list[dict[str, object]] = session_store.get("agent-1", "sess-1")
        assert len(history) == 3  # window_size=3
        assert history[0]["content"] == "m2"
        assert history[1]["content"] == "m3"
        assert history[2]["content"] == "m4"

    def test_sessions_isolated_by_agent(self, session_store: SessionStore) -> None:
        session_store.create("agent-1", "sess-1")
        session_store.create("agent-2", "sess-1")
        session_store.append(
            "agent-1", "sess-1", [{"role": "user", "content": "from agent-1"}]
        )
        session_store.append(
            "agent-2", "sess-1", [{"role": "user", "content": "from agent-2"}]
        )
        h1: list[dict[str, object]] = session_store.get("agent-1", "sess-1")
        h2: list[dict[str, object]] = session_store.get("agent-2", "sess-1")
        assert h1[0]["content"] == "from agent-1"
        assert h2[0]["content"] == "from agent-2"

    def test_sessions_isolated_by_session_id(self, session_store: SessionStore) -> None:
        session_store.create("agent-1", "sess-1")
        session_store.create("agent-1", "sess-2")
        session_store.append(
            "agent-1", "sess-1", [{"role": "user", "content": "from sess-1"}]
        )
        session_store.append(
            "agent-1", "sess-2", [{"role": "user", "content": "from sess-2"}]
        )
        h1: list[dict[str, object]] = session_store.get("agent-1", "sess-1")
        h2: list[dict[str, object]] = session_store.get("agent-1", "sess-2")
        assert h1[0]["content"] == "from sess-1"
        assert h2[0]["content"] == "from sess-2"

    def test_append_filters_system(self, session_store: SessionStore) -> None:
        session_store.create("agent-1", "sess-1")
        session_store.append(
            "agent-1",
            "sess-1",
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "ok"},
            ],
        )
        history: list[dict[str, object]] = session_store.get("agent-1", "sess-1")
        assert len(history) == 2
        assert all(m["role"] != "system" for m in history)

    def test_append_to_nonexistent_session_raises(self, session_store: SessionStore) -> None:
        with pytest.raises(SessionNotFoundError, match="SESSION_NOT_FOUND"):
            session_store.append(
                "agent-1", "nonexistent", [{"role": "user", "content": "hi"}]
            )
