"""Tests for session module."""

import pytest

from memory_bridge.core.session import (
    SessionExistsError,
    SessionNotFoundError,
    SessionStore,
)


class TestSessionStore:
    def test_create_session(self) -> None:
        store: SessionStore = SessionStore()
        store.create("agent-1", "sess-1")
        assert store.exists("agent-1", "sess-1")

    def test_create_with_initial_messages(self) -> None:
        store: SessionStore = SessionStore()
        store.create(
            "agent-1",
            "sess-1",
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        )
        history: list[dict[str, object]] = store.get("agent-1", "sess-1")
        assert len(history) == 2
        assert history[0]["content"] == "hello"
        assert history[1]["content"] == "hi"

    def test_create_duplicate_raises(self) -> None:
        store: SessionStore = SessionStore()
        store.create("agent-1", "sess-1")
        with pytest.raises(SessionExistsError, match="SESSION_EXISTS"):
            store.create("agent-1", "sess-1")

    def test_exists_returns_false_for_missing(self) -> None:
        store: SessionStore = SessionStore()
        assert not store.exists("agent-1", "nonexistent")

    def test_exists_returns_true_after_create(self) -> None:
        store: SessionStore = SessionStore()
        store.create("agent-1", "sess-1")
        assert store.exists("agent-1", "sess-1")

    def test_get_raises_for_missing(self) -> None:
        store: SessionStore = SessionStore()
        with pytest.raises(SessionNotFoundError, match="SESSION_NOT_FOUND"):
            store.get("agent-1", "nonexistent")

    def test_get_returns_history(self) -> None:
        store: SessionStore = SessionStore()
        store.create("agent-1", "sess-1", messages=[{"role": "user", "content": "q"}])
        history: list[dict[str, object]] = store.get("agent-1", "sess-1")
        assert len(history) == 1
        assert history[0]["content"] == "q"

    def test_append_adds_messages(self) -> None:
        store: SessionStore = SessionStore()
        store.create("agent-1", "sess-1")
        store.append("agent-1", "sess-1", [{"role": "user", "content": "new"}])
        history: list[dict[str, object]] = store.get("agent-1", "sess-1")
        assert len(history) == 1
        assert history[0]["content"] == "new"

    def test_append_multiple(self) -> None:
        store: SessionStore = SessionStore()
        store.create("agent-1", "sess-1", messages=[{"role": "user", "content": "m1"}])
        store.append(
            "agent-1",
            "sess-1",
            [
                {"role": "assistant", "content": "m2"},
                {"role": "user", "content": "m3"},
            ],
        )
        history: list[dict[str, object]] = store.get("agent-1", "sess-1")
        assert len(history) == 3

    def test_sessions_isolated_by_agent(self) -> None:
        store: SessionStore = SessionStore()
        store.create("agent-1", "sess-1", messages=[{"role": "user", "content": "a1"}])
        store.create("agent-2", "sess-1", messages=[{"role": "user", "content": "a2"}])
        assert store.get("agent-1", "sess-1")[0]["content"] == "a1"
        assert store.get("agent-2", "sess-1")[0]["content"] == "a2"

    def test_sessions_isolated_by_session_id(self) -> None:
        store: SessionStore = SessionStore()
        store.create("agent-1", "sess-1", messages=[{"role": "user", "content": "s1"}])
        store.create("agent-1", "sess-2", messages=[{"role": "user", "content": "s2"}])
        assert store.get("agent-1", "sess-1")[0]["content"] == "s1"
        assert store.get("agent-1", "sess-2")[0]["content"] == "s2"

    def test_clear_removes_session(self) -> None:
        store: SessionStore = SessionStore()
        store.create("agent-1", "sess-1")
        store.clear("agent-1", "sess-1")
        assert not store.exists("agent-1", "sess-1")

    def test_clear_nonexistent_does_not_raise(self) -> None:
        store: SessionStore = SessionStore()
        store.clear("agent-1", "nonexistent")

    def test_max_history_enforced_on_create(self) -> None:
        store: SessionStore = SessionStore(max_history=3)
        messages: list[dict[str, object]] = [
            {"role": "user", "content": f"m{i}"} for i in range(5)
        ]
        store.create("agent-1", "sess-1", messages=messages)
        history: list[dict[str, object]] = store.get("agent-1", "sess-1")
        assert len(history) == 3
        assert history[0]["content"] == "m2"
        assert history[2]["content"] == "m4"

    def test_max_history_enforced_on_append(self) -> None:
        store: SessionStore = SessionStore(max_history=3)
        store.create("agent-1", "sess-1")
        for i in range(5):
            store.append("agent-1", "sess-1", [{"role": "user", "content": f"m{i}"}])
        history: list[dict[str, object]] = store.get("agent-1", "sess-1")
        assert len(history) == 3
        assert history[0]["content"] == "m2"
