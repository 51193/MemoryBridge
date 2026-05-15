"""Tests for session_database module — SQLite wrapper lifecycle."""

import os
import sqlite3
import tempfile

import pytest

from memory_bridge.core.session_database import SessionDatabase, SessionDatabaseError


class TestSessionDatabase:
    def test_initialize_creates_file_and_tables(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            SessionDatabase.initialize(path)
            db: SessionDatabase = SessionDatabase(path)

            db.execute(
                "INSERT INTO sessions (agent_id, session_id) VALUES (?, ?)",
                ("agent-1", "sess-1"),
            )
            db.commit()
            db.execute(
                "INSERT INTO messages (agent_id, session_id, role, content) "
                "VALUES (?, ?, ?, ?)",
                ("agent-1", "sess-1", "user", "hello"),
            )
            db.commit()

            cursor = db.execute(
                "SELECT role, content FROM messages "
                "WHERE agent_id = ? AND session_id = ?",
                ("agent-1", "sess-1"),
            )
            rows: list[tuple[str, str]] = cursor.fetchall()
            assert len(rows) == 1
            assert rows[0] == ("user", "hello")
            db._conn.close()
        finally:
            os.unlink(path)

    def test_raises_when_file_missing(self) -> None:
        with pytest.raises(SessionDatabaseError, match="Session database not found"):
            SessionDatabase("nonexistent_sessions.db")

    def test_raises_on_invalid_schema(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            conn: sqlite3.Connection = sqlite3.connect(path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE wrong (x INTEGER)")
            conn.commit()
            conn.close()

            with pytest.raises(SessionDatabaseError, match="schema is invalid"):
                SessionDatabase(path)
        finally:
            os.unlink(path)

    async def test_close(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            SessionDatabase.initialize(path)
            db: SessionDatabase = SessionDatabase(path)
            await db.close()
        finally:
            os.unlink(path)
