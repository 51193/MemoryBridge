"""Tests for token module."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from memory_bridge.core.token_database import TokenDatabase
from memory_bridge.core.tokens import TokenRecord, TokenStore


def _init_store(db_path: str) -> TokenStore:
    TokenDatabase.initialize(db_path)
    return TokenStore(TokenDatabase(db_path))


class TestTokenStore:
    def test_is_initialized_false_when_empty(self, tmp_path: Path) -> None:
        store: TokenStore = _init_store(f"{tmp_path}/tokens.db")
        assert not store.is_initialized()

    def test_is_initialized_true_after_create(self, tmp_path: Path) -> None:
        store: TokenStore = _init_store(f"{tmp_path}/tokens.db")
        store.create("test-token")
        assert store.is_initialized()

    async def test_validate_existing_token(self, tmp_path: Path) -> None:
        store: TokenStore = _init_store(f"{tmp_path}/tokens.db")
        token: str = store.create("test")
        assert await store.validate(token)

    async def test_validate_nonexistent_token(self, tmp_path: Path) -> None:
        store: TokenStore = _init_store(f"{tmp_path}/tokens.db")
        assert not await store.validate("nonexistent")

    async def test_validate_empty_string(self, tmp_path: Path) -> None:
        store: TokenStore = _init_store(f"{tmp_path}/tokens.db")
        assert not await store.validate("")

    def test_create_generates_32_char_hex(self, tmp_path: Path) -> None:
        store: TokenStore = _init_store(f"{tmp_path}/tokens.db")
        token: str = store.create()
        assert len(token) == 32
        assert all(c in "0123456789abcdef" for c in token)

    def test_create_unique_tokens(self, tmp_path: Path) -> None:
        store: TokenStore = _init_store(f"{tmp_path}/tokens.db")
        t1: str = store.create()
        t2: str = store.create()
        assert t1 != t2

    def test_list_all(self, tmp_path: Path) -> None:
        store: TokenStore = _init_store(f"{tmp_path}/tokens.db")
        t1: str = store.create("agent-1")
        store.create("agent-2")
        records: list[TokenRecord] = store.list_all()
        assert len(records) == 2
        assert records[0].token == t1
        assert records[0].label == "agent-1"

    async def test_delete_removes_token(self, tmp_path: Path) -> None:
        store: TokenStore = _init_store(f"{tmp_path}/tokens.db")
        token: str = store.create()
        assert await store.validate(token)
        store.delete(token)
        assert not await store.validate(token)

    async def test_validate_runs_in_thread_pool(self, tmp_path: Path) -> None:
        store: TokenStore = _init_store(f"{tmp_path}/tokens.db")
        token: str = store.create("test")
        with patch("memory_bridge.core.tokens.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = (1,)
            result: bool = await store.validate(token)
            assert result is True
            mock_to_thread.assert_called_once()

    async def test_validate_returns_false_on_sqlite_error(self, tmp_path: Path) -> None:
        store: TokenStore = _init_store(f"{tmp_path}/tokens.db")
        store._db._conn.close()
        result: bool = await store.validate("any-token")
        assert result is False

    def test_delete_nonexistent_does_not_raise(self, tmp_path: Path) -> None:
        store: TokenStore = _init_store(f"{tmp_path}/tokens.db")
        store.delete("nonexistent")

    async def test_persistence_across_instances(self, tmp_path: Path) -> None:
        db_path: str = f"{tmp_path}/tokens.db"
        s1: TokenStore = _init_store(db_path)
        token: str = s1.create()
        del s1

        s2: TokenStore = TokenStore(TokenDatabase(db_path))
        assert await s2.validate(token)
        assert s2.is_initialized()

    async def test_close_closes_connection(self, tmp_path: Path) -> None:
        store: TokenStore = _init_store(f"{tmp_path}/tokens.db")
        with patch("memory_bridge.core.tokens.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = None
            await store.close()
        assert mock_to_thread.called

    async def test_close_propagates_errors(self, tmp_path: Path) -> None:
        store: TokenStore = _init_store(f"{tmp_path}/tokens.db")
        exc: sqlite3.ProgrammingError = sqlite3.ProgrammingError("already closed")
        with patch(
            "memory_bridge.core.tokens.asyncio.to_thread",
            side_effect=exc,
        ):
            with pytest.raises(sqlite3.ProgrammingError, match="already closed"):
                await store.close()
