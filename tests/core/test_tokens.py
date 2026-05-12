"""Tests for token module."""

from pathlib import Path

from memory_bridge.core.tokens import TokenRecord, TokenStore


class TestTokenStore:
    def test_is_initialized_false_when_empty(self, tmp_path: Path) -> None:
        store: TokenStore = TokenStore(f"{tmp_path}/tokens.db")
        assert not store.is_initialized()

    def test_is_initialized_true_after_create(self, tmp_path: Path) -> None:
        store: TokenStore = TokenStore(f"{tmp_path}/tokens.db")
        store.create("test-token")
        assert store.is_initialized()

    def test_validate_existing_token(self, tmp_path: Path) -> None:
        store: TokenStore = TokenStore(f"{tmp_path}/tokens.db")
        token: str = store.create("test")
        assert store.validate(token)

    def test_validate_nonexistent_token(self, tmp_path: Path) -> None:
        store: TokenStore = TokenStore(f"{tmp_path}/tokens.db")
        assert not store.validate("nonexistent")

    def test_validate_empty_string(self, tmp_path: Path) -> None:
        store: TokenStore = TokenStore(f"{tmp_path}/tokens.db")
        assert not store.validate("")

    def test_create_generates_32_char_hex(self, tmp_path: Path) -> None:
        store: TokenStore = TokenStore(f"{tmp_path}/tokens.db")
        token: str = store.create()
        assert len(token) == 32
        assert all(c in "0123456789abcdef" for c in token)

    def test_create_unique_tokens(self, tmp_path: Path) -> None:
        store: TokenStore = TokenStore(f"{tmp_path}/tokens.db")
        t1: str = store.create()
        t2: str = store.create()
        assert t1 != t2

    def test_list_all(self, tmp_path: Path) -> None:
        store: TokenStore = TokenStore(f"{tmp_path}/tokens.db")
        t1: str = store.create("agent-1")
        store.create("agent-2")
        records: list[TokenRecord] = store.list_all()
        assert len(records) == 2
        assert records[0].token == t1
        assert records[0].label == "agent-1"

    def test_delete_removes_token(self, tmp_path: Path) -> None:
        store: TokenStore = TokenStore(f"{tmp_path}/tokens.db")
        token: str = store.create()
        assert store.validate(token)
        store.delete(token)
        assert not store.validate(token)

    def test_delete_nonexistent_does_not_raise(self, tmp_path: Path) -> None:
        store: TokenStore = TokenStore(f"{tmp_path}/tokens.db")
        store.delete("nonexistent")

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        db_path: str = f"{tmp_path}/tokens.db"
        s1: TokenStore = TokenStore(db_path)
        token: str = s1.create()
        del s1

        s2: TokenStore = TokenStore(db_path)
        assert s2.validate(token)
        assert s2.is_initialized()
