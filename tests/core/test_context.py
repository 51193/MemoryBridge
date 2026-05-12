"""Tests for context module."""

from memory_bridge.core.context import ContextBuilder


class TestContextBuilder:
    def test_injects_memories_into_existing_system_message(self) -> None:
        builder: ContextBuilder = ContextBuilder()
        messages: list[dict[str, object]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        memories: list[dict[str, object]] = [
            {"id": "1", "memory": "User likes Python", "score": 0.95},
            {"id": "2", "memory": "User prefers vim", "score": 0.82},
        ]

        result: list[dict[str, object]] = builder.build(messages, memories)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        system_content: str = str(result[0]["content"])
        assert "User likes Python" in system_content
        assert "User prefers vim" in system_content
        assert "[相关历史记忆]" in system_content
        assert "You are a helpful assistant." in system_content
        assert result[1]["role"] == "user"

    def test_creates_system_message_when_none_exists(self) -> None:
        builder: ContextBuilder = ContextBuilder()
        messages: list[dict[str, object]] = [
            {"role": "user", "content": "Hello"},
        ]
        memories: list[dict[str, object]] = [
            {"id": "1", "memory": "User likes Python", "score": 0.95},
        ]

        result: list[dict[str, object]] = builder.build(messages, memories)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        system_content: str = str(result[0]["content"])
        assert "User likes Python" in system_content
        assert "[相关历史记忆]" in system_content
        assert result[1]["role"] == "user"

    def test_empty_memories_returns_messages_with_system(self) -> None:
        builder: ContextBuilder = ContextBuilder()
        messages: list[dict[str, object]] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]

        result: list[dict[str, object]] = builder.build(messages, [])

        assert len(result) == 2
        system_content: str = str(result[0]["content"])
        assert "[相关历史记忆]" in system_content
        assert "\n\n" in system_content

    def test_memory_without_memory_key_is_skipped(self) -> None:
        builder: ContextBuilder = ContextBuilder()
        messages: list[dict[str, object]] = [
            {"role": "user", "content": "Hi"},
        ]
        memories: list[dict[str, object]] = [
            {"id": "1"},
            {"id": "2", "memory": "Valid memory"},
        ]

        result: list[dict[str, object]] = builder.build(messages, memories)

        system_content: str = str(result[0]["content"])
        assert "Valid memory" in system_content

    def test_original_messages_not_mutated(self) -> None:
        builder: ContextBuilder = ContextBuilder()
        messages: list[dict[str, object]] = [
            {"role": "system", "content": "Original"},
        ]
        original_copy: list[dict[str, object]] = [dict(m) for m in messages]

        builder.build(messages, [{"memory": "test", "id": "1"}])

        assert messages == original_copy
