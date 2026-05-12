"""Tests for memory module."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from memory_bridge.config import Settings
from memory_bridge.core.memory import MemoryManager, build_mem0_config
from memory_bridge.exceptions import MemorySearchError


def _make_settings(**kwargs: str) -> Settings:
    defaults: dict[str, str] = {
        "deepseek_api_key": "sk-deepseek",
        "dashscope_api_key": "sk-dashscope",
    }
    defaults.update(kwargs)
    return Settings(_env_file=None, **defaults)  # type: ignore[arg-type, call-arg]


class TestBuildMem0Config:
    def test_builds_correct_llm_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
        settings: Settings = _make_settings()
        config: dict[str, Any] = build_mem0_config(settings)

        assert config["llm"]["provider"] == "deepseek"
        assert config["llm"]["config"]["model"] == "deepseek-chat"
        assert config["llm"]["config"]["api_key"] == "sk-deepseek"
        assert config["llm"]["config"]["deepseek_base_url"] == "https://api.deepseek.com"

    def test_builds_correct_embedder_config(self) -> None:
        settings: Settings = _make_settings()
        config: dict[str, Any] = build_mem0_config(settings)

        assert config["embedder"]["provider"] == "openai"
        assert config["embedder"]["config"]["model"] == "text-embedding-v4"
        assert config["embedder"]["config"]["api_key"] == "sk-dashscope"
        base_url: str = config["embedder"]["config"]["openai_base_url"]
        assert base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def test_builds_correct_vector_store_config(self) -> None:
        settings: Settings = _make_settings(
            qdrant_host="qdrant.local",
            qdrant_port="9999",
            embedding_dims="768",
        )
        config: dict[str, Any] = build_mem0_config(settings)

        assert config["vector_store"]["provider"] == "qdrant"
        assert config["vector_store"]["config"]["host"] == "qdrant.local"
        assert config["vector_store"]["config"]["port"] == 9999
        assert config["vector_store"]["config"]["collection_name"] == "memory_bridge"
        assert config["vector_store"]["config"]["embedding_model_dims"] == 768
        assert config["vector_store"]["config"]["on_disk"] is True

    def test_validates_secrets(self) -> None:
        settings: Settings = Settings(
            deepseek_api_key="",
            dashscope_api_key="",
            _env_file=None,
        )  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            build_mem0_config(settings)


class TestMemoryManager:
    def test_search_returns_memories(self) -> None:
        mock_memory: MagicMock = MagicMock()
        mock_memory.search.return_value = {
            "results": [
                {"id": "1", "memory": "User likes Python", "score": 0.95},
                {"id": "2", "memory": "User prefers vim", "score": 0.82},
            ]
        }

        with patch(
            "memory_bridge.core.memory.Memory.from_config", return_value=mock_memory
        ):
            manager: MemoryManager = MemoryManager({})
            results: list[dict[str, Any]] = manager.search(
                "Python", user_id="agent-1", limit=5
            )

        assert len(results) == 2
        assert results[0]["memory"] == "User likes Python"
        assert results[1]["memory"] == "User prefers vim"
        mock_memory.search.assert_called_once_with(
            "Python",
            filters={"user_id": "agent-1"},
            top_k=5,
        )

    def test_search_returns_empty_list_for_no_results(self) -> None:
        mock_memory: MagicMock = MagicMock()
        mock_memory.search.return_value = {"results": []}

        with patch(
            "memory_bridge.core.memory.Memory.from_config", return_value=mock_memory
        ):
            manager: MemoryManager = MemoryManager({})
            results: list[dict[str, Any]] = manager.search("missing", user_id="agent-1")

        assert results == []

    def test_search_missing_results_key(self) -> None:
        mock_memory: MagicMock = MagicMock()
        mock_memory.search.return_value = {}

        with patch(
            "memory_bridge.core.memory.Memory.from_config", return_value=mock_memory
        ):
            manager: MemoryManager = MemoryManager({})
            results: list[dict[str, Any]] = manager.search("query", user_id="agent-1")

        assert results == []

    def test_search_failure_raises_memory_search_error(self) -> None:
        mock_memory: MagicMock = MagicMock()
        mock_memory.search.side_effect = RuntimeError("Qdrant connection refused")

        with patch(
            "memory_bridge.core.memory.Memory.from_config", return_value=mock_memory
        ):
            manager: MemoryManager = MemoryManager({})
            with pytest.raises(MemorySearchError, match="Memory search failed"):
                manager.search("query", user_id="agent-1")

    def test_add_calls_mem0_add(self) -> None:
        mock_memory: MagicMock = MagicMock()

        with patch(
            "memory_bridge.core.memory.Memory.from_config", return_value=mock_memory
        ):
            manager: MemoryManager = MemoryManager({})
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
            manager.add(messages, user_id="agent-1", metadata={"session": "s1"})

        mock_memory.add.assert_called_once_with(
            messages,
            user_id="agent-1",
            metadata={"session": "s1"},
        )

    def test_add_with_prompt_passes_prompt_to_mem0(self) -> None:
        mock_memory: MagicMock = MagicMock()

        with patch(
            "memory_bridge.core.memory.Memory.from_config", return_value=mock_memory
        ):
            manager: MemoryManager = MemoryManager({})
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": "hello"},
            ]
            manager.add(
                messages,
                user_id="agent-1",
                prompt="重点关注技术偏好",
            )

        mock_memory.add.assert_called_once_with(
            messages,
            user_id="agent-1",
            metadata=None,
            prompt="重点关注技术偏好",
        )

    def test_add_failure_logs_and_does_not_raise(self) -> None:
        mock_memory: MagicMock = MagicMock()
        mock_memory.add.side_effect = RuntimeError("write failed")

        with patch(
            "memory_bridge.core.memory.Memory.from_config", return_value=mock_memory
        ):
            manager: MemoryManager = MemoryManager({})
            manager.add(
                [{"role": "user", "content": "hello"}],
                user_id="agent-1",
            )

        mock_memory.add.assert_called_once()
