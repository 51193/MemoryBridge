"""Tests for DeepSeek provider."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory_bridge.exceptions import ProviderNotFoundError
from memory_bridge.models.request import ChatRequest
from memory_bridge.models.response import (
    ChatResponse,
    StreamChunk,
)
from memory_bridge.providers.base import AbstractLLMProvider
from memory_bridge.providers.deepseek import DeepSeekProvider, DeepSeekProviderError
from memory_bridge.providers.deepseek_client import DeepSeekHttpClient
from memory_bridge.providers.registry import ProviderRegistry


def _make_http_client(**overrides: object) -> MagicMock:
    client: MagicMock = MagicMock(spec=DeepSeekHttpClient)
    for k, v in overrides.items():
        setattr(client, k, v)
    return client


def _make_request(**kwargs: Any) -> ChatRequest:
    defaults: dict[str, Any] = {
        "messages": [{"role": "user", "content": "hello"}],
        "agent_id": "agent-1",
        "agent_session_id": "sess-1",
    }
    defaults.update(kwargs)
    return ChatRequest(**defaults)


class TestProviderRegistry:
    def teardown_method(self) -> None:
        ProviderRegistry.reset()

    def test_register_and_get_default(self) -> None:
        mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)
        ProviderRegistry.register("deepseek-chat", mock_provider)
        assert ProviderRegistry.get_default() is mock_provider

    def test_get_default_raises_when_empty(self) -> None:
        ProviderRegistry.reset()
        with pytest.raises(ProviderNotFoundError, match="No provider registered"):
            ProviderRegistry.get_default()

    def test_reset_clears_providers(self) -> None:
        mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)
        ProviderRegistry.register("model-a", mock_provider)
        ProviderRegistry.reset()
        with pytest.raises(ProviderNotFoundError):
            ProviderRegistry.get_default()


class TestDeepSeekProvider:
    @pytest.mark.anyio
    async def test_chat_success(self) -> None:
        mock_response: dict[str, Any] = {
            "id": "resp-123",
            "created": 1700000000,
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        http_client: MagicMock = _make_http_client(
            post_json=AsyncMock(return_value=mock_response)
        )
        provider: DeepSeekProvider = DeepSeekProvider(
            client=http_client, model="deepseek-chat"
        )
        response: ChatResponse = await provider.chat(_make_request())
        assert response.id == "resp-123"
        assert response.choices[0].message.content == "Hello!"
        assert response.usage is not None
        assert response.usage.prompt_tokens == 10

    @pytest.mark.anyio
    async def test_chat_http_error(self) -> None:
        from memory_bridge.providers.deepseek_client import DeepSeekHttpError

        http_client: MagicMock = _make_http_client(
            post_json=AsyncMock(side_effect=DeepSeekHttpError("DeepSeek returned 500: boom"))
        )
        provider: DeepSeekProvider = DeepSeekProvider(
            client=http_client, model="deepseek-chat"
        )
        with pytest.raises(DeepSeekProviderError):
            await provider.chat(_make_request())

    @pytest.mark.anyio
    async def test_chat_connection_error(self) -> None:
        from memory_bridge.providers.deepseek_client import DeepSeekHttpError

        http_client: MagicMock = _make_http_client(
            post_json=AsyncMock(side_effect=DeepSeekHttpError("DeepSeek request failed: refused"))
        )
        provider: DeepSeekProvider = DeepSeekProvider(
            client=http_client, model="deepseek-chat"
        )
        with pytest.raises(DeepSeekProviderError):
            await provider.chat(_make_request())

    @pytest.mark.anyio
    async def test_chat_stream(self) -> None:
        chunks: list[str] = [
            json.dumps({
                "id": "s-1",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }),
            json.dumps({
                "id": "s-2",
                "choices": [
                    {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
                ],
            }),
            json.dumps({
                "id": "s-3",
                "choices": [
                    {"index": 0, "delta": {"content": "!"}, "finish_reason": "stop"}
                ],
            }),
        ]

        mock_response: MagicMock = MagicMock()

        async def mock_aiter_lines() -> AsyncIterator[str]:  # type: ignore[misc]
            for c in chunks:
                yield f"data: {c}"

        mock_response.aiter_lines = mock_aiter_lines

        @asynccontextmanager
        async def mock_stream(endpoint: str, json: object) -> AsyncIterator[MagicMock]:
            yield mock_response

        http_client: MagicMock = _make_http_client(stream=mock_stream)
        provider: DeepSeekProvider = DeepSeekProvider(
            client=http_client, model="deepseek-chat"
        )
        collected: list[StreamChunk] = []
        async for chunk in provider.chat_stream(_make_request()):
            collected.append(chunk)

        assert len(collected) == 3
        assert collected[0].choices[0].delta.role == "assistant"
        assert collected[1].choices[0].delta.content == "Hello"
        assert collected[2].choices[0].finish_reason == "stop"

    def test_build_payload_includes_optional_fields(self) -> None:
        http_client: MagicMock = _make_http_client()
        provider: DeepSeekProvider = DeepSeekProvider(
            client=http_client, model="deepseek-chat"
        )
        request: ChatRequest = _make_request(
            max_tokens=100,
            top_p=0.9,
            stop=["END"],
        )
        payload: dict[str, object] = provider._build_payload(request, stream=False)
        assert payload["max_tokens"] == 100
        assert payload["top_p"] == 0.9
        assert payload["stop"] == ["END"]

    def test_build_payload_excludes_defaults(self) -> None:
        http_client: MagicMock = _make_http_client()
        provider: DeepSeekProvider = DeepSeekProvider(
            client=http_client, model="deepseek-chat"
        )
        request: ChatRequest = _make_request()
        payload: dict[str, object] = provider._build_payload(request, stream=False)
        assert "max_tokens" not in payload

    def test_build_payload_thinking_enabled(self) -> None:
        http_client: MagicMock = _make_http_client()
        provider: DeepSeekProvider = DeepSeekProvider(
            client=http_client, model="deepseek-chat",
            thinking_enabled=True, reasoning_effort="max",
        )
        request: ChatRequest = _make_request()
        payload: dict[str, object] = provider._build_payload(request, stream=False)
        assert payload["thinking"] == {"type": "enabled"}
        assert payload["reasoning_effort"] == "max"
        assert "temperature" not in payload
        assert "top_p" not in payload

    def test_build_payload_thinking_enabled_default_effort(self) -> None:
        http_client: MagicMock = _make_http_client()
        provider: DeepSeekProvider = DeepSeekProvider(
            client=http_client, model="deepseek-chat",
            thinking_enabled=True,
        )
        request: ChatRequest = _make_request()
        payload: dict[str, object] = provider._build_payload(request, stream=False)
        assert payload["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in payload
        assert "temperature" not in payload

    def test_parse_response_with_reasoning_content(self) -> None:
        http_client: MagicMock = _make_http_client()
        provider: DeepSeekProvider = DeepSeekProvider(
            client=http_client, model="deepseek-chat"
        )
        data: dict[str, Any] = {
            "id": "r1",
            "created": 1700000000,
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 42.",
                        "reasoning_content": "Let me think about this...",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        response: ChatResponse = provider._parse_response(data)
        assert response.choices[0].message.content == "The answer is 42."
        assert response.choices[0].message.reasoning_content == "Let me think about this..."

    def test_parse_response_without_reasoning_content(self) -> None:
        http_client: MagicMock = _make_http_client()
        provider: DeepSeekProvider = DeepSeekProvider(
            client=http_client, model="deepseek-chat"
        )
        data: dict[str, Any] = {
            "id": "r1",
            "created": 1700000000,
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        response: ChatResponse = provider._parse_response(data)
        assert response.choices[0].message.reasoning_content is None

    @pytest.mark.anyio
    async def test_chat_stream_http_error(self) -> None:
        from memory_bridge.providers.deepseek_client import DeepSeekHttpError

        @asynccontextmanager
        async def failing_stream(endpoint: str, json: object) -> AsyncIterator[None]:
            raise DeepSeekHttpError("DeepSeek returned 429: rate limit")
            yield  # type: ignore[unreachable]

        http_client: MagicMock = _make_http_client(stream=failing_stream)
        provider: DeepSeekProvider = DeepSeekProvider(
            client=http_client, model="deepseek-chat"
        )
        with pytest.raises(DeepSeekProviderError):
            async for _ in provider.chat_stream(_make_request(memory_enabled=False)):
                pass

    @pytest.mark.anyio
    async def test_close_delegates_to_client(self) -> None:
        http_client: MagicMock = _make_http_client(close=AsyncMock())
        provider: DeepSeekProvider = DeepSeekProvider(
            client=http_client, model="deepseek-chat"
        )
        await provider.close()
        http_client.close.assert_called_once()
