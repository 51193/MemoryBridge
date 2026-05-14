"""Tests for DeepSeek provider."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from memory_bridge.exceptions import ProviderNotFoundError
from memory_bridge.models.request import ChatRequest
from memory_bridge.models.response import (
    ChatResponse,
    StreamChunk,
)
from memory_bridge.providers.base import AbstractLLMProvider
from memory_bridge.providers.deepseek import DeepSeekProvider, ProviderError
from memory_bridge.providers.registry import ProviderRegistry


def _make_request(**kwargs: Any) -> ChatRequest:
    defaults: dict[str, Any] = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "hello"}],
        "agent_id": "agent-1",
        "agent_session_id": "sess-1",
    }
    defaults.update(kwargs)
    return ChatRequest(**defaults)


class TestProviderRegistry:
    def teardown_method(self) -> None:
        ProviderRegistry.reset()

    def test_register_and_get(self) -> None:
        mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)
        ProviderRegistry.register("deepseek-chat", mock_provider)
        assert ProviderRegistry.get("deepseek-chat") is mock_provider

    def test_get_unregistered_raises(self) -> None:
        with pytest.raises(ProviderNotFoundError, match="unknown-model"):
            ProviderRegistry.get("unknown-model")

    def test_reset_clears_providers(self) -> None:
        mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)
        ProviderRegistry.register("model-a", mock_provider)
        ProviderRegistry.reset()
        with pytest.raises(ProviderNotFoundError):
            ProviderRegistry.get("model-a")


class TestDeepSeekProvider:
    @pytest.fixture
    async def provider(self) -> DeepSeekProvider:  # type: ignore[misc]  # noqa: PT004
        p: DeepSeekProvider = DeepSeekProvider(
            api_key="sk-test", base_url="https://api.deepseek.com"
        )
        yield p
        await p.close()

    @pytest.mark.anyio
    async def test_chat_success(self) -> None:
        mock_client: MagicMock = MagicMock(spec=httpx.AsyncClient)
        mock_response: MagicMock = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
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
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch(
            "memory_bridge.providers.deepseek.httpx.AsyncClient",
            return_value=mock_client,
        ):
            provider: DeepSeekProvider = DeepSeekProvider(
                api_key="sk-test", base_url="https://api.deepseek.com"
            )
            try:
                request: ChatRequest = _make_request()
                response: ChatResponse = await provider.chat(request)
                assert response.id == "resp-123"
                assert response.choices[0].message.content == "Hello!"
                assert response.usage is not None
                assert response.usage.prompt_tokens == 10
            finally:
                await provider.close()

    @pytest.mark.anyio
    async def test_chat_http_error(self) -> None:
        mock_client: MagicMock = MagicMock(spec=httpx.AsyncClient)
        mock_response: MagicMock = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch(
            "memory_bridge.providers.deepseek.httpx.AsyncClient",
            return_value=mock_client,
        ):
            provider: DeepSeekProvider = DeepSeekProvider(
                api_key="sk-test", base_url="https://api.deepseek.com"
            )
            try:
                with pytest.raises(ProviderError, match="DeepSeek returned 500"):
                    await provider.chat(_make_request())
            finally:
                await provider.close()

    @pytest.mark.anyio
    async def test_chat_connection_error(self) -> None:
        mock_client: MagicMock = MagicMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with patch(
            "memory_bridge.providers.deepseek.httpx.AsyncClient",
            return_value=mock_client,
        ):
            provider: DeepSeekProvider = DeepSeekProvider(
                api_key="sk-test", base_url="https://api.deepseek.com"
            )
            try:
                with pytest.raises(ProviderError, match="DeepSeek request failed"):
                    await provider.chat(_make_request())
            finally:
                await provider.close()

    @pytest.mark.anyio
    async def test_chat_stream(self) -> None:
        mock_client: MagicMock = MagicMock(spec=httpx.AsyncClient)
        mock_response: MagicMock = MagicMock()
        mock_response.status_code = 200

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

        mock_client.stream = MagicMock()
        mock_client.stream.return_value.__aenter__ = AsyncMock(
            return_value=mock_response
        )
        mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=None)

        async def mock_aiter_lines() -> AsyncMock:  # type: ignore[misc]
            for c in chunks:
                yield f"data: {c}"

        mock_response.aiter_lines = mock_aiter_lines

        with patch(
            "memory_bridge.providers.deepseek.httpx.AsyncClient",
            return_value=mock_client,
        ):
            provider: DeepSeekProvider = DeepSeekProvider(
                api_key="sk-test", base_url="https://api.deepseek.com"
            )
            try:
                collected: list[StreamChunk] = []
                async for chunk in provider.chat_stream(_make_request()):
                    collected.append(chunk)

                assert len(collected) == 3
                assert collected[0].choices[0].delta.role == "assistant"
                assert collected[1].choices[0].delta.content == "Hello"
                assert collected[2].choices[0].finish_reason == "stop"
            finally:
                await provider.close()

    def test_build_payload_includes_optional_fields(self) -> None:
        provider: DeepSeekProvider = DeepSeekProvider(
            api_key="sk-test", base_url="https://api.deepseek.com"
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
        provider: DeepSeekProvider = DeepSeekProvider(
            api_key="sk-test", base_url="https://api.deepseek.com"
        )
        request: ChatRequest = _make_request()
        payload: dict[str, object] = provider._build_payload(request, stream=False)
        assert "max_tokens" not in payload

    def test_build_payload_thinking_enabled(self) -> None:
        provider: DeepSeekProvider = DeepSeekProvider(
            api_key="sk-test", base_url="https://api.deepseek.com"
        )
        request: ChatRequest = _make_request(
            thinking_enabled=True,
            reasoning_effort="max",
        )
        payload: dict[str, object] = provider._build_payload(request, stream=False)
        assert payload["thinking"] == {"type": "enabled"}
        assert payload["reasoning_effort"] == "max"
        assert "temperature" not in payload
        assert "top_p" not in payload

    def test_build_payload_thinking_enabled_default_effort(self) -> None:
        provider: DeepSeekProvider = DeepSeekProvider(
            api_key="sk-test", base_url="https://api.deepseek.com"
        )
        request: ChatRequest = _make_request(thinking_enabled=True)
        payload: dict[str, object] = provider._build_payload(request, stream=False)
        assert payload["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in payload
        assert "temperature" not in payload

    def test_parse_response_with_reasoning_content(self) -> None:
        provider: DeepSeekProvider = DeepSeekProvider(
            api_key="sk-test", base_url="https://api.deepseek.com"
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
        provider: DeepSeekProvider = DeepSeekProvider(
            api_key="sk-test", base_url="https://api.deepseek.com"
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
        mock_client: MagicMock = MagicMock(spec=httpx.AsyncClient)
        mock_response: MagicMock = MagicMock()
        mock_response.status_code = 429
        mock_response.aread = AsyncMock(return_value=b"Rate limit exceeded")

        mock_client.stream = MagicMock()
        mock_client.stream.return_value.__aenter__ = AsyncMock(
            return_value=mock_response
        )
        mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "memory_bridge.providers.deepseek.httpx.AsyncClient",
            return_value=mock_client,
        ):
            provider: DeepSeekProvider = DeepSeekProvider(
                api_key="sk-test", base_url="https://api.deepseek.com"
            )
            try:
                with pytest.raises(ProviderError, match="DeepSeek returned 429"):
                    async for _ in provider.chat_stream(
                        _make_request(memory_enabled=False)
                    ):
                        pass
            finally:
                await provider.close()

    @pytest.mark.anyio
    async def test_close_closes_http_client(self) -> None:
        mock_client: MagicMock = MagicMock(spec=httpx.AsyncClient)
        mock_client.aclose = AsyncMock()

        with patch(
            "memory_bridge.providers.deepseek.httpx.AsyncClient",
            return_value=mock_client,
        ):
            provider: DeepSeekProvider = DeepSeekProvider(
                api_key="sk-test", base_url="https://api.deepseek.com"
            )
            await provider.close()

        mock_client.aclose.assert_called_once()
