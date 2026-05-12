"""DeepSeek LLM provider adapter."""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

import httpx

from ..core.logging import structured_debug, structured_info
from ..exceptions import MemoryBridgeError
from ..models.request import ChatRequest, Message
from ..models.response import (
    ChatResponse,
    Choice,
    DeltaMessage,
    StreamChoice,
    StreamChunk,
    Usage,
)
from .base import AbstractLLMProvider

logger: logging.Logger = logging.getLogger(__name__)


class ProviderError(MemoryBridgeError):
    """Raised when an LLM provider call fails."""


class DeepSeekProvider(AbstractLLMProvider):
    """DeepSeek API provider using OpenAI-compatible chat completions."""

    def __init__(self, api_key: str, base_url: str) -> None:
        self._api_key: str = api_key
        self._base_url: str = base_url.rstrip("/")
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(120.0),
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        url: str = "/v1/chat/completions"
        payload: dict[str, object] = self._build_payload(request, stream=False)

        structured_info(
            logger,
            "→ deepseek non-stream",
            model=request.model,
            messages_count=len(request.messages),
            thinking=str(request.thinking_enabled),
        )
        try:
            response: httpx.Response = await self._client.post(url, json=payload)
            if response.status_code != 200:
                logger.error(
                    "deepseek http error status=%s body=%s",
                    response.status_code,
                    response.text[:500],
                )
                raise ProviderError(
                    f"DeepSeek returned {response.status_code}: {response.text}"
                )
            result: dict[str, Any] = response.json()
        except httpx.RequestError as e:
            logger.error("deepseek request error: %s", e)
            raise ProviderError(f"DeepSeek request failed: {e}") from e

        parsed: ChatResponse = self._parse_response(result)
        usage_str: str = ""
        if parsed.usage is not None:
            usage_str = (
                f"tokens={parsed.usage.total_tokens}"
                f" finish={parsed.choices[0].finish_reason}"
            )
        structured_info(
            logger,
            "← deepseek response",
            model=parsed.model,
            usage=usage_str,
        )
        return parsed

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        url: str = "/v1/chat/completions"
        payload: dict[str, object] = self._build_payload(request, stream=True)
        request_id: str = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model: str = request.model
        created: int = int(time.time())
        chunk_count: int = 0
        total_content_len: int = 0

        structured_info(
            logger,
            "→ deepseek stream",
            model=request.model,
            messages_count=len(request.messages),
        )
        try:
            async with self._client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    body_bytes: bytes = await response.aread()
                    logger.error(
                        "deepseek stream http error status=%s body=%s",
                        response.status_code,
                        body_bytes.decode()[:500],
                    )
                    raise ProviderError(
                        f"DeepSeek returned {response.status_code}: {body_bytes.decode()}"
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data: str = line[6:]
                    if data == "[DONE]":
                        break
                    chunk: dict[str, Any] = json.loads(data)
                    chunk_count += 1
                    for choice in chunk.get("choices", []):
                        c: str = choice.get("delta", {}).get("content", "") or ""
                        total_content_len += len(c)
                    yield StreamChunk(
                        id=request_id,
                        created=created,
                        model=model,
                        choices=[
                            StreamChoice(
                                index=choice.get("index", 0),
                                delta=DeltaMessage(
                                    role=choice.get("delta", {}).get("role"),
                                    content=choice.get("delta", {}).get("content"),
                                    reasoning_content=choice.get("delta", {}).get(
                                        "reasoning_content"
                                    ),
                                ),
                                finish_reason=choice.get("finish_reason"),
                            )
                            for choice in chunk.get("choices", [])
                        ],
                    )
        except httpx.RequestError as e:
            logger.error("deepseek stream error: %s", e)
            raise ProviderError(f"DeepSeek stream request failed: {e}") from e

        structured_info(
            logger,
            "← deepseek stream complete",
            chunks=chunk_count,
            total_content_len=total_content_len,
        )

    def _build_payload(
        self, request: ChatRequest, *, stream: bool
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": request.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in request.messages
            ],
            "stream": stream,
        }

        if request.thinking_enabled:
            payload["thinking"] = {"type": "enabled"}
            if request.reasoning_effort is not None:
                payload["reasoning_effort"] = request.reasoning_effort
        else:
            payload["temperature"] = request.temperature
            if request.top_p != 1.0:
                payload["top_p"] = request.top_p

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop is not None:
            payload["stop"] = request.stop

        structured_debug(
            logger,
            "deepseek payload built",
            model=request.model,
            stream=str(stream),
            temperature=str(payload.get("temperature", "N/A")),
            max_tokens=str(payload.get("max_tokens", "N/A")),
            thinking=str(request.thinking_enabled),
        )
        return payload

    def _parse_response(self, data: dict[str, Any]) -> ChatResponse:
        usage_data: Any = data.get("usage")
        usage: Usage | None = None
        if isinstance(usage_data, dict):
            usage = Usage(
                prompt_tokens=int(usage_data.get("prompt_tokens", 0)),
                completion_tokens=int(usage_data.get("completion_tokens", 0)),
                total_tokens=int(usage_data.get("total_tokens", 0)),
            )
        return ChatResponse(
            id=str(data.get("id", "")),
            created=int(data.get("created", 0)),
            model=str(data.get("model", "")),
            choices=[
                Choice(
                    index=int(choice.get("index", 0)),
                    message=Message(
                        role=cast(
                            Literal["system", "user", "assistant", "tool"],
                            str(choice.get("message", {}).get("role", "assistant")),
                        ),
                        content=str(
                            choice.get("message", {}).get("content", "")
                        ),
                        reasoning_content=str(
                            choice.get("message", {}).get("reasoning_content", "")
                        )
                        or None,
                    ),
                    finish_reason=str(choice.get("finish_reason", "")) or None,
                )
                for choice in data.get("choices", [])
            ],
            usage=usage,
        )

    async def close(self) -> None:
        await self._client.aclose()
