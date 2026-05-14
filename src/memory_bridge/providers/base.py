"""Abstract LLM provider interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..models.request import ChatRequest
from ..models.response import ChatResponse, StreamChunk


class AbstractLLMProvider(ABC):
    """Base class for LLM provider adapters."""

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a non-streaming chat completion request."""

    @abstractmethod
    def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Send a streaming chat completion request, yielding chunks.

        Subclasses implement this as an async generator.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release provider resources (connections, pools, etc.)."""
