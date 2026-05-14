"""DeepSeek HTTP client — pure I/O transport, no business logic."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from ..exceptions import MemoryBridgeError

logger: logging.Logger = logging.getLogger(__name__)


class DeepSeekHttpError(MemoryBridgeError):
    """Raised when the DeepSeek HTTP layer encounters an error."""


class DeepSeekHttpClient:
    """Thin HTTP wrapper around httpx.AsyncClient for the DeepSeek API.

    Owns the underlying connection pool and lifecycle.
    No payload construction, no response parsing — pure I/O.
    """

    def __init__(self, api_key: str, base_url: str) -> None:
        if not api_key:
            raise DeepSeekHttpError("DeepSeek API key must not be empty")
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(120.0),
        )

    async def post_json(self, endpoint: str, json: object) -> dict[str, object]:
        """POST to the API and return the JSON response body.

        Raises:
            DeepSeekHttpError: On non-200 status or connection failure.
        """
        try:
            response: httpx.Response = await self._client.post(endpoint, json=json)
        except httpx.RequestError as e:
            logger.error("deepseek http request error: %s", e)
            raise DeepSeekHttpError(f"DeepSeek request failed: {e}") from e

        if response.status_code != 200:
            logger.error(
                "deepseek http error status=%s body=%s",
                response.status_code,
                response.text[:500],
            )
            raise DeepSeekHttpError(
                f"DeepSeek returned {response.status_code}: {response.text}"
            )
        return response.json()  # type: ignore[no-any-return]

    @asynccontextmanager
    async def stream(
        self, endpoint: str, json: object
    ) -> AsyncIterator[httpx.Response]:
        """Stream a POST response, yielding the httpx.Response for line iteration.

        Raises:
            DeepSeekHttpError: On non-200 status or connection failure.
        """
        try:
            async with self._client.stream("POST", endpoint, json=json) as response:
                if response.status_code != 200:
                    body_bytes: bytes = await response.aread()
                    logger.error(
                        "deepseek stream http error status=%s body=%s",
                        response.status_code,
                        body_bytes.decode()[:500],
                    )
                    raise DeepSeekHttpError(
                        f"DeepSeek returned {response.status_code}: {body_bytes.decode()}"
                    )
                yield response
        except httpx.RequestError as e:
            logger.error("deepseek stream request error: %s", e)
            raise DeepSeekHttpError(f"DeepSeek stream request failed: {e}") from e

    async def close(self) -> None:
        """Close the underlying HTTP client and connection pool."""
        await self._client.aclose()
