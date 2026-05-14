"""Mem0 memory layer — search and store agent memories."""

import asyncio
import logging
from typing import Any

from mem0 import Memory

from ..config import Settings
from ..exceptions import MemorySearchError, MemoryStoreError
from ..logfmt import structured_debug

logger: logging.Logger = logging.getLogger(__name__)


def build_mem0_config(settings: Settings) -> dict[str, Any]:
    """Build Mem0 configuration dict from project settings."""
    config: dict[str, Any] = {
        "llm": {
            "provider": "deepseek",
            "config": {
                "model": settings.deepseek_model,
                "api_key": settings.deepseek_api_key,
                "deepseek_base_url": settings.deepseek_base_url,
                "temperature": 0.2,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": settings.embedding_model,
                "api_key": settings.dashscope_api_key,
                "openai_base_url": settings.dashscope_base_url,
                "embedding_dims": settings.embedding_dims,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": settings.qdrant_host,
                "port": settings.qdrant_port,
                "collection_name": settings.mem0_collection_name,
                "embedding_model_dims": settings.embedding_dims,
                "on_disk": True,
            },
        },
        "history_db_path": settings.mem0_history_db_path,
    }
    structured_debug(
        logger,
        "mem0 config built",
        llm=f"deepseek:{settings.deepseek_model}",
        embedder=f"openai:{settings.embedding_model}:{settings.embedding_dims}d",
        vector=f"qdrant@{settings.qdrant_host}:{settings.qdrant_port}",
    )
    return config


class MemoryManager:
    """Wraps Mem0 for memory search and storage."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._memory: Memory = Memory.from_config(config)

    async def search(self, query: str, *, user_id: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search for memories relevant to the query.

        Args:
            query: The search query text.
            user_id: The agent/user ID for memory isolation.
            limit: Maximum number of memories to return.

        Returns:
            List of memory dicts with keys: id, memory, score, etc.

        Raises:
            MemorySearchError: If the search operation fails.
        """
        q: str = query[:100]
        structured_debug(
            logger,
            "memory search",
            query=q,
            user_id=user_id,
            limit=limit,
        )
        try:
            result: dict[str, Any] = await asyncio.to_thread(
                self._memory.search,
                query,
                filters={"user_id": user_id},
                top_k=limit,
            )
            items: list[dict[str, Any]] = list(result.get("results", []))
            scores: list[float] = [float(m.get("score", 0)) for m in items]
            top_memory: str = str(items[0].get("memory", ""))[:100] if items else ""
            structured_debug(
                logger,
                "memory search → results",
                count=len(items),
                scores=scores,
                top_memory=top_memory,
            )
            return items
        except Exception as e:
            logger.error("memory search failed: %s", e)
            raise MemorySearchError(f"Memory search failed: {e}") from e

    async def add(
        self,
        messages: list[dict[str, Any]],
        *,
        user_id: str,
        metadata: dict[str, Any] | None = None,
        prompt: str | None = None,
    ) -> None:
        """Store conversation memory.

        Uses Mem0 v2 single-pass ADD-only extraction.

        Raises:
            MemoryStoreError: If the write to Mem0/Qdrant fails.
        """
        sid: str = str(metadata.get("session_id", "")) if metadata else ""
        structured_debug(
            logger,
            "memory store",
            messages_count=len(messages),
            user_id=user_id,
            session_id=sid,
        )
        try:
            extra: dict[str, Any] = {}
            if prompt:
                extra["prompt"] = prompt
            await asyncio.to_thread(
                self._memory.add, messages, user_id=user_id, metadata=metadata, **extra
            )
            structured_debug(logger, "memory stored")
        except Exception as e:
            raise MemoryStoreError(
                f"Memory store failed for user_id={user_id}: {e}"
            ) from e

    async def close(self) -> None:
        """Release Mem0 resources (Qdrant client, SQLite history DB)."""
        try:
            await asyncio.to_thread(self._memory.close)
        except RuntimeError:
            logger.warning(
                "asyncio.to_thread pool exhausted during Mem0 close; "
                "resources may leak"
            )
        except Exception:
            logger.warning("failed to close Mem0 cleanly", exc_info=True)
