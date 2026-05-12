"""Integration tests for chat completions and sessions API."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memory_bridge.api.dependencies import (
    get_context_builder,
    get_memory_manager,
    get_session_store,
)
from memory_bridge.api.router import router
from memory_bridge.core.context import ContextBuilder
from memory_bridge.core.memory import MemoryManager
from memory_bridge.core.session import SessionStore
from memory_bridge.exceptions import MemorySearchError
from memory_bridge.models.request import Message
from memory_bridge.models.response import (
    ChatResponse,
    Choice,
    DeltaMessage,
    StreamChoice,
    StreamChunk,
)
from memory_bridge.providers.base import AbstractLLMProvider
from memory_bridge.providers.registry import ProviderRegistry


@pytest.fixture(autouse=True)
def reset_registry() -> None:
    ProviderRegistry.reset()


def _make_boxed_session_store() -> list[SessionStore]:
    """Return a boxed SessionStore so the same instance is shared across registration and tests."""
    ss: SessionStore = SessionStore()
    return [ss]


def _make_app(
    memory_manager: MemoryManager | None = None,
    session_store: SessionStore | None = None,
    mock_provider: MagicMock | None = None,
) -> FastAPI:
    app: FastAPI = FastAPI(title="MemoryBridgeTest")

    if memory_manager is not None:
        app.state.memory_manager = memory_manager
        app.dependency_overrides[get_memory_manager] = lambda: memory_manager

    if session_store is not None:
        app.dependency_overrides[get_session_store] = lambda: session_store

    app.dependency_overrides[get_context_builder] = lambda: ContextBuilder()

    if mock_provider is not None:
        ProviderRegistry.register("deepseek-chat", mock_provider)

    app.include_router(router)
    return app


def _make_chat_response(
    content: str = "hi", model: str = "deepseek-chat"
) -> ChatResponse:
    return ChatResponse(
        id="resp-1",
        created=1700000000,
        model=model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=None,
    )


# ── Health ──────────────────────────────────────────────────────────────


class TestHealth:
    def test_health_with_connected_qdrant(self) -> None:
        mock_mm: MagicMock = MagicMock(spec=MemoryManager)
        mock_mm.search.return_value = [{"id": "1", "memory": "ok"}]

        app: FastAPI = _make_app(
            memory_manager=mock_mm,
            session_store=SessionStore(),
        )
        client: TestClient = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "qdrant": "connected"}

    def test_health_with_disconnected_qdrant(self) -> None:
        mock_mm: MagicMock = MagicMock(spec=MemoryManager)
        mock_mm.search.side_effect = MemorySearchError("down")

        app: FastAPI = _make_app(
            memory_manager=mock_mm,
            session_store=SessionStore(),
        )
        client: TestClient = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "qdrant": "disconnected"}


# ── Sessions ────────────────────────────────────────────────────────────


class TestSessions:
    def test_create_session_201(self) -> None:
        session_store: SessionStore = SessionStore()
        app: FastAPI = _make_app(session_store=session_store)
        client: TestClient = TestClient(app)
        response = client.post("/v1/sessions", json={
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
        })
        assert response.status_code == 201
        assert response.json()["agent_session_id"] == "sess-1"
        assert session_store.exists("agent-1", "sess-1")

    def test_create_session_auto_generates_id(self) -> None:
        session_store: SessionStore = SessionStore()
        app: FastAPI = _make_app(session_store=session_store)
        client: TestClient = TestClient(app)
        response = client.post("/v1/sessions", json={"agent_id": "agent-1"})
        assert response.status_code == 201
        sid: str = response.json()["agent_session_id"]
        assert len(sid) == 12
        assert session_store.exists("agent-1", sid)

    def test_create_session_duplicate_409(self) -> None:
        session_store: SessionStore = SessionStore()
        session_store.create("agent-1", "sess-1")
        app: FastAPI = _make_app(session_store=session_store)
        client: TestClient = TestClient(app)
        response = client.post("/v1/sessions", json={
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
        })
        assert response.status_code == 409
        assert "SESSION_EXISTS" in response.json()["detail"]

    def test_create_session_with_initial_messages(self) -> None:
        session_store: SessionStore = SessionStore()
        app: FastAPI = _make_app(session_store=session_store)
        client: TestClient = TestClient(app)
        response = client.post("/v1/sessions", json={
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
            "initial_messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ],
        })
        assert response.status_code == 201
        assert response.json()["message_count"] == 2
        history: list[dict[str, object]] = session_store.get("agent-1", "sess-1")
        assert len(history) == 2


# ── Chat Completions ────────────────────────────────────────────────────


def _make_chat_app(
    memory_manager: MagicMock | None = None,
    session_store: SessionStore | None = None,
    mock_provider: MagicMock | None = None,
) -> TestClient:
    ss: SessionStore = session_store or SessionStore()
    mm: MagicMock = memory_manager or MagicMock(spec=MemoryManager)
    mp: MagicMock = mock_provider or MagicMock(spec=AbstractLLMProvider)
    app: FastAPI = _make_app(
        memory_manager=mm, session_store=ss, mock_provider=mp
    )
    return TestClient(app)


class TestChatCompletions:
    def test_no_provider_registered_returns_502(self) -> None:
        ss: SessionStore = SessionStore()
        ss.create("agent-1", "sess-1")
        mm: MagicMock = MagicMock(spec=MemoryManager)
        app: FastAPI = _make_app(
            memory_manager=mm, session_store=ss, mock_provider=None
        )
        client: TestClient = TestClient(app)
        response = client.post("/v1/chat/completions", json={
            "model": "unknown-model",
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
        })
        assert response.status_code == 502

    def test_session_not_found_returns_404(self) -> None:
        client: TestClient = _make_chat_app()
        response = client.post("/v1/chat/completions", json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "nonexistent",
        })
        assert response.status_code == 404
        assert "SESSION_NOT_FOUND" in response.json()["detail"]

    def test_memory_search_failure_returns_500(self) -> None:
        ss: SessionStore = SessionStore()
        ss.create("agent-1", "sess-1")
        mock_mm: MagicMock = MagicMock(spec=MemoryManager)
        mock_mm.search.side_effect = MemorySearchError("Qdrant down")

        client: TestClient = _make_chat_app(
            memory_manager=mock_mm, session_store=ss
        )
        response = client.post("/v1/chat/completions", json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
        })
        assert response.status_code == 500

    def test_memory_disabled_skips_search(self) -> None:
        ss: SessionStore = SessionStore()
        ss.create("agent-1", "sess-1")
        mock_mm: MagicMock = MagicMock(spec=MemoryManager)
        mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)
        mock_provider.chat = AsyncMock(return_value=_make_chat_response())

        client: TestClient = _make_chat_app(
            memory_manager=mock_mm, session_store=ss, mock_provider=mock_provider
        )
        response = client.post("/v1/chat/completions", json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
            "memory_enabled": False,
        })
        assert response.status_code == 200
        mock_mm.search.assert_not_called()

    def test_provider_error_returns_502(self) -> None:
        ss: SessionStore = SessionStore()
        ss.create("agent-1", "sess-1")
        mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)
        mock_provider.chat = AsyncMock(side_effect=Exception("down"))

        client: TestClient = _make_chat_app(
            session_store=ss, mock_provider=mock_provider
        )
        response = client.post("/v1/chat/completions", json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
            "memory_enabled": False,
        })
        assert response.status_code == 502

    def test_non_streaming_success(self) -> None:
        ss: SessionStore = SessionStore()
        ss.create("agent-1", "sess-1")
        mock_mm: MagicMock = MagicMock(spec=MemoryManager)
        mock_mm.search.return_value = [{"id": "1", "memory": "test memory"}]

        mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)
        mock_provider.chat = AsyncMock(
            return_value=_make_chat_response(content="Hello there!")
        )

        client: TestClient = _make_chat_app(
            memory_manager=mock_mm, session_store=ss, mock_provider=mock_provider
        )
        response = client.post("/v1/chat/completions", json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
        })

        assert response.status_code == 200
        body: dict[str, Any] = response.json()
        assert body["choices"][0]["message"]["content"] == "Hello there!"
        mock_mm.search.assert_called_once_with(
            "hello", user_id="agent-1", limit=5
        )

    def test_session_history_injected(self) -> None:
        ss: SessionStore = SessionStore()
        ss.create("agent-1", "sess-1", messages=[
            {"role": "user", "content": "history-q"},
            {"role": "assistant", "content": "history-a"},
        ])
        mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)
        mock_provider.chat = AsyncMock(return_value=_make_chat_response())

        client: TestClient = _make_chat_app(
            session_store=ss, mock_provider=mock_provider
        )
        response = client.post("/v1/chat/completions", json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "current"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
            "memory_enabled": False,
        })
        assert response.status_code == 200

        assert mock_provider.chat.call_args is not None
        call_req: Any = mock_provider.chat.call_args[0][0]
        msgs: Any = call_req.messages
        roles: list[str] = [m.role for m in msgs]
        contents: list[str] = [m.content for m in msgs if m.role != "system"]
        assert roles[0] == "system"
        assert contents[0] == "history-q"
        assert contents[1] == "history-a"
        assert contents[2] == "current"

    def test_streaming_accept_header(self) -> None:
        ss: SessionStore = SessionStore()
        ss.create("agent-1", "sess-1")
        mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)

        async def mock_stream() -> Any:
            yield StreamChunk(
                id="s1",
                created=1,
                model="m",
                choices=[
                    StreamChoice(
                        index=0,
                        delta=DeltaMessage(role="assistant", content="Hi"),
                        finish_reason=None,
                    )
                ],
            )

        mock_provider.chat_stream.return_value = mock_stream()

        client: TestClient = _make_chat_app(
            session_store=ss, mock_provider=mock_provider
        )
        response = client.post("/v1/chat/completions", json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
            "stream": True,
            "memory_enabled": False,
        })

        assert response.status_code == 200
        content_type: str = response.headers.get("content-type", "")
        assert "text/event-stream" in content_type
        assert "data:" in response.text

    def test_missing_session_id_returns_422(self) -> None:
        ss: SessionStore = SessionStore()
        client: TestClient = _make_chat_app(session_store=ss)
        response = client.post("/v1/chat/completions", json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
        })
        assert response.status_code == 422
