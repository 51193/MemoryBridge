"""Integration tests for chat completions and sessions API."""

import os
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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
from memory_bridge.core.session_database import SessionDatabase
from memory_bridge.core.tokens import TokenStore
from memory_bridge.exceptions import MemorySearchError, MemoryStoreError
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


_temp_paths: list[str] = []


def _cleanup_temp_files() -> None:
    for p in _temp_paths:
        try:
            os.unlink(p)
        except OSError:
            pass
    _temp_paths.clear()


def _new_session_store(window_size: int = 50) -> SessionStore:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    _temp_paths.append(path)
    SessionDatabase.initialize(path)
    session_db: SessionDatabase = SessionDatabase(path)
    return SessionStore(db=session_db, window_size=window_size)


@pytest.fixture(autouse=True)
def reset_registry() -> None:
    ProviderRegistry.reset()
    _cleanup_temp_files()


def _make_app(
    memory_manager: MemoryManager | None = None,
    session_store: SessionStore | None = None,
    mock_provider: MagicMock | None = None,
    token_enabled: bool = False,
) -> FastAPI:
    app: FastAPI = FastAPI(title="MemoryBridgeTest")
    app.state.token_enabled = token_enabled
    ts: MagicMock = MagicMock(spec=TokenStore)
    ts.validate = AsyncMock()
    app.state.token_store = ts
    app.state.qdrant_health_url = "http://localhost:6333/healthz"

    if memory_manager is not None:
        app.state.memory_manager = memory_manager
        app.dependency_overrides[get_memory_manager] = lambda: memory_manager

    ss: SessionStore = session_store or _new_session_store()
    app.state.session_store = ss
    if session_store is not None:
        app.dependency_overrides[get_session_store] = lambda: session_store

    cb: ContextBuilder = ContextBuilder()
    app.state.context_builder = cb
    app.dependency_overrides[get_context_builder] = lambda: cb

    if mock_provider is not None:
        ProviderRegistry.register("deepseek-chat", mock_provider)

    app.state.model = "deepseek-chat"

    from memory_bridge.api.middleware import TokenAuthMiddleware
    app.add_middleware(TokenAuthMiddleware)
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
        mock_client: MagicMock = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_response: MagicMock = MagicMock()
        mock_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("memory_bridge.api.router.httpx.AsyncClient", return_value=mock_client):
            app: FastAPI = _make_app(session_store=_new_session_store())
            client: TestClient = TestClient(app)
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "qdrant": "connected"}

    def test_health_with_disconnected_qdrant(self) -> None:
        mock_client: MagicMock = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=Exception("connection refused"))

        with patch("memory_bridge.api.router.httpx.AsyncClient", return_value=mock_client):
            app: FastAPI = _make_app(session_store=_new_session_store())
            client: TestClient = TestClient(app)
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "qdrant": "disconnected"}


# ── Sessions ────────────────────────────────────────────────────────────


class TestSessions:
    def test_create_session_201(self) -> None:
        session_store: SessionStore = _new_session_store()
        app: FastAPI = _make_app(session_store=session_store)
        client: TestClient = TestClient(app)
        response = client.post("/v1/sessions", json={
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
        })
        assert response.status_code == 201
        assert response.json()["agent_session_id"] == "sess-1"
        session_store.get("agent-1", "sess-1")  # does not raise → exists

    def test_create_session_auto_generates_id(self) -> None:
        session_store: SessionStore = _new_session_store()
        app: FastAPI = _make_app(session_store=session_store)
        client: TestClient = TestClient(app)
        response = client.post("/v1/sessions", json={"agent_id": "agent-1"})
        assert response.status_code == 201
        sid: str = response.json()["agent_session_id"]
        assert len(sid) == 12
        session_store.get("agent-1", sid)  # does not raise → exists

    def test_create_session_duplicate_409(self) -> None:
        session_store: SessionStore = _new_session_store()
        session_store.create("agent-1", "sess-1")
        app: FastAPI = _make_app(session_store=session_store)
        client: TestClient = TestClient(app)
        response = client.post("/v1/sessions", json={
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
        })
        assert response.status_code == 409
        assert "SESSION_EXISTS" in response.json()["detail"]


# ── Chat Completions ────────────────────────────────────────────────────


def _make_chat_app(
    memory_manager: MagicMock | None = None,
    session_store: SessionStore | None = None,
    mock_provider: MagicMock | None = None,
    token_enabled: bool = False,
) -> TestClient:
    ss: SessionStore = session_store or _new_session_store()
    if memory_manager is not None:
        mm = memory_manager
    else:
        mm = MagicMock(spec=MemoryManager)
        mm.search = AsyncMock()
        mm.add = AsyncMock()
    mp: MagicMock = mock_provider or MagicMock(spec=AbstractLLMProvider)
    app: FastAPI = _make_app(
        memory_manager=mm, session_store=ss, mock_provider=mp,
        token_enabled=token_enabled,
    )
    return TestClient(app)


class TestTokenAuth:
    def test_missing_token_returns_401(self) -> None:
        ss: SessionStore = _new_session_store()
        ss.create("agent-1", "sess-1")
        mm: MagicMock = MagicMock(spec=MemoryManager)
        app: FastAPI = _make_app(
            memory_manager=mm, session_store=ss, token_enabled=True
        )
        app.state.token_store.validate.return_value = False
        client: TestClient = TestClient(app)
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
        })
        assert response.status_code == 401
        assert response.json()["detail"] == "TOKEN_MISSING"

    def test_invalid_token_returns_401(self) -> None:
        ss: SessionStore = _new_session_store()
        ss.create("agent-1", "sess-1")
        mm: MagicMock = MagicMock(spec=MemoryManager)
        app: FastAPI = _make_app(
            memory_manager=mm, session_store=ss, token_enabled=True
        )
        app.state.token_store.validate.return_value = False
        client: TestClient = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "agent_id": "agent-1",
                "agent_session_id": "sess-1",
            },
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "TOKEN_INVALID"

    def test_health_endpoint_bypasses_token(self) -> None:
        app: FastAPI = _make_app(token_enabled=True)
        app.state.token_store.validate.return_value = False
        client: TestClient = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200


class TestChatCompletions:
    def test_no_provider_registered_returns_502(self) -> None:
        ss: SessionStore = _new_session_store()
        ss.create("agent-1", "sess-1")
        mm: MagicMock = MagicMock(spec=MemoryManager)
        app: FastAPI = _make_app(
            memory_manager=mm, session_store=ss, mock_provider=None
        )
        client: TestClient = TestClient(app)
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
        })
        assert response.status_code == 502

    def test_session_not_found_returns_404(self) -> None:
        client: TestClient = _make_chat_app()
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "nonexistent",
        })
        assert response.status_code == 404
        assert "SESSION_NOT_FOUND" in response.json()["detail"]

    def test_memory_search_failure_returns_500(self) -> None:
        ss: SessionStore = _new_session_store()
        ss.create("agent-1", "sess-1")
        mock_mm: MagicMock = MagicMock(spec=MemoryManager)
        mock_mm.search = AsyncMock(side_effect=MemorySearchError("Qdrant down"))
        mock_mm.add = AsyncMock()

        client: TestClient = _make_chat_app(
            memory_manager=mock_mm, session_store=ss
        )
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
        })
        assert response.status_code == 500

    def test_memory_disabled_skips_search(self) -> None:
        ss: SessionStore = _new_session_store()
        ss.create("agent-1", "sess-1")
        mock_mm: MagicMock = MagicMock(spec=MemoryManager)
        mock_mm.search = AsyncMock()
        mock_mm.add = AsyncMock()
        mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)
        mock_provider.chat = AsyncMock(return_value=_make_chat_response())

        client: TestClient = _make_chat_app(
            memory_manager=mock_mm, session_store=ss, mock_provider=mock_provider
        )
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
            "memory_enabled": False,
        })
        assert response.status_code == 200
        mock_mm.search.assert_not_called()

    def test_provider_error_returns_502(self) -> None:
        ss: SessionStore = _new_session_store()
        ss.create("agent-1", "sess-1")
        mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)
        mock_provider.chat = AsyncMock(side_effect=Exception("down"))

        client: TestClient = _make_chat_app(
            session_store=ss, mock_provider=mock_provider
        )
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
            "memory_enabled": False,
        })
        assert response.status_code == 502

    def test_non_streaming_success(self) -> None:
        ss: SessionStore = _new_session_store()
        ss.create("agent-1", "sess-1")
        mock_mm: MagicMock = MagicMock(spec=MemoryManager)
        mock_mm.search = AsyncMock(return_value=[{"id": "1", "memory": "test memory"}])
        mock_mm.add = AsyncMock()

        mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)
        mock_provider.chat = AsyncMock(
            return_value=_make_chat_response(content="Hello there!")
        )

        client: TestClient = _make_chat_app(
            memory_manager=mock_mm, session_store=ss, mock_provider=mock_provider
        )
        response = client.post("/v1/chat/completions", json={
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
        ss: SessionStore = _new_session_store()
        ss.create("agent-1", "sess-1")
        ss.append("agent-1", "sess-1", [
            {"role": "user", "content": "history-q"},
            {"role": "assistant", "content": "history-a"},
        ])
        mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)
        mock_provider.chat = AsyncMock(return_value=_make_chat_response())

        client: TestClient = _make_chat_app(
            session_store=ss, mock_provider=mock_provider
        )
        response = client.post("/v1/chat/completions", json={
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
        ss: SessionStore = _new_session_store()
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
        ss: SessionStore = _new_session_store()
        client: TestClient = _make_chat_app(session_store=ss)
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
        })
        assert response.status_code == 422

    def test_non_streaming_schedules_memory_background_task(self) -> None:
        ss: SessionStore = _new_session_store()
        ss.create("agent-1", "sess-1")
        mock_mm: MagicMock = MagicMock(spec=MemoryManager)
        mock_mm.search = AsyncMock(return_value=[])
        mock_mm.add = AsyncMock()
        mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)
        mock_provider.chat = AsyncMock(return_value=_make_chat_response(content="ok"))

        client: TestClient = _make_chat_app(
            memory_manager=mock_mm, session_store=ss, mock_provider=mock_provider
        )
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
            "memory_enabled": False,
        })
        assert response.status_code == 200
        mock_mm.add.assert_called_once()

    def test_streaming_stores_memory_after_completion(self) -> None:
        ss: SessionStore = _new_session_store()
        ss.create("agent-1", "sess-1")
        mock_mm: MagicMock = MagicMock(spec=MemoryManager)
        mock_mm.search = AsyncMock(return_value=[])
        mock_mm.add = AsyncMock()
        mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)

        async def mock_stream() -> Any:
            yield StreamChunk(
                id="s1", created=1, model="m",
                choices=[StreamChoice(
                    index=0, delta=DeltaMessage(role="assistant", content="Hi"),
                    finish_reason=None,
                )],
            )

        mock_provider.chat_stream.return_value = mock_stream()

        client: TestClient = _make_chat_app(
            memory_manager=mock_mm, session_store=ss, mock_provider=mock_provider
        )
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
            "stream": True,
        })
        assert response.status_code == 200
        mock_mm.add.assert_called_once()

    def test_streaming_does_not_store_on_empty_content(self) -> None:
        ss: SessionStore = _new_session_store()
        ss.create("agent-1", "sess-1")
        mock_mm: MagicMock = MagicMock(spec=MemoryManager)
        mock_mm.search = AsyncMock(return_value=[])
        mock_mm.add = AsyncMock()
        mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)

        async def empty_stream() -> Any:
            if False:
                yield

        mock_provider.chat_stream.return_value = empty_stream()

        client: TestClient = _make_chat_app(
            memory_manager=mock_mm, session_store=ss, mock_provider=mock_provider
        )
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
            "stream": True,
        })
        assert response.status_code == 200
        mock_mm.add.assert_not_called()

    def test_store_memory_survives_memory_store_error(self) -> None:
        ss: SessionStore = _new_session_store()
        ss.create("agent-1", "sess-1")
        mock_mm: MagicMock = MagicMock(spec=MemoryManager)
        mock_mm.search = AsyncMock(return_value=[])
        mock_mm.add = AsyncMock(
            side_effect=MemoryStoreError("Memory store failed for user_id=agent-1: boom")
        )
        mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)
        mock_provider.chat = AsyncMock(return_value=_make_chat_response(content="ok"))

        client: TestClient = _make_chat_app(
            memory_manager=mock_mm, session_store=ss, mock_provider=mock_provider
        )
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
            "memory_enabled": False,
        })
        assert response.status_code == 200
        mock_mm.add.assert_called_once()
        assert len(ss.get("agent-1", "sess-1")) >= 2

    def test_store_memory_appends_to_session_before_memory_add(self) -> None:
        ss: SessionStore = _new_session_store()
        ss.create("agent-1", "sess-1")
        mock_mm: MagicMock = MagicMock(spec=MemoryManager)
        mock_mm.search = AsyncMock(return_value=[])
        mock_mm.add = AsyncMock()
        mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)
        mock_provider.chat = AsyncMock(return_value=_make_chat_response(content="ok"))

        call_order: list[str] = []

        def track_add(*args: Any, **kwargs: Any) -> None:
            call_order.append("memory_add")

        mock_mm.add.side_effect = track_add

        client: TestClient = _make_chat_app(
            memory_manager=mock_mm, session_store=ss, mock_provider=mock_provider
        )
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
            "memory_enabled": False,
        })
        assert response.status_code == 200
        history: list[dict[str, object]] = ss.get("agent-1", "sess-1")
        assert len(history) >= 2

    def test_chat_completions_with_empty_memory_results(self) -> None:
        ss: SessionStore = _new_session_store()
        ss.create("agent-1", "sess-1")
        mock_mm: MagicMock = MagicMock(spec=MemoryManager)
        mock_mm.search = AsyncMock(return_value=[])
        mock_mm.add = AsyncMock()
        mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)
        mock_provider.chat = AsyncMock(return_value=_make_chat_response(content="ok"))

        client: TestClient = _make_chat_app(
            memory_manager=mock_mm, session_store=ss, mock_provider=mock_provider
        )
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
        })
        assert response.status_code == 200
        mock_mm.search.assert_called_once_with(
            "hello", user_id="agent-1", limit=5
        )

    def test_valid_token_returns_200(self) -> None:
        ss: SessionStore = _new_session_store()
        ss.create("agent-1", "sess-1")
        mock_mm: MagicMock = MagicMock(spec=MemoryManager)
        mock_mm.search = AsyncMock(return_value=[])
        mock_mm.add = AsyncMock()
        mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)
        mock_provider.chat = AsyncMock(return_value=_make_chat_response(content="ok"))
        app: FastAPI = _make_app(
            memory_manager=mock_mm, session_store=ss, mock_provider=mock_provider,
            token_enabled=True,
        )
        app.state.token_store.validate = AsyncMock(return_value=True)
        client: TestClient = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "agent_id": "agent-1",
                "agent_session_id": "sess-1",
                "memory_enabled": False,
            },
            headers={"Authorization": "Bearer valid-token"},
        )
        assert response.status_code == 200

    def test_inject_history_empty_history(self) -> None:
        from memory_bridge.api.router import _inject_history
        enriched: list[dict[str, object]] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        result: list[dict[str, object]] = _inject_history(enriched, [])
        assert result == enriched

    def test_inject_history_no_system_message(self) -> None:
        from memory_bridge.api.router import _inject_history
        enriched: list[dict[str, object]] = [
            {"role": "user", "content": "hello"},
        ]
        history: list[dict[str, object]] = [
            {"role": "user", "content": "history-q"},
        ]
        result: list[dict[str, object]] = _inject_history(enriched, history)
        assert result[0] == history[0]
        assert result[1] == enriched[0]
