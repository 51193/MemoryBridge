"""Validation tests for chat completions and sessions endpoints.

These test Pydantic request validation — all post-dependency.
They use a test app with all required dependencies overridden.
"""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memory_bridge.api.dependencies import (
    get_context_builder,
    get_memory_manager,
    get_provider_registry,
    get_session_store,
)
from memory_bridge.api.router import router
from memory_bridge.core.context import ContextBuilder
from memory_bridge.core.memory import MemoryManager
from memory_bridge.core.session import SessionStore
from memory_bridge.core.session_database import SessionDatabase
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


def _make_temp_session_store() -> SessionStore:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    _temp_paths.append(path)
    SessionDatabase.initialize(path)
    session_db: SessionDatabase = SessionDatabase(path)
    return SessionStore(db=session_db, window_size=50)


@pytest.fixture(autouse=True)
def reset_state() -> None:
    _cleanup_temp_files()


def _make_app() -> FastAPI:
    session_store: SessionStore = _make_temp_session_store()
    app: FastAPI = FastAPI(title="MemoryBridgeTest")
    app.state.token_enabled = False
    app.state.qdrant_health_url = "http://localhost:6333/healthz"

    mock_mm: MagicMock = MagicMock(spec=MemoryManager)
    mock_mm.search = AsyncMock()
    mock_mm.add = AsyncMock()
    app.state.memory_manager = mock_mm
    app.dependency_overrides[get_memory_manager] = lambda: mock_mm
    app.dependency_overrides[get_session_store] = lambda: session_store
    app.dependency_overrides[get_context_builder] = lambda: ContextBuilder()

    mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)
    registry: ProviderRegistry = ProviderRegistry()
    registry.register("deepseek-chat", mock_provider)
    app.state.provider_registry = registry
    app.dependency_overrides[get_provider_registry] = lambda: registry

    app.include_router(router)
    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(_make_app())


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        mock_client: MagicMock = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_response: MagicMock = MagicMock()
        mock_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("memory_bridge.api.router.httpx.AsyncClient", return_value=mock_client):
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "qdrant": "connected"}


class TestChatCompletionsValidation:
    def test_missing_agent_id_returns_422(self, client: TestClient) -> None:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "agent_session_id": "sess-1",
        })
        assert response.status_code == 422

    def test_missing_session_id_returns_422(self, client: TestClient) -> None:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
        })
        assert response.status_code == 422

    def test_empty_messages_returns_422(self, client: TestClient) -> None:
        response = client.post("/v1/chat/completions", json={
            "messages": [],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
        })
        assert response.status_code == 422

    def test_empty_agent_id_returns_422(self, client: TestClient) -> None:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "",
            "agent_session_id": "sess-1",
        })
        assert response.status_code == 422

    def test_empty_session_id_returns_422(self, client: TestClient) -> None:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "",
        })
        assert response.status_code == 422

    def test_memory_limit_out_of_range_returns_422(self, client: TestClient) -> None:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
            "memory_limit": 0,
        })
        assert response.status_code == 422

    def test_invalid_role_returns_422(self, client: TestClient) -> None:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "invalid", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
        })
        assert response.status_code == 422


class TestSessionsValidation:
    def test_create_session_minimal(self, client: TestClient) -> None:
        response = client.post("/v1/sessions", json={
            "agent_id": "agent-1",
        })
        assert response.status_code == 201
        body = response.json()
        assert body["agent_id"] == "agent-1"
        assert len(body["agent_session_id"]) == 12

    def test_create_session_with_id(self, client: TestClient) -> None:
        response = client.post("/v1/sessions", json={
            "agent_id": "agent-1",
            "agent_session_id": "sess-001",
        })
        assert response.status_code == 201
        assert response.json()["agent_session_id"] == "sess-001"

    def test_create_session_duplicate_returns_409(self, client: TestClient) -> None:
        client.post("/v1/sessions", json={
            "agent_id": "agent-1",
            "agent_session_id": "sess-001",
        })
        response = client.post("/v1/sessions", json={
            "agent_id": "agent-1",
            "agent_session_id": "sess-001",
        })
        assert response.status_code == 409
        assert "SESSION_EXISTS" in response.json()["detail"]

    def test_create_session_missing_agent_id(self, client: TestClient) -> None:
        response = client.post("/v1/sessions", json={
            "agent_session_id": "sess-1",
        })
        assert response.status_code == 422
