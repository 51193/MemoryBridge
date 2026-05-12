"""Validation tests for chat completions and sessions endpoints.

These test Pydantic request validation — all post-dependency.
They use a test app with all required dependencies overridden.
"""

from unittest.mock import MagicMock

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
from memory_bridge.providers.base import AbstractLLMProvider
from memory_bridge.providers.registry import ProviderRegistry


@pytest.fixture(autouse=True)
def reset_state() -> None:
    ProviderRegistry.reset()
    global _session_store
    _session_store = SessionStore()


def _make_app() -> FastAPI:
    app: FastAPI = FastAPI(title="MemoryBridgeTest")

    mock_mm: MagicMock = MagicMock(spec=MemoryManager)
    app.state.memory_manager = mock_mm
    app.dependency_overrides[get_memory_manager] = lambda: mock_mm
    app.dependency_overrides[get_session_store] = lambda: _session_store
    app.dependency_overrides[get_context_builder] = lambda: ContextBuilder()

    mock_provider: MagicMock = MagicMock(spec=AbstractLLMProvider)
    ProviderRegistry.register("deepseek-chat", mock_provider)

    app.include_router(router)
    return app


_session_store: SessionStore = SessionStore()


@pytest.fixture
def client() -> TestClient:
    return TestClient(_make_app())


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "qdrant": "connected"}


class TestChatCompletionsValidation:
    def test_missing_agent_id_returns_422(self, client: TestClient) -> None:
        response = client.post("/v1/chat/completions", json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hello"}],
            "agent_session_id": "sess-1",
        })
        assert response.status_code == 422

    def test_missing_session_id_returns_422(self, client: TestClient) -> None:
        response = client.post("/v1/chat/completions", json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
        })
        assert response.status_code == 422

    def test_empty_messages_returns_422(self, client: TestClient) -> None:
        response = client.post("/v1/chat/completions", json={
            "model": "deepseek-chat",
            "messages": [],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
        })
        assert response.status_code == 422

    def test_empty_agent_id_returns_422(self, client: TestClient) -> None:
        response = client.post("/v1/chat/completions", json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "",
            "agent_session_id": "sess-1",
        })
        assert response.status_code == 422

    def test_empty_session_id_returns_422(self, client: TestClient) -> None:
        response = client.post("/v1/chat/completions", json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "",
        })
        assert response.status_code == 422

    def test_memory_limit_out_of_range_returns_422(self, client: TestClient) -> None:
        response = client.post("/v1/chat/completions", json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hello"}],
            "agent_id": "agent-1",
            "agent_session_id": "sess-1",
            "memory_limit": 0,
        })
        assert response.status_code == 422

    def test_invalid_role_returns_422(self, client: TestClient) -> None:
        response = client.post("/v1/chat/completions", json={
            "model": "deepseek-chat",
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
        assert body["message_count"] == 0

    def test_create_session_with_id(self, client: TestClient) -> None:
        response = client.post("/v1/sessions", json={
            "agent_id": "agent-1",
            "agent_session_id": "sess-001",
        })
        assert response.status_code == 201
        assert response.json()["agent_session_id"] == "sess-001"

    def test_create_session_with_initial_messages(self, client: TestClient) -> None:
        response = client.post("/v1/sessions", json={
            "agent_id": "agent-1",
            "agent_session_id": "sess-001",
            "initial_messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        })
        assert response.status_code == 201
        assert response.json()["message_count"] == 2

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

    def test_create_session_invalid_message_role(self, client: TestClient) -> None:
        response = client.post("/v1/sessions", json={
            "agent_id": "agent-1",
            "initial_messages": [{"role": "invalid", "content": "hi"}],
        })
        assert response.status_code == 422
