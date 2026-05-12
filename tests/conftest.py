from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memory_bridge.api.router import router


@pytest.fixture
def client() -> TestClient:
    app: FastAPI = FastAPI(title="MemoryBridgeTest")
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def mock_memory_manager() -> MagicMock:
    return MagicMock()
