"""Tests for main module — app creation and lifespan lifecycle."""

from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

from memory_bridge.main import lifespan


@pytest.fixture
def mock_settings() -> Generator[MagicMock, None, None]:
    with patch("memory_bridge.main.Settings") as mock_cls:
        settings: MagicMock = MagicMock()
        settings.deepseek_api_key = "sk-test"
        settings.dashscope_api_key = "ds-test"
        settings.token_db_path = "data/test_tokens.db"
        settings.deepseek_model = "deepseek-chat"
        settings.deepseek_base_url = "https://api.deepseek.com"
        settings.deepseek_thinking_enabled = False
        settings.deepseek_reasoning_effort = None
        settings.qdrant_host = "localhost"
        settings.qdrant_port = 6333
        settings.session_window_size = 10
        settings.session_db_path = "data/test_sessions.db"
        settings.mem0_history_db_path = "data/mem0_history.db"
        settings.embedding_model = "text-embedding-v4"
        settings.embedding_dims = 1024
        settings.dashscope_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        settings.mem0_collection_name = "memory_bridge"
        settings.prompts_dir = "prompts"
        mock_cls.return_value = settings
        yield settings


@pytest.fixture
def mock_components() -> Generator[dict[str, MagicMock], None, None]:
    with (
        patch("memory_bridge.main.TokenDatabase") as mock_tdb,
        patch("memory_bridge.main.TokenStore") as mock_ts,
        patch("memory_bridge.main.build_mem0_config", return_value={"key": "val"}),
        patch("memory_bridge.main.MemoryManager") as mock_mm,
        patch("memory_bridge.main.DeepSeekHttpClient") as mock_dhc,
        patch("memory_bridge.main.DeepSeekProvider") as mock_dp,
        patch("memory_bridge.main.SessionStore") as mock_ss,
        patch("memory_bridge.main.SessionDatabase") as mock_sdb,
        patch("memory_bridge.main.ContextBuilder") as mock_cb,
    ):
        tdb: MagicMock = MagicMock()
        tdb.close = AsyncMock()
        mock_tdb.return_value = tdb

        ts: MagicMock = MagicMock()
        ts.is_initialized.return_value = False
        ts.validate = AsyncMock(return_value=True)
        ts.close = AsyncMock()
        mock_ts.return_value = ts

        mm: MagicMock = MagicMock()
        mm.close = AsyncMock()
        mock_mm.return_value = mm

        dhc: MagicMock = MagicMock()
        dhc.close = AsyncMock()
        mock_dhc.return_value = dhc

        dp: MagicMock = MagicMock()
        dp.close = AsyncMock()
        mock_dp.return_value = dp

        ss: MagicMock = MagicMock()
        mock_ss.return_value = ss

        cb: MagicMock = MagicMock()
        mock_cb.return_value = cb

        sdb: MagicMock = MagicMock()
        sdb.close = AsyncMock()
        mock_sdb.return_value = sdb

        yield {
            "token_db": tdb,
            "token_store": ts,
            "memory_manager": mm,
            "deepseek_http_client": dhc,
            "provider": dp,
            "session_store": ss,
            "session_db": sdb,
            "context_builder": cb,
        }


class TestCreateApp:
    def test_create_app_returns_fastapi_instance(self) -> None:
        from memory_bridge.main import create_app
        app: FastAPI = create_app()
        assert app.title == "MemoryBridge"

    def test_create_app_has_health_route(self) -> None:
        from memory_bridge.main import create_app
        app: FastAPI = create_app()
        routes: list[str] = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/health" in routes
        assert "/v1/chat/completions" in routes


class TestLifespan:
    async def test_creates_token_store(
        self, mock_settings: MagicMock, mock_components: dict[str, MagicMock]
    ) -> None:
        app: FastAPI = FastAPI(title="Test")
        async with lifespan(app):
            assert hasattr(app.state, "token_store")

    async def test_creates_memory_manager(
        self, mock_settings: MagicMock, mock_components: dict[str, MagicMock]
    ) -> None:
        app: FastAPI = FastAPI(title="Test")
        async with lifespan(app):
            assert hasattr(app.state, "memory_manager")

    async def test_registers_provider(
        self, mock_settings: MagicMock, mock_components: dict[str, MagicMock]
    ) -> None:
        app: FastAPI = FastAPI(title="Test")
        async with lifespan(app):
            from memory_bridge.providers.registry import ProviderRegistry
            assert ProviderRegistry.get_default() is mock_components["provider"]

    async def test_sets_token_disabled_when_no_tokens(
        self,
        mock_settings: MagicMock,
        mock_components: dict[str, MagicMock],
    ) -> None:
        mock_components["token_store"].is_initialized.return_value = False
        app: FastAPI = FastAPI(title="Test")
        async with lifespan(app):
            assert not app.state.token_enabled

    async def test_sets_token_enabled_when_tokens_exist(
        self, mock_settings: MagicMock, mock_components: dict[str, MagicMock]
    ) -> None:
        mock_components["token_store"].is_initialized.return_value = True
        app: FastAPI = FastAPI(title="Test")
        async with lifespan(app):
            assert app.state.token_enabled

    async def test_closes_all_components_on_shutdown(
        self, mock_settings: MagicMock, mock_components: dict[str, MagicMock]
    ) -> None:
        app: FastAPI = FastAPI(title="Test")
        async with lifespan(app):
            pass
        mock_components["provider"].close.assert_called_once()
        mock_components["memory_manager"].close.assert_called_once()
        mock_components["token_store"].close.assert_called_once()
        mock_components["session_db"].close.assert_called_once()

    async def test_does_not_crash_if_close_fails(
        self, mock_settings: MagicMock, mock_components: dict[str, MagicMock]
    ) -> None:
        mm = mock_components["memory_manager"]
        ts = mock_components["token_store"]
        sdb = mock_components["session_db"]
        close_order: list[str] = []

        async def tracked_mm_close() -> None:
            close_order.append("memory_manager")
            raise RuntimeError("fail")

        async def tracked_ts_close() -> None:
            close_order.append("token_store")
            raise RuntimeError("fail too")

        async def tracked_sdb_close() -> None:
            close_order.append("session_db")
            raise RuntimeError("fail three")

        mm.close = tracked_mm_close
        ts.close = tracked_ts_close
        sdb.close = tracked_sdb_close

        app: FastAPI = FastAPI(title="Test")
        try:
            async with lifespan(app):
                pass
        except RuntimeError:
            pass

        mock_components["provider"].close.assert_called_once()
        assert "memory_manager" in close_order
