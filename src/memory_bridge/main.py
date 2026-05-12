import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version as pkg_version

from fastapi import FastAPI

from .api.dependencies import init_session_store
from .api.middleware import TokenAuthMiddleware
from .api.router import router
from .config import Settings
from .core.logging import setup_logging
from .core.memory import MemoryManager, build_mem0_config
from .core.tokens import TokenStore
from .providers.deepseek import DeepSeekProvider
from .providers.registry import ProviderRegistry

logger: logging.Logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    settings: Settings = Settings()
    settings.validate_secrets()

    token_store: TokenStore = TokenStore(settings.token_db_path)
    app.state.token_store = token_store
    app.state.token_enabled = token_store.is_initialized()
    if not app.state.token_enabled:
        logger.warning(
            "Token system not initialized. "
            "Run --init-token or scripts/token_admin.sh to create a token."
        )
        logger.warning("Authentication is DISABLED until tokens exist.")

    init_session_store(settings.session_max_history)

    config: dict[str, object] = build_mem0_config(settings)
    app.state.memory_manager = MemoryManager(config)
    app.state.provider = DeepSeekProvider(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )
    app.state.qdrant_health_url = (
        f"http://{settings.qdrant_host}:{settings.qdrant_port}/healthz"
    )
    ProviderRegistry.register(settings.deepseek_model, app.state.provider)
    yield
    await app.state.provider.close()
    app.state.memory_manager.close()
    app.state.token_store.close()


def create_app() -> FastAPI:
    try:
        ver: str = pkg_version("memory-bridge")
    except Exception:
        ver = "dev"
    app: FastAPI = FastAPI(
        title="MemoryBridge",
        version=ver,
        lifespan=lifespan,
    )
    app.add_middleware(TokenAuthMiddleware)
    app.include_router(router)
    return app


app: FastAPI = create_app()
