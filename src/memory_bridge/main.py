import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

from fastapi import FastAPI

from .api.middleware import TokenAuthMiddleware
from .api.router import router
from .config import Settings
from .core.context import ContextBuilder
from .core.logging import setup_logging
from .core.memory import MemoryManager, build_mem0_config
from .core.session import SessionStore
from .core.session_database import SessionDatabase
from .core.token_database import TokenDatabase
from .core.tokens import TokenStore
from .providers.deepseek import DeepSeekProvider
from .providers.deepseek_client import DeepSeekHttpClient
from .providers.registry import ProviderRegistry

logger: logging.Logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    settings: Settings = Settings()

    token_db: TokenDatabase = TokenDatabase(settings.token_db_path)
    app.state.token_store = TokenStore(token_db)
    app.state.token_enabled = app.state.token_store.is_initialized()
    if not app.state.token_enabled:
        logger.warning(
            "Token system not initialized. "
            "Run --init-token or scripts/token_admin.sh to create a token."
        )
        logger.warning("Authentication is DISABLED until tokens exist.")

    app.state.session_db = SessionDatabase(settings.session_db_path)
    app.state.session_store = SessionStore(
        db=app.state.session_db,
        window_size=settings.session_window_size,
    )
    app.state.context_builder = ContextBuilder()

    deepseek_client: DeepSeekHttpClient = DeepSeekHttpClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )
    app.state.provider = DeepSeekProvider(
        client=deepseek_client,
        model=settings.deepseek_model,
        thinking_enabled=settings.deepseek_thinking_enabled,
        reasoning_effort=settings.deepseek_reasoning_effort,
    )
    app.state.model = settings.deepseek_model
    app.state.qdrant_health_url = (
        f"http://{settings.qdrant_host}:{settings.qdrant_port}/healthz"
    )

    config: dict[str, object] = build_mem0_config(settings)
    app.state.memory_manager = MemoryManager(config)

    ProviderRegistry.register(settings.deepseek_model, app.state.provider)
    yield
    await app.state.provider.close()
    await app.state.memory_manager.close()
    await app.state.token_store.close()
    await app.state.session_db.close()


def create_app() -> FastAPI:
    try:
        ver: str = pkg_version("memory-bridge")
    except PackageNotFoundError:
        ver = "dev"
    except Exception:
        ver = "dev"
        logger.warning("could not determine package version", exc_info=True)
    app: FastAPI = FastAPI(
        title="MemoryBridge",
        version=ver,
        lifespan=lifespan,
    )
    app.add_middleware(TokenAuthMiddleware)
    app.include_router(router)
    return app


app: FastAPI = create_app()
