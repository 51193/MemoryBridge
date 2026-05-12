from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.router import router
from .config import Settings
from .core.logging import setup_logging
from .core.memory import MemoryManager, build_mem0_config
from .providers.deepseek import DeepSeekProvider
from .providers.registry import ProviderRegistry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    settings: Settings = Settings()
    settings.validate_secrets()
    config: dict[str, object] = build_mem0_config(settings)
    app.state.memory_manager = MemoryManager(config)
    app.state.provider = DeepSeekProvider(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )
    ProviderRegistry.register(settings.deepseek_model, app.state.provider)
    yield
    await app.state.provider.close()


def create_app() -> FastAPI:
    app: FastAPI = FastAPI(
        title="MemoryBridge",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app: FastAPI = create_app()
