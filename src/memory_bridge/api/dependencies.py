"""FastAPI dependencies for dependency injection.

All dependencies are stored on app.state and resolved via Request.
No module-level mutable singletons.
"""

from typing import cast

from fastapi import Request

from ..core.context import ContextBuilder
from ..core.memory import MemoryManager
from ..core.session import SessionStore
from ..providers.registry import ProviderRegistry


def get_memory_manager(request: Request) -> MemoryManager:
    return cast(MemoryManager, request.app.state.memory_manager)


def get_session_store(request: Request) -> SessionStore:
    return cast(SessionStore, request.app.state.session_store)


def get_context_builder(request: Request) -> ContextBuilder:
    return cast(ContextBuilder, request.app.state.context_builder)


def get_provider_registry(request: Request) -> ProviderRegistry:
    return cast(ProviderRegistry, request.app.state.provider_registry)
