"""FastAPI dependencies for dependency injection."""

from typing import cast

from fastapi import Request

from ..core.context import ContextBuilder
from ..core.memory import MemoryManager
from ..core.session import SessionStore

_session_store: SessionStore = SessionStore()
_context_builder: ContextBuilder = ContextBuilder()


def get_memory_manager(request: Request) -> MemoryManager:
    return cast(MemoryManager, request.app.state.memory_manager)


def get_session_store() -> SessionStore:
    return _session_store


def get_context_builder() -> ContextBuilder:
    return _context_builder
