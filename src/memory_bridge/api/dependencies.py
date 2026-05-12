"""FastAPI dependencies for dependency injection."""

from typing import cast

from fastapi import Request

from ..core.context import ContextBuilder
from ..core.memory import MemoryManager
from ..core.session import SessionStore

_session_store: SessionStore | None = None
_context_builder: ContextBuilder = ContextBuilder()


def init_session_store(max_history: int) -> None:
    global _session_store
    _session_store = SessionStore(max_history=max_history)


def get_memory_manager(request: Request) -> MemoryManager:
    return cast(MemoryManager, request.app.state.memory_manager)


def get_session_store() -> SessionStore:
    assert _session_store is not None, "session store not initialized"
    return _session_store


def get_context_builder() -> ContextBuilder:
    return _context_builder
