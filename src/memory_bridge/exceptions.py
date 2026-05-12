"""Custom exceptions for MemoryBridge."""


class MemoryBridgeError(Exception):
    """Base exception for all MemoryBridge errors."""


class MemorySearchError(MemoryBridgeError):
    """Raised when memory retrieval fails."""


class MemoryStoreError(MemoryBridgeError):
    """Raised when memory storage fails."""


class ProviderNotFoundError(MemoryBridgeError):
    """Raised when no LLM provider is registered for a given model."""
