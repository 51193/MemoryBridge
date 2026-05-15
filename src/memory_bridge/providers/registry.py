"""Provider registry — maps model names to LLM providers.

Instance-based — no class-level mutable state. Create one instance,
register providers, and inject it where needed via FastAPI dependencies.
"""

from ..exceptions import ProviderNotFoundError
from .base import AbstractLLMProvider


class ProviderRegistry:
    """Registry for LLM providers, keyed by model name."""

    def __init__(self) -> None:
        self._providers: dict[str, AbstractLLMProvider] = {}

    def register(self, model: str, provider: AbstractLLMProvider) -> None:
        self._providers[model] = provider

    def get_default(self) -> AbstractLLMProvider:
        if not self._providers:
            raise ProviderNotFoundError("No provider registered")
        return next(iter(self._providers.values()))

    def reset(self) -> None:
        self._providers.clear()
