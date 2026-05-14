"""Provider registry — maps model names to LLM providers."""

from ..exceptions import ProviderNotFoundError
from .base import AbstractLLMProvider


class ProviderRegistry:
    """Singleton registry for LLM providers.

    Providers are registered by model name. Use get_default() to resolve
    the currently registered provider.
    """

    _providers: dict[str, AbstractLLMProvider] = {}

    @classmethod
    def register(cls, model: str, provider: AbstractLLMProvider) -> None:
        """Register a provider for a given model name."""
        cls._providers[model] = provider

    @classmethod
    def get_default(cls) -> AbstractLLMProvider:
        """Return the single registered provider.

        Raises:
            ProviderNotFoundError: If no provider has been registered.
        """
        if not cls._providers:
            raise ProviderNotFoundError("No provider registered")
        return next(iter(cls._providers.values()))

    @classmethod
    def reset(cls) -> None:
        """Clear all registered providers (for testing)."""
        cls._providers.clear()
