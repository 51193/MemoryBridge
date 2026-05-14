"""Provider registry — maps model names to LLM providers."""

from ..exceptions import ProviderNotFoundError
from .base import AbstractLLMProvider


class ProviderRegistry:
    """Singleton registry for LLM providers.

    Providers are registered by model name and looked up at request time.
    """

    _providers: dict[str, AbstractLLMProvider] = {}

    @classmethod
    def register(cls, model: str, provider: AbstractLLMProvider) -> None:
        """Register a provider for a given model name."""
        cls._providers[model] = provider

    @classmethod
    def get(cls, model: str) -> AbstractLLMProvider:
        """Get the provider for a model name.

        Raises:
            ProviderNotFoundError: If no provider is registered for the model.
        """
        if model not in cls._providers:
            raise ProviderNotFoundError(f"No provider registered for model: {model}")
        return cls._providers[model]

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
