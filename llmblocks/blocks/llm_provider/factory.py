"""
LLM Provider Factory Module

This module provides a factory class for creating LLM provider instances based on
configuration parameters. It uses a simple dictionary-based approach for provider management.
"""

from typing import Dict, Type, Any, List
from pydantic import ValidationError

from llmblocks.blocks.llm_provider.base import BaseLLMProvider
from llmblocks.blocks.llm_provider.openai_provider import OpenAIProvider


# Constant dictionary of available providers (normalized to lowercase)
AVAILABLE_PROVIDERS: Dict[str, Type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
}


class LLMFactory:
    """
    Factory class for creating LLM provider instances.
    
    This factory uses a simple dictionary-based approach for provider management
    and provides utility methods for provider registration and creation.
    """
    
    @classmethod
    def _determine_provider_name(cls, kwargs: Dict[str, Any]) -> str:
        """
        Determine the provider name from configuration parameters.
        
        Args:
            kwargs: Configuration parameters containing provider specification
            
        Returns:
            The provider name identifier (normalized to lowercase)
            
        Raises:
            ValueError: If provider is not specified
        """
        # Check for explicit provider specification
        provider = kwargs.get('provider_name')
        if provider:
            return str(provider).lower().strip()
        
        # Fallback: Check for provider key (alternative)
        provider_alt = kwargs.get('provider')
        if provider_alt:
            return str(provider_alt).lower().strip()
        
        # If no explicit provider specified, raise error
        raise ValueError(
            "Provider must be explicitly specified using 'provider_name' or 'provider' key in kwargs. "
            f"Available providers: {', '.join(sorted(AVAILABLE_PROVIDERS.keys()))}"
        )
    
    @classmethod
    def get_provider(cls, provider_name: str) -> Type[BaseLLMProvider]:
        """
        Get the provider class for the given provider name.
        
        Args:
            provider_name: The provider name identifier (case-insensitive)
            
        Returns:
            The provider class
            
        Raises:
            KeyError: If the provider is not registered
        """
        normalized_name = provider_name.lower().strip()
        if normalized_name not in AVAILABLE_PROVIDERS:
            available_providers = ', '.join(sorted(AVAILABLE_PROVIDERS.keys()))
            raise KeyError(
                f"Provider '{provider_name}' is not registered. "
                f"Available providers: {available_providers}"
            )
        
        return AVAILABLE_PROVIDERS[normalized_name]
    
    @classmethod
    def list_available_providers(cls) -> List[str]:
        """
        Get a list of all registered provider names.
        
        Returns:
            List of available provider names (sorted alphabetically)
        """
        return sorted(AVAILABLE_PROVIDERS.keys())
    
    @classmethod
    def is_provider_available(cls, provider_name: str) -> bool:
        """
        Check if a provider is available.
        
        Args:
            provider_name: The provider name to check (case-insensitive)
            
        Returns:
            True if the provider is available, False otherwise
        """
        normalized_name = provider_name.lower().strip()
        return normalized_name in AVAILABLE_PROVIDERS
    
    @classmethod
    def create_provider(cls, **kwargs) -> BaseLLMProvider:
        """
        Create and return an LLM provider instance based on configuration.
        
        Args:
            **kwargs: Configuration parameters for the LLM provider
                - provider_name (str): The provider name (required)
                - api_key (str): API key for the provider (required)
                - model_name (str): Model name to use (optional, provider-specific default)
                - Additional provider-specific parameters (temperature, max_tokens, etc.)
        
        Returns:
            An instance of the appropriate LLM provider
            
        Raises:
            ValueError: If provider is not specified or provider creation fails
            KeyError: If the specified provider is not registered
            ValidationError: If provider-specific validation fails (chained)
        """
        # Determine the provider name
        provider_name = cls._determine_provider_name(kwargs)
        
        # Get the provider class
        provider_class = cls.get_provider(provider_name)
        
        # Create and return the provider instance
        try:
            return provider_class(**kwargs)
        except ValidationError as e:
            # Chain the original validation error for better debugging
            raise ValueError(f"Invalid configuration for {provider_name} provider: {e}") from e
        except TypeError as e:
            # Handle type errors (e.g., missing required arguments)
            raise ValueError(f"Invalid arguments for {provider_name} provider: {e}") from e
        except Exception as e:
            # Handle other unexpected errors
            raise ValueError(f"Failed to create {provider_name} provider: {e}") from e
    
    @classmethod
    def register_provider(cls, name: str, provider_cls: Type[BaseLLMProvider]) -> None:
        """
        Register a new provider with the factory.
        
        Args:
            name: The provider name (will be normalized to lowercase)
            provider_cls: The provider class that inherits from BaseLLMProvider
            
        Raises:
            ValueError: If name is empty or provider_cls is invalid
        """
        if not name or not name.strip():
            raise ValueError("Provider name cannot be empty")
        
        if not issubclass(provider_cls, BaseLLMProvider):
            raise ValueError(f"Provider class must inherit from BaseLLMProvider, got {provider_cls}")
        
        normalized_name = name.lower().strip()
        AVAILABLE_PROVIDERS[normalized_name] = provider_cls
    



if __name__ == "__main__":
    # Run examples from the separate examples module
    from examples.llm_provider import run_examples
    run_examples()
