"""
LLM Provider Factory for LLMBlocks.

This module provides a factory for creating and managing LLM provider instances
with automatic discovery, configuration validation, and fallback handling.
"""

import asyncio
from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime, UTC
import importlib

from .base import BaseLLMProvider, LLMProviderConfig
from ...core.registry import get_registry, BlockType
from ...utils.logging import get_logger
from ...utils.exceptions import (
    LLMProviderError,
    BlockNotFoundError,
    ConfigurationError,
    ErrorCodes
)


class LLMProviderFactory:
    """
    Factory for creating and managing LLM provider instances.
    
    This factory provides:
    - Automatic provider discovery and registration
    - Configuration validation and merging
    - Provider fallback and load balancing
    - Connection pooling and reuse
    - Health monitoring and failover
    """
    
    def __init__(self):
        """Initialize the LLM provider factory."""
        self.logger = get_logger("LLMProviderFactory")
        self.registry = get_registry()
        
        # Provider instances cache
        self._provider_instances: Dict[str, BaseLLMProvider] = {}
        self._provider_classes: Dict[str, Type[BaseLLMProvider]] = {}
        
        # Health monitoring
        self._provider_health: Dict[str, Dict[str, Any]] = {}
        self._last_health_check = datetime.now(UTC)
        
        # Auto-register built-in providers
        self._register_builtin_providers()
        
        self.logger.info("LLM provider factory initialized")
    
    def _register_builtin_providers(self) -> None:
        """Register built-in LLM providers."""
        builtin_providers = [
            ("openai", "OpenAIProvider", "llmblocks.blocks.llm_provider.openai_provider"),
            ("gemini", "GeminiProvider", "llmblocks.blocks.llm_provider.gemini_provider"),
            ("anthropic", "AnthropicProvider", "llmblocks.blocks.llm_provider.anthropic_provider"),
        ]
        
        for provider_name, class_name, module_path in builtin_providers:
            try:
                # Import the module
                module = importlib.import_module(module_path)
                provider_class = getattr(module, class_name)
                
                # Register the provider class
                self._provider_classes[provider_name] = provider_class
                
                # Register in the global registry
                self.registry.register_block(
                    block_class=provider_class,
                    name=provider_name,
                    description=f"{provider_name.title()} LLM Provider",
                    version="1.0.0",
                    author="LLMBlocks Team",
                    tags=["llm", "provider", provider_name],
                    override=True
                )
                
                self.logger.debug(f"Registered built-in provider: {provider_name}")
                
            except ImportError as e:
                self.logger.warning(
                    f"Failed to import built-in provider {provider_name}: {e}"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to register built-in provider {provider_name}: {e}"
                )
    
    def register_provider(
        self,
        provider_name: str,
        provider_class: Type[BaseLLMProvider],
        override: bool = False
    ) -> bool:
        """
        Register a custom LLM provider.
        
        Args:
            provider_name: Name of the provider
            provider_class: Provider class
            override: Whether to override existing provider
            
        Returns:
            True if registered successfully, False otherwise
        """
        try:
            # Validate provider class
            if not issubclass(provider_class, BaseLLMProvider):
                raise ValueError(f"Provider class must inherit from BaseLLMProvider")
            
            # Check if already registered
            if provider_name in self._provider_classes and not override:
                self.logger.warning(f"Provider {provider_name} already registered")
                return False
            
            # Register the provider
            self._provider_classes[provider_name] = provider_class
            
            # Register in the global registry
            self.registry.register_block(
                block_class=provider_class,
                name=provider_name,
                description=f"Custom {provider_name} LLM Provider",
                version="1.0.0",
                tags=["llm", "provider", "custom", provider_name],
                override=override
            )
            
            self.logger.info(f"Registered custom provider: {provider_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register provider {provider_name}: {e}")
            return False
    
    def list_providers(self) -> List[str]:
        """
        List all available providers.
        
        Returns:
            List of provider names
        """
        return list(self._provider_classes.keys())
    
    def is_provider_available(self, provider_name: str) -> bool:
        """
        Check if a provider is available.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            True if provider is available, False otherwise
        """
        return provider_name in self._provider_classes
    
    async def create_provider(
        self,
        provider_name: str,
        config: Optional[Union[Dict[str, Any], LLMProviderConfig]] = None,
        **kwargs
    ) -> BaseLLMProvider:
        """
        Create a new LLM provider instance.
        
        Args:
            provider_name: Name of the provider
            config: Provider configuration
            **kwargs: Additional configuration parameters
            
        Returns:
            LLM provider instance
            
        Raises:
            LLMProviderError: If provider creation fails
        """
        if not self.is_provider_available(provider_name):
            available_providers = ", ".join(self.list_providers())
            raise LLMProviderError(
                f"Provider '{provider_name}' not available. "
                f"Available providers: {available_providers}",
                error_code=ErrorCodes.LLM_CONNECTION_FAILED
            )
        
        try:
            # Get provider class
            provider_class = self._provider_classes[provider_name]
            
            # Prepare configuration
            if config is None:
                config = {}
            elif isinstance(config, LLMProviderConfig):
                config = config.model_dump()
            
            # Add provider name to config
            config["provider_name"] = provider_name
            config.update(kwargs)
            
            # Create provider instance
            provider = provider_class(config)
            
            # Initialize the provider
            await provider.initialize()
            
            self.logger.info(
                f"Created provider instance: {provider_name}",
                provider_id=provider.block_id
            )
            
            return provider
            
        except Exception as e:
            self.logger.error(f"Failed to create provider {provider_name}: {e}")
            raise LLMProviderError(
                f"Failed to create provider {provider_name}: {e}",
                error_code=ErrorCodes.LLM_CONNECTION_FAILED
            ) from e
    
    async def get_or_create_provider(
        self,
        provider_name: str,
        config: Optional[Union[Dict[str, Any], LLMProviderConfig]] = None,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> BaseLLMProvider:
        """
        Get an existing provider instance or create a new one.
        
        Args:
            provider_name: Name of the provider
            config: Provider configuration
            cache_key: Cache key for the instance
            **kwargs: Additional configuration parameters
            
        Returns:
            LLM provider instance
        """
        # Generate cache key if not provided
        if cache_key is None:
            config_hash = hash(str(sorted((config or {}).items())))
            cache_key = f"{provider_name}_{config_hash}"
        
        # Check if instance exists and is healthy
        if cache_key in self._provider_instances:
            provider = self._provider_instances[cache_key]
            if provider.is_ready:
                return provider
            else:
                # Remove unhealthy instance
                del self._provider_instances[cache_key]
        
        # Create new instance
        provider = await self.create_provider(provider_name, config, **kwargs)
        
        # Cache the instance
        self._provider_instances[cache_key] = provider
        
        return provider
    
    async def create_provider_with_fallback(
        self,
        primary_provider: str,
        fallback_providers: List[str],
        config: Optional[Union[Dict[str, Any], LLMProviderConfig]] = None,
        **kwargs
    ) -> BaseLLMProvider:
        """
        Create a provider with fallback options.
        
        Args:
            primary_provider: Primary provider name
            fallback_providers: List of fallback provider names
            config: Provider configuration
            **kwargs: Additional configuration parameters
            
        Returns:
            LLM provider instance
            
        Raises:
            LLMProviderError: If all providers fail
        """
        providers_to_try = [primary_provider] + fallback_providers
        last_error = None
        
        for provider_name in providers_to_try:
            try:
                provider = await self.create_provider(provider_name, config, **kwargs)
                
                if provider_name != primary_provider:
                    self.logger.warning(
                        f"Using fallback provider: {provider_name} "
                        f"(primary {primary_provider} failed)"
                    )
                
                return provider
                
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"Provider {provider_name} failed, trying next: {e}"
                )
                continue
        
        # All providers failed
        raise LLMProviderError(
            f"All providers failed. Last error: {last_error}",
            error_code=ErrorCodes.LLM_CONNECTION_FAILED
        ) from last_error
    
    async def health_check_providers(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform health check on all cached provider instances.
        
        Returns:
            Health status for all providers
        """
        health_results = {}
        
        for cache_key, provider in self._provider_instances.items():
            try:
                health = await provider.health_check()
                health_results[cache_key] = health
                
                # Update health cache
                self._provider_health[cache_key] = {
                    "is_healthy": health.get("is_healthy", True),
                    "last_check": datetime.now(UTC),
                    "error": None
                }
                
            except Exception as e:
                health_results[cache_key] = {
                    "is_healthy": False,
                    "error": str(e),
                    "provider_name": getattr(provider, "provider_name", "unknown")
                }
                
                # Update health cache
                self._provider_health[cache_key] = {
                    "is_healthy": False,
                    "last_check": datetime.now(UTC),
                    "error": str(e)
                }
        
        self._last_health_check = datetime.now(UTC)
        return health_results
    
    async def cleanup_unhealthy_providers(self) -> int:
        """
        Clean up unhealthy provider instances.
        
        Returns:
            Number of providers cleaned up
        """
        cleaned_up = 0
        unhealthy_keys = []
        
        for cache_key, provider in self._provider_instances.items():
            try:
                if not provider.is_ready:
                    unhealthy_keys.append(cache_key)
                    continue
                
                # Check if provider is responsive
                health = await provider.health_check()
                if not health.get("is_healthy", True):
                    unhealthy_keys.append(cache_key)
                    
            except Exception:
                unhealthy_keys.append(cache_key)
        
        # Clean up unhealthy providers
        for cache_key in unhealthy_keys:
            try:
                provider = self._provider_instances[cache_key]
                await provider.cleanup()
                del self._provider_instances[cache_key]
                
                if cache_key in self._provider_health:
                    del self._provider_health[cache_key]
                
                cleaned_up += 1
                
            except Exception as e:
                self.logger.error(f"Error cleaning up provider {cache_key}: {e}")
        
        if cleaned_up > 0:
            self.logger.info(f"Cleaned up {cleaned_up} unhealthy providers")
        
        return cleaned_up
    
    async def cleanup_all_providers(self) -> None:
        """Clean up all cached provider instances."""
        for cache_key, provider in list(self._provider_instances.items()):
            try:
                await provider.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up provider {cache_key}: {e}")
        
        self._provider_instances.clear()
        self._provider_health.clear()
        
        self.logger.info("All providers cleaned up")
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the provider factory.
        
        Returns:
            Factory statistics
        """
        return {
            "total_registered_providers": len(self._provider_classes),
            "cached_instances": len(self._provider_instances),
            "available_providers": self.list_providers(),
            "last_health_check": self._last_health_check.isoformat(),
            "provider_health": self._provider_health.copy()
        }


# Global factory instance
_global_factory: Optional[LLMProviderFactory] = None


def get_factory() -> LLMProviderFactory:
    """Get the global LLM provider factory instance."""
    global _global_factory
    
    if _global_factory is None:
        _global_factory = LLMProviderFactory()
    
    return _global_factory


async def get_provider(
    provider_name: str,
    config: Optional[Union[Dict[str, Any], LLMProviderConfig]] = None,
    **kwargs
) -> BaseLLMProvider:
    """
    Get an LLM provider instance using the global factory.
    
    Args:
        provider_name: Name of the provider
        config: Provider configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        LLM provider instance
    """
    factory = get_factory()
    return await factory.create_provider(provider_name, config, **kwargs)


async def create_provider_with_fallback(
    primary_provider: str,
    fallback_providers: List[str],
    config: Optional[Union[Dict[str, Any], LLMProviderConfig]] = None,
    **kwargs
) -> BaseLLMProvider:
    """
    Create a provider with fallback options using the global factory.
    
    Args:
        primary_provider: Primary provider name
        fallback_providers: List of fallback provider names
        config: Provider configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        LLM provider instance
    """
    factory = get_factory()
    return await factory.create_provider_with_fallback(
        primary_provider, fallback_providers, config, **kwargs
    )


def list_available_providers() -> List[str]:
    """List all available providers using the global factory."""
    factory = get_factory()
    return factory.list_providers()


def register_custom_provider(
    provider_name: str,
    provider_class: Type[BaseLLMProvider],
    override: bool = False
) -> bool:
    """
    Register a custom provider using the global factory.
    
    Args:
        provider_name: Name of the provider
        provider_class: Provider class
        override: Whether to override existing provider
        
    Returns:
        True if registered successfully, False otherwise
    """
    factory = get_factory()
    return factory.register_provider(provider_name, provider_class, override)
