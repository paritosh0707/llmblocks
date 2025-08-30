"""
LLM Provider blocks for LLMBlocks.

This module provides a unified interface for different LLM providers
with async support, connection pooling, and comprehensive error handling.
"""

from .base import BaseLLMProvider, LLMProviderConfig, LLMResponse, LLMMessage, LLMRole
from .factory import (
    LLMProviderFactory, 
    get_provider, 
    create_provider_with_fallback,
    list_available_providers,
    register_custom_provider
)
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider
from .anthropic_provider import AnthropicProvider

# Memory-enhanced providers
from .memory_enhanced import (
    MemoryEnhancedLLMProvider,
    get_stateful_ai,
    get_persistent_ai,
    create_stateful_ai,
    create_persistent_ai
)

__all__ = [
    # Base classes
    "BaseLLMProvider",
    "LLMProviderConfig", 
    "LLMResponse",
    "LLMMessage",
    "LLMRole",
    
    # Factory
    "LLMProviderFactory",
    "get_provider",
    "create_provider_with_fallback",
    "list_available_providers",
    "register_custom_provider",
    
    # Providers
    "OpenAIProvider",
    "GeminiProvider", 
    "AnthropicProvider",
    
    # Memory-enhanced providers
    "MemoryEnhancedLLMProvider",
    "get_stateful_ai",
    "get_persistent_ai",
    "create_stateful_ai",
    "create_persistent_ai",
]
