"""
Base LLM Provider Module

This module defines the abstract base class for all LLM providers.
"""

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All LLM providers must inherit from this class and implement
    the required methods and attributes.
    """
    
    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the LLM provider with configuration parameters."""
        pass
    
    @property
    @abstractmethod
    def PROVIDER_NAME(self) -> str:
        """Return the provider name identifier."""
        pass
    
    @abstractmethod
    def get_llm(self):
        """Get the underlying LLM instance."""
        pass 