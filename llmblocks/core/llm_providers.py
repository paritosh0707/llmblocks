"""
LLM Provider Factory for LLMBlocks.

This module provides a unified interface for different LLM providers
including OpenAI, Google, Hugging Face, Groq, and others.
"""

import os
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import logging

from langchain.llms.base import LLM
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel

# Import providers conditionally to avoid import errors
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from langchain_community.llms import HuggingFaceHub
    from langchain_community.chat_models import ChatHuggingFace
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from langchain_community.chat_models import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def create_llm(self, config: Dict[str, Any]) -> Union[LLM, BaseChatModel]:
        """Create an LLM instance with the given configuration."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate the provider-specific configuration."""
        pass
    
    @abstractmethod
    def get_required_env_vars(self) -> list[str]:
        """Get list of required environment variables for this provider."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    def create_llm(self, config: Dict[str, Any]) -> ChatOpenAI:
        """Create OpenAI ChatOpenAI instance."""
        return ChatOpenAI(
            model_name=config.get('model', 'gpt-3.5-turbo'),
            temperature=config.get('temperature', 0.0),
            max_tokens=config.get('max_tokens', 1000),
            streaming=config.get('streaming', False),
            openai_api_key=config.get('api_key') or os.getenv('OPENAI_API_KEY')
        )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate OpenAI configuration."""
        required_fields = ['model']
        return all(field in config for field in required_fields)
    
    def get_required_env_vars(self) -> list[str]:
        """Get required environment variables."""
        return ['OPENAI_API_KEY']


class GoogleProvider(LLMProvider):
    """Google Gemini provider implementation."""
    
    def create_llm(self, config: Dict[str, Any]) -> Union[LLM, BaseChatModel]:
        """Create Google ChatGoogleGenerativeAI instance."""
        if not GOOGLE_AVAILABLE:
            raise ImportError("Google provider not available. Install with: pip install langchain-google-genai")
        
        return ChatGoogleGenerativeAI(
            model=config.get('model', 'gemini-pro'),
            temperature=config.get('temperature', 0.0),
            max_output_tokens=config.get('max_tokens', 1000),
            google_api_key=config.get('api_key') or os.getenv('GOOGLE_API_KEY')
        )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Google configuration."""
        if not GOOGLE_AVAILABLE:
            return False
        required_fields = ['model']
        return all(field in config for field in required_fields)
    
    def get_required_env_vars(self) -> list[str]:
        """Get required environment variables."""
        return ['GOOGLE_API_KEY']


class HuggingFaceProvider(LLMProvider):
    """Hugging Face provider implementation."""
    
    def create_llm(self, config: Dict[str, Any]) -> Union[ChatHuggingFace, HuggingFaceHub]:
        """Create Hugging Face LLM instance."""
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("Hugging Face provider not available. Install with: pip install langchain-community")
        
        model_id = config.get('model')
        task = config.get('task', 'text-generation')
        
        if task == 'text-generation':
            return HuggingFaceHub(
                repo_id=model_id,
                task=task,
                model_kwargs={
                    'temperature': config.get('temperature', 0.0),
                    'max_length': config.get('max_tokens', 1000),
                },
                huggingfacehub_api_token=config.get('api_key') or os.getenv('HUGGINGFACE_API_KEY')
            )
        else:
            return ChatHuggingFace(
                model_id=model_id,
                task=task,
                model_kwargs={
                    'temperature': config.get('temperature', 0.0),
                    'max_length': config.get('max_tokens', 1000),
                },
                huggingfacehub_api_token=config.get('api_key') or os.getenv('HUGGINGFACE_API_KEY')
            )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Hugging Face configuration."""
        if not HUGGINGFACE_AVAILABLE:
            return False
        required_fields = ['model']
        return all(field in config for field in required_fields)
    
    def get_required_env_vars(self) -> list[str]:
        """Get required environment variables."""
        return ['HUGGINGFACE_API_KEY']


class GroqProvider(LLMProvider):
    """Groq provider implementation."""
    
    def create_llm(self, config: Dict[str, Any]) -> Union[LLM, BaseChatModel]:
        """Create Groq ChatGroq instance."""
        if not GROQ_AVAILABLE:
            raise ImportError("Groq provider not available. Install with: pip install langchain-groq")
        
        return ChatGroq(
            model_name=config.get('model', 'llama3-8b-8192'),
            temperature=config.get('temperature', 0.0),
            max_tokens=config.get('max_tokens', 1000),
            groq_api_key=config.get('api_key') or os.getenv('GROQ_API_KEY')
        )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Groq configuration."""
        if not GROQ_AVAILABLE:
            return False
        required_fields = ['model']
        return all(field in config for field in required_fields)
    
    def get_required_env_vars(self) -> list[str]:
        """Get required environment variables."""
        return ['GROQ_API_KEY']


# class AnthropicProvider(LLMProvider):
#     """Anthropic provider implementation."""
    
#     def create_llm(self, config: Dict[str, Any]) -> ChatAnthropic:
#         """Create Anthropic ChatAnthropic instance."""
#         if not ANTHROPIC_AVAILABLE:
#             raise ImportError("Anthropic provider not available. Install with: pip install langchain-anthropic")
        
#         return ChatAnthropic(
#             model=config.get('model', 'claude-3-sonnet-20240229'),
#             temperature=config.get('temperature', 0.0),
#             max_tokens=config.get('max_tokens', 1000),
#             anthropic_api_key=config.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
#         )
    
#     def validate_config(self, config: Dict[str, Any]) -> bool:
#         """Validate Anthropic configuration."""
#         if not ANTHROPIC_AVAILABLE:
#             return False
#         required_fields = ['model']
#         return all(field in config for field in required_fields)
    
#     def get_required_env_vars(self) -> list[str]:
#         """Get required environment variables."""
#         return ['ANTHROPIC_API_KEY']


class OllamaProvider(LLMProvider):
    """Ollama provider implementation."""
    
    def create_llm(self, config: Dict[str, Any]) -> ChatOllama:
        """Create Ollama ChatOllama instance."""
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama provider not available. Install with: pip install langchain-community")
        
        return ChatOllama(
            model=config.get('model', 'llama2'),
            temperature=config.get('temperature', 0.0),
            base_url=config.get('base_url', 'http://localhost:11434')
        )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Ollama configuration."""
        if not OLLAMA_AVAILABLE:
            return False
        required_fields = ['model']
        return all(field in config for field in required_fields)
    
    def get_required_env_vars(self) -> list[str]:
        """Get required environment variables."""
        return []  # Ollama doesn't require API keys


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    _providers: Dict[str, LLMProvider] = {
        'openai': OpenAIProvider(),
        'google': GoogleProvider(),
        'huggingface': HuggingFaceProvider(),
        'groq': GroqProvider(),
        # 'anthropic': AnthropicProvider(),
        'ollama': OllamaProvider(),
    }
    
    @classmethod
    def register_provider(cls, name: str, provider: LLMProvider) -> None:
        """Register a new provider."""
        cls._providers[name] = provider
    
    @classmethod
    def get_provider(cls, name: str) -> LLMProvider:
        """Get a provider by name."""
        if name not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(f"Unknown provider '{name}'. Available providers: {available}")
        return cls._providers[name]
    
    @classmethod
    def create_llm(cls, provider_name: str, config: Dict[str, Any]) -> Union[LLM, BaseChatModel]:
        """Create an LLM instance using the specified provider."""
        provider = cls.get_provider(provider_name)
        
        # Validate configuration
        if not provider.validate_config(config):
            raise ValueError(f"Invalid configuration for provider '{provider_name}'")
        
        # Check environment variables
        missing_vars = []
        for env_var in provider.get_required_env_vars():
            if not os.getenv(env_var) and not config.get('api_key'):
                missing_vars.append(env_var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables for {provider_name}: {missing_vars}")
        
        return provider.create_llm(config)
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all available providers."""
        return list(cls._providers.keys())
    
    @classmethod
    def get_provider_info(cls, name: str) -> Dict[str, Any]:
        """Get information about a provider."""
        provider = cls.get_provider(name)
        return {
            'name': name,
            'required_env_vars': provider.get_required_env_vars(),
            'available': provider.validate_config({'model': 'test'})  # Basic availability check
        } 