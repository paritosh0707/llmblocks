"""
Tests for LLM provider factory.
"""

import pytest
from unittest.mock import Mock, patch
import os

from llmblocks.core.llm_providers import (
    LLMProviderFactory,
    OpenAIProvider,
    GoogleProvider,
    HuggingFaceProvider,
    GroqProvider,
    AnthropicProvider,
    OllamaProvider
)


class TestLLMProviderFactory:
    """Test LLM provider factory functionality."""
    
    def test_list_providers(self):
        """Test listing available providers."""
        providers = LLMProviderFactory.list_providers()
        expected = ['openai', 'google', 'huggingface', 'groq', 'anthropic', 'ollama']
        
        assert all(provider in providers for provider in expected)
    
    def test_get_provider(self):
        """Test getting a specific provider."""
        provider = LLMProviderFactory.get_provider('openai')
        assert isinstance(provider, OpenAIProvider)
    
    def test_get_invalid_provider(self):
        """Test getting an invalid provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMProviderFactory.get_provider('invalid_provider')
    
    def test_register_provider(self):
        """Test registering a custom provider."""
        custom_provider = Mock()
        LLMProviderFactory.register_provider('custom', custom_provider)
        
        assert 'custom' in LLMProviderFactory.list_providers()
        assert LLMProviderFactory.get_provider('custom') == custom_provider
    
    def test_get_provider_info(self):
        """Test getting provider information."""
        info = LLMProviderFactory.get_provider_info('openai')
        
        assert 'name' in info
        assert 'required_env_vars' in info
        assert 'available' in info
        assert info['name'] == 'openai'


class TestOpenAIProvider:
    """Test OpenAI provider."""
    
    def test_validate_config(self):
        """Test OpenAI configuration validation."""
        provider = OpenAIProvider()
        
        # Valid config
        assert provider.validate_config({'model': 'gpt-4'})
        
        # Invalid config
        assert not provider.validate_config({})
    
    def test_get_required_env_vars(self):
        """Test OpenAI required environment variables."""
        provider = OpenAIProvider()
        env_vars = provider.get_required_env_vars()
        
        assert 'OPENAI_API_KEY' in env_vars
    
    @patch('os.getenv')
    def test_create_llm_with_env_key(self, mock_getenv):
        """Test creating OpenAI LLM with environment variable."""
        mock_getenv.return_value = 'test-key'
        provider = OpenAIProvider()
        
        config = {'model': 'gpt-4', 'temperature': 0.1}
        llm = provider.create_llm(config)
        
        assert llm.model_name == 'gpt-4'
        assert llm.temperature == 0.1
    
    def test_create_llm_with_config_key(self):
        """Test creating OpenAI LLM with config API key."""
        provider = OpenAIProvider()
        
        config = {
            'model': 'gpt-4',
            'temperature': 0.1,
            'api_key': 'config-key'
        }
        llm = provider.create_llm(config)
        
        assert llm.model_name == 'gpt-4'
        assert llm.temperature == 0.1


class TestGoogleProvider:
    """Test Google provider."""
    
    def test_validate_config(self):
        """Test Google configuration validation."""
        provider = GoogleProvider()
        
        # Valid config
        assert provider.validate_config({'model': 'gemini-pro'})
        
        # Invalid config
        assert not provider.validate_config({})
    
    def test_get_required_env_vars(self):
        """Test Google required environment variables."""
        provider = GoogleProvider()
        env_vars = provider.get_required_env_vars()
        
        assert 'GOOGLE_API_KEY' in env_vars


class TestHuggingFaceProvider:
    """Test Hugging Face provider."""
    
    def test_validate_config(self):
        """Test Hugging Face configuration validation."""
        provider = HuggingFaceProvider()
        
        # Valid config
        assert provider.validate_config({'model': 'test-model'})
        
        # Invalid config
        assert not provider.validate_config({})
    
    def test_get_required_env_vars(self):
        """Test Hugging Face required environment variables."""
        provider = HuggingFaceProvider()
        env_vars = provider.get_required_env_vars()
        
        assert 'HUGGINGFACE_API_KEY' in env_vars


class TestGroqProvider:
    """Test Groq provider."""
    
    def test_validate_config(self):
        """Test Groq configuration validation."""
        provider = GroqProvider()
        
        # Valid config
        assert provider.validate_config({'model': 'llama3-8b-8192'})
        
        # Invalid config
        assert not provider.validate_config({})
    
    def test_get_required_env_vars(self):
        """Test Groq required environment variables."""
        provider = GroqProvider()
        env_vars = provider.get_required_env_vars()
        
        assert 'GROQ_API_KEY' in env_vars


class TestAnthropicProvider:
    """Test Anthropic provider."""
    
    def test_validate_config(self):
        """Test Anthropic configuration validation."""
        provider = AnthropicProvider()
        
        # Valid config
        assert provider.validate_config({'model': 'claude-3-sonnet-20240229'})
        
        # Invalid config
        assert not provider.validate_config({})
    
    def test_get_required_env_vars(self):
        """Test Anthropic required environment variables."""
        provider = AnthropicProvider()
        env_vars = provider.get_required_env_vars()
        
        assert 'ANTHROPIC_API_KEY' in env_vars


class TestOllamaProvider:
    """Test Ollama provider."""
    
    def test_validate_config(self):
        """Test Ollama configuration validation."""
        provider = OllamaProvider()
        
        # Valid config
        assert provider.validate_config({'model': 'llama2'})
        
        # Invalid config
        assert not provider.validate_config({})
    
    def test_get_required_env_vars(self):
        """Test Ollama required environment variables."""
        provider = OllamaProvider()
        env_vars = provider.get_required_env_vars()
        
        # Ollama doesn't require API keys
        assert len(env_vars) == 0


class TestLLMProviderFactoryIntegration:
    """Integration tests for LLM provider factory."""
    
    @patch('os.getenv')
    def test_create_llm_with_openai(self, mock_getenv):
        """Test creating LLM with OpenAI provider."""
        mock_getenv.return_value = 'test-key'
        
        config = {
            'model': 'gpt-4',
            'temperature': 0.1,
            'max_tokens': 500
        }
        
        llm = LLMProviderFactory.create_llm('openai', config)
        
        assert llm.model_name == 'gpt-4'
        assert llm.temperature == 0.1
    
    def test_create_llm_missing_env_vars(self):
        """Test creating LLM with missing environment variables."""
        # Clear any existing environment variables
        with patch.dict(os.environ, {}, clear=True):
            config = {'model': 'gpt-4'}
            
            with pytest.raises(ValueError, match="Missing required environment variables"):
                LLMProviderFactory.create_llm('openai', config)
    
    def test_create_llm_invalid_config(self):
        """Test creating LLM with invalid configuration."""
        config = {}  # Missing required 'model' field
        
        with pytest.raises(ValueError, match="Invalid configuration"):
            LLMProviderFactory.create_llm('openai', config) 