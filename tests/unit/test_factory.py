"""
Unit tests for LLMProviderFactory.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any

from llmblocks.blocks.llm_provider.factory import LLMProviderFactory
from llmblocks.blocks.llm_provider.base import BaseLLMProvider, LLMProviderConfig


class MockProvider(BaseLLMProvider):
    """Mock provider for testing factory."""
    
    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
    
    async def _initialize_clients(self) -> None:
        pass
    
    async def _close_client(self) -> None:
        pass
    
    async def _generate_impl(self, messages, **kwargs):
        from llmblocks.blocks.llm_provider.base import LLMResponse
        return LLMResponse(content="Mock response", metadata={})
    
    async def _generate_stream_impl(self, messages, **kwargs):
        from llmblocks.blocks.llm_provider.base import LLMStreamingResponse
        yield LLMStreamingResponse(content="Mock", metadata={})


class MockConfig(LLMProviderConfig):
    """Mock config for testing."""
    test_param: str = "default"


class TestLLMProviderFactory:
    """Test cases for LLMProviderFactory."""
    
    @pytest.fixture
    def factory(self):
        """Create a fresh factory instance."""
        return LLMProviderFactory()
    
    def test_factory_initialization(self, factory):
        """Test factory initialization."""
        assert isinstance(factory._provider_classes, dict)
        assert len(factory._provider_classes) >= 0  # May have built-in providers
    
    def test_register_provider(self, factory):
        """Test provider registration."""
        factory.register_provider("mock", MockProvider, MockConfig)
        
        assert "mock" in factory._provider_classes
        assert factory._provider_classes["mock"] == MockProvider
    
    def test_register_provider_duplicate(self, factory):
        """Test registering duplicate provider name."""
        factory.register_provider("mock", MockProvider, MockConfig)
        
        # Should not raise error, just overwrite
        factory.register_provider("mock", MockProvider, MockConfig)
        assert "mock" in factory._provider_classes
    
    def test_list_available_providers(self, factory):
        """Test listing available providers."""
        factory.register_provider("mock1", MockProvider, MockConfig)
        factory.register_provider("mock2", MockProvider, MockConfig)
        
        providers = factory.list_providers()
        assert "mock1" in providers
        assert "mock2" in providers
    
    async def test_create_provider_success(self, factory):
        """Test successful provider creation."""
        factory.register_provider("mock", MockProvider, MockConfig)
        
        config = {"test_param": "test_value"}
        provider = await factory.create_provider("mock", **config)
        
        assert isinstance(provider, MockProvider)
        assert provider._config.test_param == "test_value"
    
    async def test_create_provider_unknown(self, factory):
        """Test creating unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider type: unknown"):
            await factory.create_provider("unknown")
    
    async def test_create_provider_with_api_key(self, factory):
        """Test creating provider with API key."""
        factory.register_provider("mock", MockProvider, MockConfig)
        
        provider = await factory.create_provider(
            "mock",
            api_key="test-key",
            test_param="test_value"
        )
        
        assert isinstance(provider, MockProvider)
    
    @patch('llmblocks.blocks.llm_provider.factory.importlib.import_module')
    def test_register_builtin_providers_success(self, mock_import, factory):
        """Test successful built-in provider registration."""
        # Mock the module and class
        mock_module = MagicMock()
        mock_provider_class = MagicMock()
        mock_module.OpenAIProvider = mock_provider_class
        mock_import.return_value = mock_module
        
        # Mock the provider class to have a config_class
        mock_provider_class.config_class = MockConfig
        
        factory._register_builtin_providers()
        
        # Should attempt to import built-in providers
        assert mock_import.called
    
    @patch('llmblocks.blocks.llm_provider.factory.importlib.import_module')
    def test_register_builtin_providers_import_error(self, mock_import, factory):
        """Test built-in provider registration with import error."""
        mock_import.side_effect = ImportError("Module not found")
        
        # Should not raise error, just log warning
        factory._register_builtin_providers()
        
        # Factory should still be usable
        assert isinstance(factory._provider_classes, dict)
    
    def test_provider_availability(self, factory):
        """Test checking provider availability."""
        factory.register_provider("mock", MockProvider, MockConfig)
        
        assert factory.is_provider_available("mock")
        assert not factory.is_provider_available("unknown")
    
    async def test_create_provider_config_validation(self, factory):
        """Test provider creation with config validation."""
        factory.register_provider("mock", MockProvider, MockConfig)
        
        # Valid config
        provider = await factory.create_provider("mock", test_param="valid")
        assert provider._config.test_param == "valid"
    
    def test_factory_singleton_behavior(self):
        """Test that factory behaves consistently."""
        factory1 = LLMProviderFactory()
        factory2 = LLMProviderFactory()
        
        # They should be independent instances
        factory1.register_provider("test1", MockProvider, MockConfig)
        factory2.register_provider("test2", MockProvider, MockConfig)
        
        assert "test1" in factory1._provider_classes
        assert "test2" in factory2._provider_classes
        # But they shouldn't share state
        assert "test2" not in factory1._provider_classes
        assert "test1" not in factory2._provider_classes


class TestFactoryIntegration:
    """Integration tests for factory with real provider classes."""
    
    @pytest.fixture
    def factory(self):
        """Create factory with built-in providers."""
        factory = LLMProviderFactory()
        # Don't register built-ins to avoid import issues in tests
        return factory
    
    def test_factory_with_mock_builtin_providers(self, factory):
        """Test factory behavior with mock built-in providers."""
        # Manually register some providers to simulate built-ins
        factory.register_provider("openai", MockProvider, MockConfig)
        factory.register_provider("gemini", MockProvider, MockConfig)
        
        providers = factory.list_providers()
        assert "openai" in providers
        assert "gemini" in providers
    
    async def test_create_multiple_providers(self, factory):
        """Test creating multiple different providers."""
        factory.register_provider("provider1", MockProvider, MockConfig)
        factory.register_provider("provider2", MockProvider, MockConfig)
        
        p1 = await factory.create_provider("provider1", test_param="value1")
        p2 = await factory.create_provider("provider2", test_param="value2")
        
        assert p1._config.test_param == "value1"
        assert p2._config.test_param == "value2"
        assert p1.block_id != p2.block_id  # Different instances
    
    async def test_provider_lifecycle_through_factory(self, factory):
        """Test complete provider lifecycle through factory."""
        factory.register_provider("mock", MockProvider, MockConfig)
        
        # Create provider
        provider = await factory.create_provider("mock", test_param="test")
        
        # Initialize and use
        await provider.initialize()
        response = await provider.generate("Hello")
        
        assert response.content == "Mock response"
        
        # Cleanup
        await provider.close()


# Test the convenience function
class TestGetProvider:
    """Test the get_provider convenience function."""
    
    @patch('llmblocks.blocks.llm_provider.factory.LLMProviderFactory')
    async def test_get_provider_function(self, mock_factory_class):
        """Test the get_provider convenience function."""
        from llmblocks.blocks.llm_provider import get_provider
        
        # Mock the factory instance and its methods
        mock_factory = MagicMock()
        mock_provider = AsyncMock()
        mock_factory.create_provider.return_value = mock_provider
        mock_factory_class.return_value = mock_factory
        
        # Call get_provider
        result = await get_provider("test_provider", param="value")
        
        # Verify factory was called correctly
        mock_factory.create_provider.assert_called_once_with("test_provider", param="value")
        assert result == mock_provider
    
    @patch('llmblocks.blocks.llm_provider.factory.LLMProviderFactory')
    async def test_get_provider_with_api_key(self, mock_factory_class):
        """Test get_provider with API key."""
        from llmblocks.blocks.llm_provider import get_provider
        
        mock_factory = MagicMock()
        mock_provider = AsyncMock()
        mock_factory.create_provider.return_value = mock_provider
        mock_factory_class.return_value = mock_factory
        
        await get_provider("openai", api_key="test-key", model="gpt-4")
        
        mock_factory.create_provider.assert_called_once_with(
            "openai", 
            api_key="test-key", 
            model="gpt-4"
        )
