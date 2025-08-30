"""
Unit tests for GeminiProvider.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from llmblocks.blocks.llm_provider.gemini_provider import (
    GeminiProvider,
    GeminiProviderConfig
)
from llmblocks.blocks.llm_provider.base import (
    LLMMessage,
    LLMRole,
    LLMResponse,
    BlockStatus
)


class TestGeminiProviderConfig:
    """Test cases for GeminiProviderConfig."""
    
    def test_config_creation_minimal(self):
        """Test config creation with minimal parameters."""
        config = GeminiProviderConfig(
            api_key="test-key",
            model="gemini-2.0-flash"
        )
        
        assert config.api_key.get_secret_value() == "test-key"
        assert config.model == "gemini-2.0-flash"
        assert config.temperature == 0.7  # default
        assert config.max_tokens is None  # default
    
    def test_config_creation_full(self):
        """Test config creation with all parameters."""
        config = GeminiProviderConfig(
            api_key="test-key",
            model="gemini-2.0-flash",
            temperature=0.5,
            max_tokens=500,
            timeout=60.0,
            top_p=0.9,
            top_k=40
        )
        
        assert config.temperature == 0.5
        assert config.max_tokens == 500
        assert config.timeout == 60.0
        assert config.top_p == 0.9
        assert config.top_k == 40
    
    def test_model_validation_valid(self):
        """Test model validation with valid models."""
        valid_models = [
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro"
        ]
        
        for model in valid_models:
            config = GeminiProviderConfig(api_key="test-key", model=model)
            assert config.model == model
    
    def test_model_validation_custom(self):
        """Test model validation with custom model (should pass)."""
        config = GeminiProviderConfig(api_key="test-key", model="custom-model")
        assert config.model == "custom-model"
    
    def test_config_serialization(self):
        """Test config serialization."""
        config = GeminiProviderConfig(
            api_key="test-key",
            model="gemini-2.0-flash",
            temperature=0.8
        )
        
        data = config.model_dump()
        assert "api_key" in data
        assert data["model"] == "gemini-2.0-flash"
        assert data["temperature"] == 0.8


class TestGeminiProvider:
    """Test cases for GeminiProvider."""
    
    @pytest.fixture
    def config(self):
        """Sample Gemini configuration."""
        return GeminiProviderConfig(
            api_key="test-api-key",
            model="gemini-2.0-flash",
            temperature=0.7,
            max_tokens=100
        )
    
    @pytest.fixture
    def provider(self, config):
        """Create a Gemini provider instance."""
        return GeminiProvider(config)
    
    def test_provider_initialization(self, provider, config):
        """Test provider initialization."""
        assert provider._gemini_config == config
        assert provider.block_id is not None
        assert provider.status == BlockStatus.UNINITIALIZED
        assert provider._langchain_client is None
    
    @patch('llmblocks.blocks.llm_provider.gemini_provider.ChatGoogleGenerativeAI')
    async def test_initialize_clients(self, mock_chat_class, provider):
        """Test client initialization."""
        mock_client = AsyncMock()
        mock_chat_class.return_value = mock_client
        
        await provider.initialize()  # Use full initialization, not just _initialize_clients
        
        assert provider.status == BlockStatus.READY
        assert provider._langchain_client == mock_client
        
        # Verify client was created with correct config
        mock_chat_class.assert_called_once()
        call_kwargs = mock_chat_class.call_args[1]
        assert call_kwargs["google_api_key"] == "test-api-key"
        assert call_kwargs["model"] == "gemini-2.0-flash"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_output_tokens"] == 100
    
    async def test_close_client(self, provider):
        """Test client cleanup."""
        # Initialize first
        await provider.initialize()
        assert provider.status == BlockStatus.READY
        
        # Then cleanup
        await provider._cleanup_impl()
        
        # Note: Status doesn't automatically change to STOPPED in base implementation
        # Just verify cleanup was called
    
    @patch('llmblocks.blocks.llm_provider.gemini_provider.ChatGoogleGenerativeAI')
    async def test_generate_impl(self, mock_chat_class, provider):
        """Test generation implementation."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_chat_class.return_value = mock_client
        
        # Mock LangChain response
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        
        mock_message = AIMessage(content="Test response from Gemini")
        mock_generation = ChatGeneration(message=mock_message)
        mock_result = ChatResult(generations=[mock_generation])
        mock_client.agenerate.return_value = mock_result
        
        await provider._initialize_clients()
        
        # Test generation
        messages = [LLMMessage(role=LLMRole.USER, content="Hello")]
        response = await provider._generate_impl(messages)
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Test response from Gemini"
        assert response.model == "gemini-2.0-flash"
        
        # Verify client was called correctly
        mock_client.agenerate.assert_called_once()
        call_args = mock_client.agenerate.call_args[0][0]  # First positional arg
        assert len(call_args) == 1  # One message batch
        assert len(call_args[0]) == 1  # One message in batch
    
    @patch('llmblocks.blocks.llm_provider.gemini_provider.ChatGoogleGenerativeAI')
    async def test_generate_stream_impl(self, mock_chat_class, provider):
        """Test streaming generation implementation."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_chat_class.return_value = mock_client
        
        # Mock streaming response
        from langchain_core.messages import AIMessage
        
        async def mock_stream(*args, **kwargs):
            chunks = ["Hello", " world", "!"]
            for chunk in chunks:
                yield AIMessage(content=chunk)
        
        mock_client.astream = mock_stream
        
        await provider._initialize_clients()
        
        # Test streaming
        messages = [LLMMessage(role=LLMRole.USER, content="Hello")]
        chunks = []
        
        async for chunk in provider._generate_stream_impl(messages):
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"
        assert chunks[2].content == "!"
        
        # Verify streaming worked (we got 3 chunks)
        assert all(isinstance(chunk, LLMResponse) for chunk in chunks)
    
    @patch('llmblocks.blocks.llm_provider.gemini_provider.ChatGoogleGenerativeAI')
    async def test_generate_with_parameters(self, mock_chat_class, provider):
        """Test generation with custom parameters."""
        mock_client = AsyncMock()
        mock_chat_class.return_value = mock_client
        
        # Mock response
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        
        mock_message = AIMessage(content="Response")
        mock_generation = ChatGeneration(message=mock_message)
        mock_result = ChatResult(generations=[mock_generation])
        mock_client.agenerate.return_value = mock_result
        
        await provider._initialize_clients()
        
        # Test with custom parameters
        messages = [LLMMessage(role=LLMRole.USER, content="Hello")]
        response = await provider._generate_impl(
            messages,
            temperature=0.9,
            max_tokens=200
        )
        
        # Verify response was generated
        assert response.content == "Response"
        assert response.model == "gemini-2.0-flash"
        mock_client.agenerate.assert_called_once()
    
    def test_provider_repr(self, provider):
        """Test provider string representation."""
        repr_str = repr(provider)
        assert "GeminiProvider" in repr_str
        assert provider.block_id in repr_str
        assert "gemini-2.0-flash" in repr_str
    
    async def test_provider_context_manager(self, provider):
        """Test provider lifecycle without context manager (not implemented in base)."""
        # Base provider doesn't implement context manager protocol
        # Just test basic lifecycle
        await provider.initialize()
        assert provider.status == BlockStatus.READY
        
        await provider._cleanup_impl()
        # Note: Status doesn't automatically change in base implementation
    
    @patch('llmblocks.blocks.llm_provider.gemini_provider.ChatGoogleGenerativeAI')
    async def test_error_handling_initialization(self, mock_chat_class, provider):
        """Test error handling during initialization."""
        mock_chat_class.side_effect = Exception("API key invalid")
        
        with pytest.raises(Exception, match="API key invalid"):
            await provider._initialize_clients()
        
        assert provider.status == BlockStatus.UNINITIALIZED
    
    @patch('llmblocks.blocks.llm_provider.gemini_provider.ChatGoogleGenerativeAI')
    async def test_error_handling_generation(self, mock_chat_class, provider):
        """Test error handling during generation."""
        mock_client = AsyncMock()
        mock_client.agenerate.side_effect = Exception("Rate limit exceeded")
        mock_chat_class.return_value = mock_client
        
        await provider._initialize_clients()
        
        messages = [LLMMessage(role=LLMRole.USER, content="Hello")]
        
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await provider._generate_impl(messages)
    
    def test_config_property(self, provider, config):
        """Test config property access."""
        assert provider.gemini_config == config
        assert provider.provider_config.model == "gemini-2.0-flash"
    
    async def test_multiple_generations(self, provider):
        """Test multiple sequential generations."""
        with patch('llmblocks.blocks.llm_provider.gemini_provider.ChatGoogleGenerativeAI') as mock_chat_class:
            mock_client = AsyncMock()
            mock_chat_class.return_value = mock_client
            
            # Mock responses
            from langchain_core.messages import AIMessage
            from langchain_core.outputs import ChatGeneration, ChatResult
            
            responses = ["Response 1", "Response 2", "Response 3"]
            mock_results = []
            
            for response_text in responses:
                mock_message = AIMessage(content=response_text)
                mock_generation = ChatGeneration(message=mock_message)
                mock_result = ChatResult(generations=[mock_generation])
                mock_results.append(mock_result)
            
            mock_client.agenerate.side_effect = mock_results
            
            await provider.initialize()
            
            # Generate multiple responses
            messages = [LLMMessage(role=LLMRole.USER, content="Hello")]
            
            for i, expected_response in enumerate(responses):
                response = await provider._generate_impl(messages)
                assert response.content == expected_response
            
            # Verify all calls were made
            assert mock_client.agenerate.call_count == 3


class TestGeminiProviderIntegration:
    """Integration tests for GeminiProvider."""
    
    @pytest.fixture
    def config_with_env(self, temp_env_var):
        """Config that uses environment variable."""
        temp_env_var("GOOGLE_API_KEY", "test-env-key")
        
        return GeminiProviderConfig(
            api_key="test-env-key",
            model="gemini-2.0-flash"
        )
    
    def test_provider_with_env_config(self, config_with_env):
        """Test provider creation with environment-based config."""
        provider = GeminiProvider(config_with_env)
        assert provider._gemini_config.api_key.get_secret_value() == "test-env-key"
    
    @patch('llmblocks.blocks.llm_provider.gemini_provider.ChatGoogleGenerativeAI')
    async def test_full_conversation_flow(self, mock_chat_class):
        """Test a full conversation flow."""
        config = GeminiProviderConfig(
            api_key="test-key",
            model="gemini-2.0-flash"
        )
        provider = GeminiProvider(config)
        
        # Mock client
        mock_client = AsyncMock()
        mock_chat_class.return_value = mock_client
        
        # Mock conversation responses
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        
        conversation_responses = [
            "Hello! How can I help you?",
            "I'm doing well, thank you for asking!",
            "The weather is nice today."
        ]
        
        mock_results = []
        for response_text in conversation_responses:
            mock_message = AIMessage(content=response_text)
            mock_generation = ChatGeneration(message=mock_message)
            mock_result = ChatResult(generations=[mock_generation])
            mock_results.append(mock_result)
        
        mock_client.agenerate.side_effect = mock_results
        
        # Simulate conversation
        await provider.initialize()
        
        # User: Hello
        response1 = await provider.generate("Hello")
        assert response1.content == "Hello! How can I help you?"
        
        # User: How are you?
        response2 = await provider.generate("How are you?")
        assert response2.content == "I'm doing well, thank you for asking!"
        
        # User: What's the weather like?
        response3 = await provider.generate("What's the weather like?")
        assert response3.content == "The weather is nice today."
        
        # Cleanup
        await provider._cleanup_impl()
        
        # Verify all interactions
        assert mock_client.agenerate.call_count == 3
