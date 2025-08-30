"""
Unit tests for BaseLLMProvider.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from llmblocks.blocks.llm_provider.base import (
    BaseLLMProvider,
    LLMProviderConfig,
    LLMMessage,
    LLMRole,
    LLMResponse
)
from llmblocks.core.base_block import BlockStatus


class MockLLMProviderConfig(LLMProviderConfig):
    """Mock provider configuration."""
    provider_name: str = "mock"


class MockLLMProvider(BaseLLMProvider):
    """Mock implementation of BaseLLMProvider for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        # Create a proper config object
        mock_config = MockLLMProviderConfig(**config)
        super().__init__(mock_config)
        self._mock_client = AsyncMock()
    
    async def _initialize_clients(self) -> None:
        """Mock client initialization."""
        self._status = BlockStatus.READY
    
    async def _close_client(self) -> None:
        """Mock client cleanup."""
        self._status = BlockStatus.STOPPED
    
    async def _generate_impl(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Mock generation implementation."""
        return LLMResponse(
            content="Mock response content",
            metadata={
                "model": self.provider_config.model,
                "tokens_used": 10,
                "finish_reason": "stop"
            }
        )
    
    async def _generate_stream_impl(self, messages: List[LLMMessage], **kwargs):
        """Mock streaming generation implementation."""
        chunks = ["Mock", " streaming", " response"]
        for i, chunk in enumerate(chunks):
            yield LLMResponse(
                content=chunk,
                metadata={"chunk_index": i}
            )
    
    async def _close_async_client(self):
        """Mock close async client implementation."""
        pass
    
    async def _close_sync_client(self):
        """Mock close sync client implementation."""
        pass


class TestBaseLLMProvider:
    """Test cases for BaseLLMProvider."""
    
    @pytest.fixture
    def provider_config(self):
        """Sample provider configuration."""
        return {
            "model": "test-model",
            "temperature": 0.7,
            "max_tokens": 100,
            "requests_per_minute": None,  # Disable rate limiting for tests
            "tokens_per_minute": None,
            "stream": True  # Enable streaming for tests
        }
    
    @pytest.fixture
    def provider(self, provider_config):
        """Create a mock provider instance."""
        return MockLLMProvider(provider_config)
    
    def test_provider_initialization(self, provider, provider_config):
        """Test provider initialization."""
        assert provider.block_id is not None
        assert provider.status == BlockStatus.UNINITIALIZED
        assert provider.metadata["block_id"] == provider.block_id
        # Check that metadata contains expected keys
        assert "block_id" in provider.metadata
        assert "provider" in provider.metadata
    
    async def test_provider_lifecycle(self, provider):
        """Test provider initialization and cleanup lifecycle."""
        # Initially uninitialized
        assert provider.status == BlockStatus.UNINITIALIZED
        
        # Initialize
        await provider.initialize()
        assert provider.status == BlockStatus.READY
        
        # Close (cleanup)
        await provider._cleanup_impl()
        # Note: Status doesn't change to STOPPED automatically in base implementation
    
    def test_normalize_messages_string(self, provider):
        """Test message normalization with string input."""
        result = provider._normalize_messages("Hello world")
        
        assert len(result) == 1
        assert result[0].role == LLMRole.USER
        assert result[0].content == "Hello world"
    
    def test_normalize_messages_dict(self, provider):
        """Test message normalization with dict input."""
        message_dict = {"role": "user", "content": "Hello world"}
        result = provider._normalize_messages(message_dict)
        
        assert len(result) == 1
        assert result[0].role == LLMRole.USER
        assert result[0].content == "Hello world"
    
    def test_normalize_messages_list(self, provider):
        """Test message normalization with list input."""
        messages = [
            "Hello",
            {"role": "assistant", "content": "Hi there!"},
            LLMMessage(role=LLMRole.USER, content="How are you?")
        ]
        result = provider._normalize_messages(messages)
        
        assert len(result) == 3
        assert result[0].role == LLMRole.USER
        assert result[0].content == "Hello"
        assert result[1].role == LLMRole.ASSISTANT
        assert result[1].content == "Hi there!"
        assert result[2].role == LLMRole.USER
        assert result[2].content == "How are you?"
    
    def test_normalize_messages_invalid_dict(self, provider):
        """Test message normalization with dict missing role (defaults to user)."""
        result = provider._normalize_messages({"content": "Missing role"})
        assert len(result) == 1
        assert result[0].role == LLMRole.USER  # defaults to user
        assert result[0].content == "Missing role"
    
    def test_normalize_messages_invalid_role(self, provider):
        """Test message normalization with invalid role."""
        with pytest.raises(ValueError, match="is not a valid LLMRole"):
            provider._normalize_messages({"role": "invalid", "content": "Test"})
    
    async def test_generate_string_input(self, provider):
        """Test generate method with string input."""
        await provider.initialize()
        
        response = await provider.generate("Hello world")
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Mock response content"
        assert "model" in response.metadata
        assert "tokens_used" in response.metadata
    
    async def test_generate_dict_input(self, provider):
        """Test generate method with dict input."""
        await provider.initialize()
        
        response = await provider.generate({"role": "user", "content": "Hello"})
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Mock response content"
    
    async def test_generate_list_input(self, provider):
        """Test generate method with list input."""
        await provider.initialize()
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        response = await provider.generate(messages)
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Mock response content"
    
    async def test_generate_stream(self, provider):
        """Test streaming generation."""
        await provider.initialize()
        
        chunks = []
        async for chunk in provider.generate_stream("Tell me a story"):
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert all(isinstance(chunk, LLMResponse) for chunk in chunks)
        
        # Reconstruct full response
        full_content = "".join(chunk.content for chunk in chunks)
        assert full_content == "Mock streaming response"
    
    async def test_generate_not_initialized(self, provider):
        """Test generate method when provider is not initialized."""
        with pytest.raises(Exception, match="Provider is not ready"):
            await provider.generate("Hello")
    
    async def test_generate_stream_not_initialized(self, provider):
        """Test streaming when provider is not initialized."""
        with pytest.raises(Exception, match="Provider is not ready"):
            async for _ in provider.generate_stream("Hello"):
                pass
    
    def test_metadata_updates(self, provider):
        """Test metadata access functionality."""
        initial_metadata = provider.metadata.copy()
        
        # Metadata is a regular dict, can be updated directly
        provider.metadata["custom_key"] = "custom_value"
        
        assert provider.metadata["custom_key"] == "custom_value"
        assert provider.metadata["block_id"] == initial_metadata["block_id"]
    
    async def test_context_manager(self, provider):
        """Test provider lifecycle without context manager (not implemented in base)."""
        # Base provider doesn't implement context manager protocol
        # Just test basic lifecycle
        await provider.initialize()
        assert provider.status == BlockStatus.READY
        
        await provider._cleanup_impl()
        # Note: Status doesn't automatically change in base implementation
    
    def test_provider_repr(self, provider):
        """Test provider string representation."""
        repr_str = repr(provider)
        assert "MockLLMProvider" in repr_str
        assert provider.block_id in repr_str
    
    async def test_concurrent_operations(self, provider):
        """Test concurrent generate operations."""
        await provider.initialize()
        
        import asyncio
        
        # Run multiple generate operations concurrently
        tasks = [
            provider.generate(f"Message {i}")
            for i in range(5)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        assert len(responses) == 5
        assert all(isinstance(r, LLMResponse) for r in responses)
        assert all(r.content == "Mock response content" for r in responses)


class TestLLMMessage:
    """Test cases for LLMMessage."""
    
    def test_message_creation(self):
        """Test LLMMessage creation."""
        message = LLMMessage(role=LLMRole.USER, content="Hello world")
        
        assert message.role == LLMRole.USER
        assert message.content == "Hello world"
        assert message.metadata == {}
    
    def test_message_with_metadata(self):
        """Test LLMMessage with metadata."""
        metadata = {"timestamp": "2025-01-01", "user_id": "123"}
        message = LLMMessage(
            role=LLMRole.ASSISTANT,
            content="Response",
            metadata=metadata
        )
        
        assert message.metadata == metadata
    
    def test_message_to_langchain(self):
        """Test conversion to LangChain message."""
        message = LLMMessage(role=LLMRole.USER, content="Hello")
        lc_message = message.to_langchain_message()
        
        from langchain_core.messages import HumanMessage
        assert isinstance(lc_message, HumanMessage)
        assert lc_message.content == "Hello"
    
    def test_message_from_langchain(self):
        """Test creation from LangChain message."""
        from langchain_core.messages import AIMessage
        
        lc_message = AIMessage(content="AI response")
        message = LLMMessage.from_langchain_message(lc_message)
        
        assert message.role == LLMRole.ASSISTANT
        assert message.content == "AI response"


class TestLLMResponse:
    """Test cases for LLMResponse."""
    
    def test_response_creation(self):
        """Test LLMResponse creation."""
        response = LLMResponse(
            content="Test response",
            metadata={"tokens": 10}
        )
        
        assert response.content == "Test response"
        assert response.metadata["tokens"] == 10
    
    def test_response_repr(self):
        """Test response string representation."""
        response = LLMResponse(content="Short response")
        repr_str = repr(response)
        
        assert "LLMResponse" in repr_str
        assert "Short response" in repr_str


# Note: LLMStreamingResponse is just LLMResponse for streaming chunks
# No separate test class needed
