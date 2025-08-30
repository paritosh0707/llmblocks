"""
Comprehensive LLM Provider Tests

This module provides extensive testing for the LLMBlocks LLM provider system,
including memory-enhanced providers and all integrations.
"""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from llmblocks.blocks.llm_provider import (
    get_provider,
    get_stateful_ai,
    get_persistent_ai,
    MemoryEnhancedLLMProvider,
    BaseLLMProvider,
    LLMResponse,
    LLMRole
)


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, config):
        super().__init__(config)
        self.generate_calls = []
        self.stream_calls = []
    
    async def _initialize_impl(self):
        """Initialize mock provider."""
        pass
    
    async def generate(self, messages, **kwargs):
        """Mock generate method."""
        self.generate_calls.append((messages, kwargs))
        
        # Create a realistic response based on the last message
        last_message = messages[-1] if messages else {"content": "Hello"}
        response_content = f"Mock response to: {last_message.get('content', 'unknown')}"
        
        return LLMResponse(
            content=response_content,
            role=LLMRole.ASSISTANT,
            model="mock-model",
            usage={"input_tokens": 10, "output_tokens": 20},
            metadata={"mock": True}
        )
    
    async def generate_stream(self, messages, **kwargs):
        """Mock streaming generate method."""
        self.stream_calls.append((messages, kwargs))
        
        last_message = messages[-1] if messages else {"content": "Hello"}
        response_content = f"Mock stream response to: {last_message.get('content', 'unknown')}"
        
        # Yield response in chunks
        words = response_content.split()
        for word in words:
            yield LLMResponse(
                content=word + " ",
                role=LLMRole.ASSISTANT,
                model="mock-model",
                usage={"input_tokens": 1, "output_tokens": 1},
                metadata={"mock": True, "chunk": True}
            )


class TestLLMProviderCore:
    """Test core LLM provider functionality."""
    
    @pytest.mark.asyncio
    async def test_mock_provider_generation(self):
        """Test mock provider basic generation."""
        from llmblocks.blocks.llm_provider.base import LLMProviderConfig
        
        config = LLMProviderConfig(provider_name="mock", model="mock-model")
        provider = MockLLMProvider(config)
        await provider.initialize()
        
        messages = [{"role": "user", "content": "Hello!"}]
        response = await provider.generate(messages)
        
        assert response.content == "Mock response to: Hello!"
        assert response.role == LLMRole.ASSISTANT
        assert len(provider.generate_calls) == 1
    
    @pytest.mark.asyncio
    async def test_mock_provider_streaming(self):
        """Test mock provider streaming."""
        from llmblocks.blocks.llm_provider.base import LLMProviderConfig
        
        config = LLMProviderConfig(provider_name="mock", model="mock-model")
        provider = MockLLMProvider(config)
        await provider.initialize()
        
        messages = [{"role": "user", "content": "Stream test"}]
        chunks = []
        
        async for chunk in provider.generate_stream(messages):
            chunks.append(chunk.content)
        
        full_response = "".join(chunks)
        assert "Mock stream response to: Stream test" in full_response
        assert len(provider.stream_calls) == 1


class TestMemoryEnhancedProvider:
    """Test memory-enhanced LLM providers."""
    
    @pytest.mark.asyncio
    async def test_memory_enhanced_creation(self):
        """Test creating memory-enhanced provider."""
        from llmblocks.blocks.llm_provider.base import LLMProviderConfig
        from llmblocks.blocks.memory import get_conversation_memory
        
        # Create mock LLM provider
        config = LLMProviderConfig(provider_name="mock", model="mock-model")
        llm_provider = MockLLMProvider(config)
        await llm_provider.initialize()
        
        # Create memory
        memory = await get_conversation_memory(backend_type="in_memory")
        
        # Create memory-enhanced provider
        enhanced_provider = MemoryEnhancedLLMProvider(llm_provider, memory)
        await enhanced_provider.initialize()
        
        assert enhanced_provider.llm_provider == llm_provider
        assert enhanced_provider.memory == memory
        
        await enhanced_provider.close()
    
    @pytest.mark.asyncio
    async def test_memory_enhanced_chat(self):
        """Test memory-enhanced chat functionality."""
        from llmblocks.blocks.llm_provider.base import LLMProviderConfig
        from llmblocks.blocks.memory import get_conversation_memory
        
        # Create components
        config = LLMProviderConfig(provider_name="mock", model="mock-model")
        llm_provider = MockLLMProvider(config)
        await llm_provider.initialize()
        
        memory = await get_conversation_memory(backend_type="in_memory")
        enhanced_provider = MemoryEnhancedLLMProvider(llm_provider, memory)
        await enhanced_provider.initialize()
        
        # Test chat
        response = await enhanced_provider.chat("Hello, I'm Alice!")
        assert "Mock response to: Hello, I'm Alice!" in response
        
        # Check memory was updated
        history = await memory.get_conversation_history()
        assert len(history) == 2  # User message + AI response
        assert history[0].content == "Hello, I'm Alice!"
        assert "Mock response" in history[1].content
        
        # Test context awareness
        response2 = await enhanced_provider.chat("What's my name?")
        
        # Should have context from previous messages
        history2 = await memory.get_conversation_history()
        assert len(history2) == 4  # 2 previous + 2 new
        
        await enhanced_provider.close()
    
    @pytest.mark.asyncio
    async def test_memory_enhanced_streaming(self):
        """Test memory-enhanced streaming."""
        from llmblocks.blocks.llm_provider.base import LLMProviderConfig
        from llmblocks.blocks.memory import get_conversation_memory
        
        # Create components
        config = LLMProviderConfig(provider_name="mock", model="mock-model")
        llm_provider = MockLLMProvider(config)
        await llm_provider.initialize()
        
        memory = await get_conversation_memory(backend_type="in_memory")
        enhanced_provider = MemoryEnhancedLLMProvider(llm_provider, memory)
        await enhanced_provider.initialize()
        
        # Test streaming chat
        chunks = []
        async for chunk in enhanced_provider.chat_stream("Stream test message"):
            chunks.append(chunk)
        
        full_response = "".join(chunks)
        assert "Mock stream response" in full_response
        
        # Check memory was updated
        history = await memory.get_conversation_history()
        assert len(history) == 2  # User message + AI response
        assert history[0].content == "Stream test message"
        
        await enhanced_provider.close()


class TestStatefulAIFactory:
    """Test stateful AI factory functions."""
    
    @pytest.mark.asyncio
    @patch('llmblocks.blocks.llm_provider.factory.get_provider')
    async def test_get_stateful_ai(self, mock_get_provider):
        """Test get_stateful_ai factory function."""
        # Mock the LLM provider
        mock_provider = MockLLMProvider(MagicMock())
        await mock_provider.initialize()
        mock_get_provider.return_value = mock_provider
        
        # Test creation
        ai = await get_stateful_ai(
            provider_type="mock",
            api_key="test-key",
            session_id="test-session"
        )
        
        assert isinstance(ai, MemoryEnhancedLLMProvider)
        assert ai.get_session_id() == "test-session"
        
        # Test chat functionality
        response = await ai.chat("Hello!")
        assert "Mock response" in response
        
        await ai.close()
    
    @pytest.mark.asyncio
    @patch('llmblocks.blocks.llm_provider.factory.get_provider')
    async def test_get_persistent_ai(self, mock_get_provider):
        """Test get_persistent_ai factory function."""
        # Mock the LLM provider
        mock_provider = MockLLMProvider(MagicMock())
        await mock_provider.initialize()
        mock_get_provider.return_value = mock_provider
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test creation
            ai = await get_persistent_ai(
                provider_type="mock",
                api_key="test-key",
                session_id="persistent-session",
                storage_dir=temp_dir
            )
            
            assert isinstance(ai, MemoryEnhancedLLMProvider)
            assert ai.get_session_id() == "persistent-session"
            
            # Test persistence
            await ai.chat("Remember this message")
            await ai.close()
            
            # Create new instance with same session
            ai2 = await get_persistent_ai(
                provider_type="mock",
                api_key="test-key",
                session_id="persistent-session",
                storage_dir=temp_dir
            )
            
            # Should have previous conversation
            summary = await ai2.get_conversation_summary()
            assert summary["total_messages"] > 0
            
            await ai2.close()


class TestMemoryEnhancedFeatures:
    """Test advanced features of memory-enhanced providers."""
    
    @pytest.mark.asyncio
    async def test_conversation_summary(self):
        """Test conversation summary functionality."""
        from llmblocks.blocks.llm_provider.base import LLMProviderConfig
        from llmblocks.blocks.memory import get_conversation_memory
        
        # Create components
        config = LLMProviderConfig(provider_name="mock", model="mock-model")
        llm_provider = MockLLMProvider(config)
        await llm_provider.initialize()
        
        memory = await get_conversation_memory(backend_type="in_memory")
        enhanced_provider = MemoryEnhancedLLMProvider(llm_provider, memory)
        await enhanced_provider.initialize()
        
        # Have a conversation
        await enhanced_provider.chat("Hello!")
        await enhanced_provider.chat("How are you?")
        
        # Get summary
        summary = await enhanced_provider.get_conversation_summary()
        
        assert summary["total_messages"] == 4  # 2 user + 2 AI
        assert summary["conversation_turns"] == 2
        assert summary["memory_backend"] == "in_memory"
        assert summary["session_id"] is not None
        
        await enhanced_provider.close()
    
    @pytest.mark.asyncio
    async def test_conversation_search(self):
        """Test conversation search functionality."""
        from llmblocks.blocks.llm_provider.base import LLMProviderConfig
        from llmblocks.blocks.memory import get_conversation_memory
        
        # Create components
        config = LLMProviderConfig(provider_name="mock", model="mock-model")
        llm_provider = MockLLMProvider(config)
        await llm_provider.initialize()
        
        memory = await get_conversation_memory(backend_type="in_memory")
        enhanced_provider = MemoryEnhancedLLMProvider(llm_provider, memory)
        await enhanced_provider.initialize()
        
        # Have a conversation with searchable content
        await enhanced_provider.chat("I love Python programming")
        await enhanced_provider.chat("JavaScript is also interesting")
        await enhanced_provider.chat("But Python is my favorite")
        
        # Search conversation
        results = await enhanced_provider.search_conversation("Python")
        
        assert len(results) >= 1  # Should find messages with "Python"
        assert any("Python" in msg.content for msg in results)
        
        await enhanced_provider.close()
    
    @pytest.mark.asyncio
    async def test_session_management(self):
        """Test session management functionality."""
        from llmblocks.blocks.llm_provider.base import LLMProviderConfig
        from llmblocks.blocks.memory import get_conversation_memory
        
        # Create components
        config = LLMProviderConfig(provider_name="mock", model="mock-model")
        llm_provider = MockLLMProvider(config)
        await llm_provider.initialize()
        
        memory = await get_conversation_memory(backend_type="in_memory")
        enhanced_provider = MemoryEnhancedLLMProvider(llm_provider, memory)
        await enhanced_provider.initialize()
        
        # Test initial session
        original_session = enhanced_provider.get_session_id()
        await enhanced_provider.chat("Message in original session")
        
        # Switch session
        enhanced_provider.set_session_id("new-session")
        assert enhanced_provider.get_session_id() == "new-session"
        
        # New session should be empty
        summary = await enhanced_provider.get_conversation_summary()
        assert summary["total_messages"] == 0
        
        await enhanced_provider.close()
    
    @pytest.mark.asyncio
    async def test_clear_conversation(self):
        """Test conversation clearing functionality."""
        from llmblocks.blocks.llm_provider.base import LLMProviderConfig
        from llmblocks.blocks.memory import get_conversation_memory
        
        # Create components
        config = LLMProviderConfig(provider_name="mock", model="mock-model")
        llm_provider = MockLLMProvider(config)
        await llm_provider.initialize()
        
        memory = await get_conversation_memory(backend_type="in_memory")
        enhanced_provider = MemoryEnhancedLLMProvider(llm_provider, memory)
        await enhanced_provider.initialize(system_prompt="You are a helpful assistant")
        
        # Have a conversation
        await enhanced_provider.chat("Hello!")
        await enhanced_provider.chat("How are you?")
        
        summary_before = await enhanced_provider.get_conversation_summary()
        assert summary_before["total_messages"] > 0
        
        # Clear conversation
        await enhanced_provider.clear_conversation()
        
        summary_after = await enhanced_provider.get_conversation_summary()
        # Should only have system prompt remaining
        assert summary_after["total_messages"] == 1
        
        await enhanced_provider.close()


class TestMemoryEnhancedIntegration:
    """Integration tests for memory-enhanced providers."""
    
    @pytest.mark.asyncio
    async def test_context_manager_support(self):
        """Test context manager support."""
        from llmblocks.blocks.llm_provider.base import LLMProviderConfig
        from llmblocks.blocks.memory import get_conversation_memory
        
        config = LLMProviderConfig(provider_name="mock", model="mock-model")
        llm_provider = MockLLMProvider(config)
        await llm_provider.initialize()
        
        memory = await get_conversation_memory(backend_type="in_memory")
        
        # Test context manager
        async with MemoryEnhancedLLMProvider(llm_provider, memory) as ai:
            await ai.initialize()
            response = await ai.chat("Hello!")
            assert "Mock response" in response
        
        # Should be closed automatically
    
    @pytest.mark.asyncio
    async def test_repr_string(self):
        """Test string representation."""
        from llmblocks.blocks.llm_provider.base import LLMProviderConfig
        from llmblocks.blocks.memory import get_conversation_memory
        
        config = LLMProviderConfig(provider_name="mock", model="mock-model")
        llm_provider = MockLLMProvider(config)
        await llm_provider.initialize()
        
        memory = await get_conversation_memory(
            backend_type="in_memory",
            session_id="test-session"
        )
        
        enhanced_provider = MemoryEnhancedLLMProvider(llm_provider, memory)
        
        repr_str = repr(enhanced_provider)
        assert "MemoryEnhancedLLMProvider" in repr_str
        assert "MockLLMProvider" in repr_str
        assert "test-session" in repr_str
        assert "in_memory" in repr_str
        
        await enhanced_provider.close()


@pytest.mark.integration
class TestRealProviderIntegration:
    """Integration tests with real providers (requires API keys)."""
    
    @pytest.mark.skipif(
        not os.getenv('GOOGLE_API_KEY'),
        reason="GOOGLE_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_gemini_stateful_ai(self):
        """Test stateful AI with real Gemini provider."""
        ai = await get_stateful_ai(
            provider_type="gemini",
            api_key=os.getenv('GOOGLE_API_KEY'),
            session_id="integration-test"
        )
        
        # Test basic functionality
        response1 = await ai.chat("Hello! My name is TestBot.")
        assert len(response1) > 0
        
        response2 = await ai.chat("What's my name?")
        assert len(response2) > 0
        
        # Check conversation summary
        summary = await ai.get_conversation_summary()
        assert summary["conversation_turns"] == 2
        assert summary["total_messages"] == 4
        
        await ai.close()


# Fixtures
@pytest.fixture
async def mock_llm_provider():
    """Fixture providing a mock LLM provider."""
    from llmblocks.blocks.llm_provider.base import LLMProviderConfig
    
    config = LLMProviderConfig(provider_name="mock", model="mock-model")
    provider = MockLLMProvider(config)
    await provider.initialize()
    yield provider


@pytest.fixture
async def memory_enhanced_provider(mock_llm_provider):
    """Fixture providing a memory-enhanced provider."""
    from llmblocks.blocks.memory import get_conversation_memory
    
    memory = await get_conversation_memory(backend_type="in_memory")
    enhanced_provider = MemoryEnhancedLLMProvider(mock_llm_provider, memory)
    await enhanced_provider.initialize()
    
    yield enhanced_provider
    
    await enhanced_provider.close()
