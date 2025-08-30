"""
Comprehensive Memory System Tests

This module provides extensive testing for the LLMBlocks memory system,
covering all functionality, edge cases, and performance scenarios.
"""

import pytest
import asyncio
import tempfile
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Dict, Any

from llmblocks.blocks.memory import (
    get_conversation_memory,
    get_memory,
    MemoryRole,
    MemoryMessage,
    ConversationMemory,
    InMemoryBackend,
    FileBackend
)


class TestMemoryCore:
    """Test core memory functionality."""
    
    @pytest.mark.asyncio
    async def test_memory_creation(self):
        """Test memory creation with different backends."""
        # In-memory backend
        memory = await get_conversation_memory(backend_type="in_memory")
        assert memory is not None
        assert memory.config.backend_type == "in_memory"
        await memory.close()
        
        # File backend
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = await get_conversation_memory(
                backend_type="file",
                backend_config={"storage_dir": temp_dir}
            )
            assert memory is not None
            assert memory.config.backend_type == "file"
            await memory.close()
    
    @pytest.mark.asyncio
    async def test_message_operations(self):
        """Test basic message operations."""
        memory = await get_conversation_memory(backend_type="in_memory")
        
        # Add messages
        user_msg = await memory.add_user_message("Hello!")
        assert user_msg.role == MemoryRole.USER
        assert user_msg.content == "Hello!"
        assert user_msg.message_id is not None
        
        ai_msg = await memory.add_assistant_message("Hi there!")
        assert ai_msg.role == MemoryRole.ASSISTANT
        assert ai_msg.content == "Hi there!"
        
        system_msg = await memory.add_system_message("System message")
        assert system_msg.role == MemoryRole.SYSTEM
        
        # Get messages
        history = await memory.get_conversation_history()
        assert len(history) == 3
        assert history[0].content == "Hello!"
        assert history[1].content == "Hi there!"
        assert history[2].content == "System message"
        
        await memory.close()
    
    @pytest.mark.asyncio
    async def test_context_window(self):
        """Test context window functionality."""
        memory = await get_conversation_memory(
            backend_type="in_memory",
            max_context_messages=3
        )
        
        # Add more messages than context window
        for i in range(5):
            await memory.add_user_message(f"Message {i}")
        
        context = await memory.get_context_window()
        assert len(context) <= 3
        
        await memory.close()
    
    @pytest.mark.asyncio
    async def test_message_search(self):
        """Test message search functionality."""
        memory = await get_conversation_memory(backend_type="in_memory")
        
        await memory.add_user_message("I love Python programming")
        await memory.add_user_message("JavaScript is also great")
        await memory.add_user_message("Python is my favorite")
        
        # Search for Python
        results = await memory.search_messages("Python")
        assert len(results) == 2
        assert all("Python" in msg.content for msg in results)
        
        # Search with limit
        results = await memory.search_messages("Python", limit=1)
        assert len(results) == 1
        
        await memory.close()
    
    @pytest.mark.asyncio
    async def test_memory_stats(self):
        """Test memory statistics."""
        memory = await get_conversation_memory(backend_type="in_memory")
        
        await memory.add_user_message("Hello")
        await memory.add_assistant_message("Hi")
        
        stats = await memory.get_conversation_stats()
        assert stats["total_messages"] == 2
        assert stats["exists"] == True
        assert "role_distribution" in stats
        
        await memory.close()
    
    @pytest.mark.asyncio
    async def test_memory_clear(self):
        """Test memory clearing."""
        memory = await get_conversation_memory(backend_type="in_memory")
        
        await memory.add_user_message("Hello")
        await memory.add_assistant_message("Hi")
        
        history_before = await memory.get_conversation_history()
        assert len(history_before) == 2
        
        cleared_count = await memory.clear_memory()
        assert cleared_count == 2
        
        history_after = await memory.get_conversation_history()
        assert len(history_after) == 0
        
        await memory.close()


class TestMemoryPersistence:
    """Test memory persistence across sessions."""
    
    @pytest.mark.asyncio
    async def test_file_persistence(self):
        """Test file-based persistence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            session_id = "test_persistence"
            
            # Session 1: Save data
            memory1 = await get_conversation_memory(
                backend_type="file",
                session_id=session_id,
                backend_config={"storage_dir": temp_dir}
            )
            
            await memory1.add_user_message("Persistent message 1")
            await memory1.add_assistant_message("Persistent response 1")
            await memory1.close()
            
            # Session 2: Load data
            memory2 = await get_conversation_memory(
                backend_type="file",
                session_id=session_id,
                backend_config={"storage_dir": temp_dir}
            )
            
            history = await memory2.get_conversation_history()
            assert len(history) == 2
            assert "Persistent message 1" in history[0].content
            assert "Persistent response 1" in history[1].content
            
            await memory2.close()
    
    @pytest.mark.asyncio
    async def test_session_isolation(self):
        """Test that different sessions are isolated."""
        memory1 = await get_conversation_memory(
            backend_type="in_memory",
            session_id="session1"
        )
        memory2 = await get_conversation_memory(
            backend_type="in_memory", 
            session_id="session2"
        )
        
        await memory1.add_user_message("Message for session 1")
        await memory2.add_user_message("Message for session 2")
        
        history1 = await memory1.get_conversation_history()
        history2 = await memory2.get_conversation_history()
        
        assert len(history1) == 1
        assert len(history2) == 1
        assert history1[0].content != history2[0].content
        
        await memory1.close()
        await memory2.close()


class TestMemoryEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_empty_messages(self):
        """Test handling of empty messages."""
        memory = await get_conversation_memory(backend_type="in_memory")
        
        # Empty message should still be stored
        msg = await memory.add_user_message("")
        assert msg.content == ""
        
        history = await memory.get_conversation_history()
        assert len(history) == 1
        
        await memory.close()
    
    @pytest.mark.asyncio
    async def test_large_messages(self):
        """Test handling of large messages."""
        memory = await get_conversation_memory(backend_type="in_memory")
        
        # 10KB message
        large_content = "A" * 10000
        msg = await memory.add_user_message(large_content)
        assert len(msg.content) == 10000
        
        history = await memory.get_conversation_history()
        assert len(history[0].content) == 10000
        
        await memory.close()
    
    @pytest.mark.asyncio
    async def test_unicode_messages(self):
        """Test handling of Unicode and special characters."""
        memory = await get_conversation_memory(backend_type="in_memory")
        
        unicode_content = "Hello! 🚀 Testing émojis and spëcial chars: @#$%^&*()"
        msg = await memory.add_user_message(unicode_content)
        assert msg.content == unicode_content
        
        history = await memory.get_conversation_history()
        assert history[0].content == unicode_content
        
        await memory.close()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent memory operations."""
        memory = await get_conversation_memory(backend_type="in_memory")
        
        # Concurrent message additions
        tasks = []
        for i in range(50):
            tasks.append(memory.add_user_message(f"Concurrent message {i}"))
        
        messages = await asyncio.gather(*tasks)
        assert len(messages) == 50
        
        history = await memory.get_conversation_history()
        assert len(history) == 50
        
        await memory.close()


class TestMemoryPerformance:
    """Test memory system performance."""
    
    @pytest.mark.asyncio
    async def test_bulk_operations(self):
        """Test performance with bulk operations."""
        memory = await get_conversation_memory(backend_type="in_memory")
        
        # Bulk insertion
        start_time = time.time()
        for i in range(1000):
            await memory.add_user_message(f"Performance test message {i}")
        insertion_time = time.time() - start_time
        
        # Should be able to insert 1000 messages in reasonable time
        assert insertion_time < 5.0  # 5 seconds max
        
        # Bulk retrieval
        start_time = time.time()
        history = await memory.get_conversation_history()
        retrieval_time = time.time() - start_time
        
        assert len(history) == 1000
        assert retrieval_time < 1.0  # 1 second max
        
        await memory.close()
    
    @pytest.mark.asyncio
    async def test_search_performance(self):
        """Test search performance."""
        memory = await get_conversation_memory(backend_type="in_memory")
        
        # Add many messages
        for i in range(500):
            await memory.add_user_message(f"Message {i} with search term")
            await memory.add_user_message(f"Message {i} without term")
        
        # Search performance
        start_time = time.time()
        results = await memory.search_messages("search term", limit=100)
        search_time = time.time() - start_time
        
        assert len(results) == 100  # Limited by limit parameter
        assert search_time < 1.0  # Should be fast
        
        await memory.close()


class TestMemoryBackends:
    """Test different memory backends."""
    
    @pytest.mark.asyncio
    async def test_in_memory_backend(self):
        """Test in-memory backend functionality."""
        memory = await get_conversation_memory(backend_type="in_memory")
        
        await memory.add_user_message("Test message")
        history = await memory.get_conversation_history()
        assert len(history) == 1
        
        stats = await memory.get_conversation_stats()
        assert stats["backend_type"] == "in_memory"
        
        await memory.close()
    
    @pytest.mark.asyncio
    async def test_file_backend(self):
        """Test file backend functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = await get_conversation_memory(
                backend_type="file",
                backend_config={"storage_dir": temp_dir}
            )
            
            await memory.add_user_message("Test message")
            history = await memory.get_conversation_history()
            assert len(history) == 1
            
            stats = await memory.get_conversation_stats()
            assert stats["backend_type"] == "file"
            
            # Check file was created
            files = list(Path(temp_dir).glob("*.json"))
            assert len(files) == 1
            
            await memory.close()


class TestMemoryContextStrategies:
    """Test different context management strategies."""
    
    @pytest.mark.asyncio
    async def test_sliding_window_strategy(self):
        """Test sliding window context strategy."""
        memory = await get_conversation_memory(
            backend_type="in_memory",
            context_strategy="sliding_window",
            max_context_messages=3
        )
        
        # Add more messages than window size
        for i in range(6):
            await memory.add_user_message(f"Message {i}")
        
        context = await memory.get_context_window()
        assert len(context) == 3
        # Should contain the last 3 messages
        assert "Message 3" in context[0].content
        assert "Message 4" in context[1].content
        assert "Message 5" in context[2].content
        
        await memory.close()
    
    @pytest.mark.asyncio
    async def test_priority_based_strategy(self):
        """Test priority-based context strategy."""
        memory = await get_conversation_memory(
            backend_type="in_memory",
            context_strategy="priority_based",
            max_context_messages=3
        )
        
        # Add messages with different priorities
        for i in range(6):
            await memory.add_user_message(f"Message {i}")
        
        context = await memory.get_context_window()
        assert len(context) <= 3
        
        await memory.close()


@pytest.mark.integration
class TestMemoryIntegration:
    """Integration tests for memory system."""
    
    @pytest.mark.asyncio
    async def test_memory_with_metadata(self):
        """Test memory operations with metadata."""
        memory = await get_conversation_memory(backend_type="in_memory")
        
        metadata = {"user_id": "123", "session_type": "chat"}
        msg = await memory.add_user_message("Hello", metadata=metadata)
        
        assert msg.metadata["user_id"] == "123"
        assert msg.metadata["session_type"] == "chat"
        
        history = await memory.get_conversation_history()
        assert history[0].metadata["user_id"] == "123"
        
        await memory.close()
    
    @pytest.mark.asyncio
    async def test_memory_factory_functions(self):
        """Test memory factory functions."""
        # Test get_memory function
        memory1 = await get_memory("conversation", "in_memory")
        assert isinstance(memory1, ConversationMemory)
        await memory1.close()
        
        # Test get_conversation_memory function
        memory2 = await get_conversation_memory("in_memory")
        assert isinstance(memory2, ConversationMemory)
        await memory2.close()


@pytest.mark.slow
class TestMemoryStress:
    """Stress tests for memory system."""
    
    @pytest.mark.asyncio
    async def test_high_volume_operations(self):
        """Test memory under high volume."""
        memory = await get_conversation_memory(backend_type="in_memory")
        
        # Add 10,000 messages
        start_time = time.time()
        for i in range(10000):
            await memory.add_user_message(f"Stress test message {i}")
        
        total_time = time.time() - start_time
        messages_per_second = 10000 / total_time
        
        # Should handle at least 1000 messages per second
        assert messages_per_second > 1000
        
        # Verify all messages are stored
        history = await memory.get_conversation_history()
        assert len(history) == 10000
        
        await memory.close()
    
    @pytest.mark.asyncio
    async def test_memory_cleanup(self):
        """Test memory cleanup and resource management."""
        memories = []
        
        # Create many memory instances
        for i in range(100):
            memory = await get_conversation_memory(
                backend_type="in_memory",
                session_id=f"session_{i}"
            )
            await memory.add_user_message(f"Message for session {i}")
            memories.append(memory)
        
        # Close all memories
        for memory in memories:
            await memory.close()
        
        # Should complete without errors
        assert len(memories) == 100


# Fixtures for testing
@pytest.fixture
async def memory_instance():
    """Fixture providing a memory instance."""
    memory = await get_conversation_memory(backend_type="in_memory")
    yield memory
    await memory.close()


@pytest.fixture
async def populated_memory():
    """Fixture providing a memory instance with sample data."""
    memory = await get_conversation_memory(backend_type="in_memory")
    
    await memory.add_user_message("Hello!")
    await memory.add_assistant_message("Hi there!")
    await memory.add_user_message("How are you?")
    await memory.add_assistant_message("I'm doing well, thank you!")
    
    yield memory
    await memory.close()


@pytest.fixture
def temp_storage_dir():
    """Fixture providing a temporary storage directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir
