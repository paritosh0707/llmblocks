"""
Memory-Enhanced LLM Provider

This module provides LLM providers with built-in memory capabilities,
enabling stateful conversations with minimal code.

The dream: Stateful AI in just 3 lines of code!
"""

import asyncio
from typing import Any, Dict, List, Optional, Union

from .base import BaseLLMProvider, LLMResponse
from ..memory import get_conversation_memory, ConversationMemory
from ...utils.logging import get_logger


class MemoryEnhancedLLMProvider:
    """
    LLM Provider with built-in conversation memory.
    
    This class wraps any LLM provider and adds automatic memory management,
    enabling stateful conversations with zero additional code.
    
    Features:
    - Automatic conversation history management
    - Context-aware responses
    - Persistent memory across sessions
    - Multiple backend support (in-memory, file, Redis)
    - LangChain compatibility
    
    Example:
        ```python
        # 3-line stateful AI!
        ai = await get_stateful_ai("gemini", api_key="your-key")
        response = await ai.chat("Hello! I'm Alice.")
        response2 = await ai.chat("What's my name?")  # AI remembers: "Alice"
        ```
    """
    
    def __init__(self, llm_provider: BaseLLMProvider, memory: ConversationMemory):
        self.llm_provider = llm_provider
        self.memory = memory
        self.logger = get_logger("MemoryEnhancedLLMProvider")
        
        # Configuration
        self.system_prompt = None
        self.auto_add_system_prompt = True
        self.include_context_in_response = False
    
    async def initialize(self, system_prompt: Optional[str] = None):
        """Initialize the memory-enhanced LLM provider."""
        if system_prompt:
            self.system_prompt = system_prompt
            if self.auto_add_system_prompt:
                await self.memory.add_system_message(system_prompt)
    
    async def chat(self, message: str, **kwargs) -> str:
        """
        Chat with the AI (with automatic memory management).
        
        This is the main method that provides the "3-line AI" experience:
        1. Adds user message to memory
        2. Gets conversation context
        3. Generates response with context
        4. Adds AI response to memory
        5. Returns the response
        
        Args:
            message: User's message
            **kwargs: Additional parameters for LLM generation
            
        Returns:
            AI's response as a string
        """
        # Add user message to memory
        await self.memory.add_user_message(message)
        
        # Get conversation context
        context_messages = await self.memory.get_context_window()
        
        # Convert to LLM format
        llm_messages = []
        for msg in context_messages:
            llm_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        # Generate response with context
        response = await self.llm_provider.generate(llm_messages, **kwargs)
        ai_response = response.content
        
        # Add AI response to memory
        await self.memory.add_assistant_message(ai_response)
        
        return ai_response
    
    async def chat_stream(self, message: str, **kwargs):
        """
        Stream chat with the AI (with automatic memory management).
        
        Similar to chat() but streams the response in real-time.
        """
        # Add user message to memory
        await self.memory.add_user_message(message)
        
        # Get conversation context
        context_messages = await self.memory.get_context_window()
        
        # Convert to LLM format
        llm_messages = []
        for msg in context_messages:
            llm_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        # Stream response
        full_response = ""
        async for chunk in self.llm_provider.generate_stream(llm_messages, **kwargs):
            full_response += chunk.content
            yield chunk.content
        
        # Add complete AI response to memory
        await self.memory.add_assistant_message(full_response)
    
    async def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation."""
        stats = await self.memory.get_conversation_stats()
        history = await self.memory.get_conversation_history()
        
        return {
            "session_id": self.memory.get_session_id(),
            "total_messages": len(history),
            "conversation_turns": len([m for m in history if m.role.value == "user"]),
            "context_messages": stats.get("context_messages", 0),
            "memory_backend": stats.get("backend_type", "unknown"),
            "last_message": history[-1].content if history else None
        }
    
    async def clear_conversation(self):
        """Clear the conversation history."""
        await self.memory.clear_memory()
        
        # Re-add system prompt if configured
        if self.system_prompt and self.auto_add_system_prompt:
            await self.memory.add_system_message(self.system_prompt)
    
    async def search_conversation(self, query: str, limit: int = 10):
        """Search the conversation history."""
        return await self.memory.search_messages(query, limit)
    
    def set_session_id(self, session_id: str):
        """Change the session ID (switch conversations)."""
        self.memory.set_session_id(session_id)
    
    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self.memory.get_session_id()
    
    # LangChain compatibility
    def to_langchain_memory(self):
        """Convert to LangChain-compatible memory."""
        return self.memory.to_langchain_memory()
    
    def to_langchain_chat_history(self):
        """Convert to LangChain chat message history."""
        return self.memory.to_langchain_chat_history()
    
    async def close(self):
        """Close the provider and cleanup resources."""
        await self.memory.close()
        # Only close LLM provider if it has a close method
        if hasattr(self.llm_provider, 'close'):
            await self.llm_provider.close()
    
    # Context manager support
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    def __repr__(self) -> str:
        return (
            f"MemoryEnhancedLLMProvider("
            f"provider={self.llm_provider.__class__.__name__}, "
            f"session='{self.get_session_id()}', "
            f"backend='{self.memory.config.backend_type}'"
            f")"
        )


# Factory functions for the "3-line AI" experience
async def get_stateful_ai(
    provider_type: str,
    api_key: str,
    session_id: Optional[str] = None,
    memory_backend: str = "in_memory",
    system_prompt: Optional[str] = None,
    **kwargs
) -> MemoryEnhancedLLMProvider:
    """
    Create a stateful AI in one line!
    
    This is the ultimate convenience function that creates an LLM provider
    with built-in memory management.
    
    Args:
        provider_type: LLM provider type ("gemini", "openai", "anthropic")
        api_key: API key for the provider
        session_id: Optional session ID for conversation persistence
        memory_backend: Memory backend ("in_memory", "file", "redis")
        system_prompt: Optional system prompt
        **kwargs: Additional configuration
        
    Returns:
        Memory-enhanced LLM provider ready for stateful conversations
        
    Example:
        ```python
        # The dream: 3-line stateful AI!
        ai = await get_stateful_ai("gemini", api_key="your-key")
        response = await ai.chat("Hello! I'm Alice.")
        response2 = await ai.chat("What's my name?")  # Remembers "Alice"!
        ```
    """
    from . import get_provider
    
    # Create LLM provider
    llm_provider = await get_provider(provider_type, api_key=api_key, **kwargs)
    
    # Create memory
    memory_config = {}
    if memory_backend == "file":
        memory_config["storage_dir"] = kwargs.get("storage_dir", "./conversations")
    elif memory_backend == "redis":
        memory_config.update(kwargs.get("redis_config", {}))
    
    memory = await get_conversation_memory(
        backend_type=memory_backend,
        session_id=session_id,
        backend_config=memory_config,
        max_context_messages=kwargs.get("max_context_messages", 50),
        context_strategy=kwargs.get("context_strategy", "sliding_window")
    )
    
    # Create memory-enhanced provider
    enhanced_provider = MemoryEnhancedLLMProvider(llm_provider, memory)
    
    # Initialize with system prompt
    if system_prompt:
        await enhanced_provider.initialize(system_prompt)
    
    return enhanced_provider


async def get_persistent_ai(
    provider_type: str,
    api_key: str,
    session_id: str,
    storage_dir: str = "./conversations",
    system_prompt: Optional[str] = None,
    **kwargs
) -> MemoryEnhancedLLMProvider:
    """
    Create a persistent AI that remembers across restarts.
    
    This creates an AI with file-based memory that persists conversations
    across application restarts.
    
    Args:
        provider_type: LLM provider type
        api_key: API key
        session_id: Session ID for persistence
        storage_dir: Directory to store conversation files
        system_prompt: Optional system prompt
        **kwargs: Additional configuration
        
    Returns:
        Persistent memory-enhanced LLM provider
    """
    return await get_stateful_ai(
        provider_type=provider_type,
        api_key=api_key,
        session_id=session_id,
        memory_backend="file",
        storage_dir=storage_dir,
        system_prompt=system_prompt,
        **kwargs
    )


# Convenience aliases
create_stateful_ai = get_stateful_ai
create_persistent_ai = get_persistent_ai
