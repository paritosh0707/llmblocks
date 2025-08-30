"""
Memory factory for LLMBlocks.

This module provides a factory pattern for creating and managing
different types of memory systems and backends.
"""

import asyncio
from typing import Any, Dict, List, Optional, Type, Union

from .base import BaseMemory, MemoryConfig
from .conversation import ConversationMemory, ConversationConfig
from .backends import get_backend
from ...utils.logging import get_logger
from ...utils.exceptions import LLMBlocksError


class MemoryError(LLMBlocksError):
    """Memory factory related errors."""
    pass


class MemoryFactory:
    """
    Factory for creating and managing memory instances.
    
    Supports different memory types (conversation, state, etc.) and
    various storage backends (in-memory, file, Redis, database).
    """
    
    def __init__(self):
        self.logger = get_logger("MemoryFactory")
        self._memory_types: Dict[str, Type[BaseMemory]] = {}
        self._memory_configs: Dict[str, Type[MemoryConfig]] = {}
        self._register_builtin_types()
    
    def _register_builtin_types(self):
        """Register built-in memory types."""
        self.register_memory_type("conversation", ConversationMemory, ConversationConfig)
        # Add more types as they're implemented
        # self.register_memory_type("state", StateMemory, StateConfig)
        # self.register_memory_type("vector", VectorMemory, VectorConfig)
    
    def register_memory_type(
        self, 
        name: str, 
        memory_class: Type[BaseMemory], 
        config_class: Type[MemoryConfig]
    ) -> None:
        """Register a new memory type."""
        self._memory_types[name] = memory_class
        self._memory_configs[name] = config_class
        self.logger.debug(f"Registered memory type: {name}")
    
    def list_memory_types(self) -> List[str]:
        """List available memory types."""
        return list(self._memory_types.keys())
    
    def is_memory_type_available(self, memory_type: str) -> bool:
        """Check if a memory type is available."""
        return memory_type in self._memory_types
    
    async def create_memory(
        self, 
        memory_type: str,
        backend_type: str = "in_memory",
        config: Optional[Union[Dict[str, Any], MemoryConfig]] = None,
        **kwargs
    ) -> BaseMemory:
        """
        Create a memory instance.
        
        Args:
            memory_type: Type of memory (e.g., "conversation")
            backend_type: Storage backend (e.g., "in_memory", "file", "redis")
            config: Configuration dict or MemoryConfig instance
            **kwargs: Additional configuration parameters
            
        Returns:
            Initialized memory instance
        """
        if memory_type not in self._memory_types:
            available_types = ", ".join(self.list_memory_types())
            raise MemoryError(
                f"Unknown memory type: {memory_type}. "
                f"Available types: {available_types}"
            )
        
        memory_class = self._memory_types[memory_type]
        config_class = self._memory_configs[memory_type]
        
        try:
            # Prepare configuration
            if config is None:
                config = {}
            elif isinstance(config, MemoryConfig):
                config = config.model_dump()
            
            # Add backend type and kwargs
            config["backend_type"] = backend_type
            config.update(kwargs)
            
            # Create config instance
            memory_config = config_class(**config)
            
            # Create memory instance
            memory = memory_class(memory_config)
            
            # Initialize the memory
            await memory.initialize()
            
            self.logger.info(
                f"Created {memory_type} memory with {backend_type} backend",
                session_id=memory.session_id
            )
            
            return memory
            
        except Exception as e:
            raise MemoryError(f"Failed to create {memory_type} memory: {e}") from e
    
    async def create_conversation_memory(
        self,
        backend_type: str = "in_memory",
        session_id: Optional[str] = None,
        max_context_messages: int = 50,
        **kwargs
    ) -> ConversationMemory:
        """
        Convenience method to create conversation memory.
        
        Args:
            backend_type: Storage backend type
            session_id: Optional session ID
            max_context_messages: Maximum messages in context window
            **kwargs: Additional configuration
            
        Returns:
            Initialized ConversationMemory instance
        """
        config = {
            "max_context_messages": max_context_messages,
            **kwargs
        }
        
        memory = await self.create_memory("conversation", backend_type, config)
        
        if session_id:
            memory.set_session_id(session_id)
        
        return memory
    
    async def get_or_create_memory(
        self,
        memory_type: str,
        session_id: str,
        backend_type: str = "in_memory",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseMemory:
        """
        Get existing memory or create new one for a session.
        
        This method can be extended to implement memory pooling
        and session management in the future.
        """
        # For now, just create a new memory instance
        # In the future, this could check for existing instances
        memory = await self.create_memory(memory_type, backend_type, config, **kwargs)
        memory.set_session_id(session_id)
        return memory


# Global factory instance
_memory_factory = MemoryFactory()


# Convenience functions
async def get_memory(
    memory_type: str,
    backend_type: str = "in_memory",
    session_id: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], MemoryConfig]] = None,
    **kwargs
) -> BaseMemory:
    """
    Get a memory instance (convenience function).
    
    Args:
        memory_type: Type of memory (e.g., "conversation")
        backend_type: Storage backend (e.g., "in_memory", "file", "redis")
        session_id: Optional session ID
        config: Configuration dict or MemoryConfig instance
        **kwargs: Additional configuration parameters
        
    Returns:
        Initialized memory instance
        
    Example:
        ```python
        # Simple conversation memory
        memory = await get_memory("conversation")
        
        # With specific backend and session
        memory = await get_memory(
            "conversation", 
            backend_type="redis",
            session_id="user_123",
            max_context_messages=100
        )
        ```
    """
    memory = await _memory_factory.create_memory(memory_type, backend_type, config, **kwargs)
    
    if session_id:
        memory.set_session_id(session_id)
    
    return memory


async def get_conversation_memory(
    backend_type: str = "in_memory",
    session_id: Optional[str] = None,
    **kwargs
) -> ConversationMemory:
    """
    Get a conversation memory instance (convenience function).
    
    Args:
        backend_type: Storage backend type
        session_id: Optional session ID
        **kwargs: Additional configuration
        
    Returns:
        Initialized ConversationMemory instance
        
    Example:
        ```python
        # Simple conversation memory
        memory = await get_conversation_memory()
        
        # With Redis backend
        memory = await get_conversation_memory(
            backend_type="redis",
            session_id="chat_123",
            max_context_messages=100
        )
        ```
    """
    return await _memory_factory.create_conversation_memory(
        backend_type=backend_type,
        session_id=session_id,
        **kwargs
    )


def list_memory_types() -> List[str]:
    """List available memory types."""
    return _memory_factory.list_memory_types()


def register_memory_type(
    name: str, 
    memory_class: Type[BaseMemory], 
    config_class: Type[MemoryConfig]
) -> None:
    """Register a custom memory type."""
    _memory_factory.register_memory_type(name, memory_class, config_class)


def get_memory_factory() -> MemoryFactory:
    """Get the global memory factory instance."""
    return _memory_factory


# LangChain integration helpers
async def create_langchain_memory(
    memory_type: str = "conversation",
    backend_type: str = "in_memory",
    session_id: Optional[str] = None,
    memory_key: str = "history",
    **kwargs
):
    """
    Create LangChain-compatible memory.
    
    Returns a LangChain memory adapter that can be used directly
    in LangChain applications.
    
    Example:
        ```python
        # Create LangChain-compatible memory
        langchain_memory = await create_langchain_memory(
            memory_type="conversation",
            backend_type="redis",
            session_id="user_123"
        )
        
        # Use in LangChain chain
        from langchain.chains import ConversationChain
        chain = ConversationChain(
            llm=llm,
            memory=langchain_memory
        )
        ```
    """
    from .langchain_integration import to_langchain_memory
    
    # Create LLMBlocks memory
    memory = await get_memory(memory_type, backend_type, session_id, **kwargs)
    
    # Convert to LangChain format
    return to_langchain_memory(memory, memory_key)


async def create_langchain_chat_history(
    memory_type: str = "conversation",
    backend_type: str = "in_memory",
    session_id: Optional[str] = None,
    **kwargs
):
    """
    Create LangChain-compatible chat message history.
    
    Example:
        ```python
        # Create chat history
        chat_history = await create_langchain_chat_history(
            backend_type="redis",
            session_id="chat_123"
        )
        
        # Use with LangChain
        from langchain.memory import ConversationBufferMemory
        memory = ConversationBufferMemory(
            chat_memory=chat_history,
            return_messages=True
        )
        ```
    """
    from .langchain_integration import to_langchain_chat_history
    
    # Create LLMBlocks memory
    memory = await get_memory(memory_type, backend_type, session_id, **kwargs)
    
    # Convert to LangChain format
    return to_langchain_chat_history(memory, session_id)
