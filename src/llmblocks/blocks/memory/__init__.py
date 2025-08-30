"""
Memory and State Management for LLMBlocks.

This module provides conversation memory, state persistence, and various
storage backends for maintaining context across interactions.

Key Components:
- BaseMemory: Abstract memory interface
- ConversationMemory: Chat history management
- StateManager: Application state persistence
- Memory Backends: In-memory, file, Redis, database storage
- LangChain Integration: Compatible with LangChain memory systems

Example:
    ```python
    from llmblocks.blocks.memory import ConversationMemory, get_memory
    
    # Simple conversation memory
    memory = await get_memory("conversation", backend="in_memory")
    await memory.add_message("user", "Hello!")
    await memory.add_message("assistant", "Hi there!")
    
    # Get conversation history
    history = await memory.get_messages()
    print(f"Conversation has {len(history)} messages")
    ```
"""

from .base import (
    BaseMemory,
    MemoryConfig,
    MemoryMessage,
    MemoryRole,
    MemoryBackend,
    MemoryError
)

from .conversation import (
    ConversationMemory,
    ConversationConfig
)

# State management will be implemented in future versions
# from .state import (
#     StateManager,
#     StateConfig
# )

from .backends import (
    InMemoryBackend,
    FileBackend,
    RedisBackend
)

from .factory import (
    MemoryFactory,
    get_memory,
    get_conversation_memory,
    list_memory_types
)

# LangChain integration
from .langchain_integration import (
    LangChainMemoryAdapter,
    to_langchain_memory,
    from_langchain_memory
)

__all__ = [
    # Core interfaces
    "BaseMemory",
    "MemoryConfig", 
    "MemoryMessage",
    "MemoryRole",
    "MemoryBackend",
    "MemoryError",
    
    # Memory types
    "ConversationMemory",
    "ConversationConfig",
    # "StateManager",
    # "StateConfig",
    
    # Backends
    "InMemoryBackend",
    "FileBackend", 
    "RedisBackend",
    
    # Factory and utilities
    "MemoryFactory",
    "get_memory",
    "get_conversation_memory",
    "list_memory_types",
    
    # LangChain integration
    "LangChainMemoryAdapter",
    "to_langchain_memory",
    "from_langchain_memory"
]