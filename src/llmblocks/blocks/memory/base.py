"""
Base memory interfaces and data structures for LLMBlocks.

This module defines the core abstractions for memory and state management,
providing a consistent interface across different storage backends.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
import uuid

from pydantic import BaseModel, Field, ConfigDict
from ...core.base_block import BaseBlock, BlockConfig, BlockStatus
from ...utils.exceptions import LLMBlocksError
from ...utils.logging import get_logger


class MemoryError(LLMBlocksError):
    """Base exception for memory-related errors."""
    pass


class MemoryBackendError(MemoryError):
    """Exception for memory backend errors."""
    pass


class MemoryRole(Enum):
    """Roles for memory messages."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    TOOL = "tool"


@dataclass
class MemoryMessage:
    """A message stored in memory."""
    role: MemoryRole
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Function/tool calling support
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "metadata": self.metadata,
            "function_call": self.function_call,
            "tool_calls": self.tool_calls
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryMessage":
        """Create message from dictionary."""
        return cls(
            role=MemoryRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message_id=data["message_id"],
            metadata=data.get("metadata", {}),
            function_call=data.get("function_call"),
            tool_calls=data.get("tool_calls")
        )
    
    def to_langchain_format(self) -> Dict[str, Any]:
        """Convert to LangChain message format."""
        return {
            "role": self.role.value,
            "content": self.content,
            "additional_kwargs": {
                "timestamp": self.timestamp.isoformat(),
                "message_id": self.message_id,
                **self.metadata
            }
        }


class MemoryConfig(BlockConfig):
    """Base configuration for memory systems."""
    model_config = ConfigDict(extra="allow")
    
    # Memory settings
    max_messages: Optional[int] = Field(default=None, description="Maximum messages to store")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to store")
    ttl_seconds: Optional[int] = Field(default=None, description="Time-to-live for messages")
    
    # Backend settings
    backend_type: str = Field(default="in_memory", description="Memory backend type")
    backend_config: Dict[str, Any] = Field(default_factory=dict, description="Backend-specific config")
    
    # Serialization settings
    compression: bool = Field(default=False, description="Enable compression")
    encryption: bool = Field(default=False, description="Enable encryption")
    encryption_key: Optional[str] = Field(default=None, description="Encryption key")


class MemoryBackend(ABC):
    """Abstract base class for memory storage backends."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"{self.__class__.__name__}")
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the backend and cleanup resources."""
        pass
    
    @abstractmethod
    async def store_message(self, session_id: str, message: MemoryMessage) -> None:
        """Store a message in the backend."""
        pass
    
    @abstractmethod
    async def get_messages(
        self, 
        session_id: str, 
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[MemoryMessage]:
        """Retrieve messages from the backend."""
        pass
    
    @abstractmethod
    async def delete_messages(self, session_id: str, message_ids: List[str]) -> int:
        """Delete specific messages. Returns number of deleted messages."""
        pass
    
    @abstractmethod
    async def clear_session(self, session_id: str) -> int:
        """Clear all messages for a session. Returns number of deleted messages."""
        pass
    
    @abstractmethod
    async def get_sessions(self) -> List[str]:
        """Get list of all session IDs."""
        pass
    
    @abstractmethod
    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        pass
    
    @abstractmethod
    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        pass
    
    # Optional methods for advanced backends
    async def search_messages(
        self, 
        session_id: str, 
        query: str, 
        limit: int = 10
    ) -> List[MemoryMessage]:
        """Search messages by content (optional)."""
        # Default implementation: simple text search
        messages = await self.get_messages(session_id)
        results = []
        
        for message in messages:
            if query.lower() in message.content.lower():
                results.append(message)
                if len(results) >= limit:
                    break
        
        return results
    
    async def get_message_by_id(self, session_id: str, message_id: str) -> Optional[MemoryMessage]:
        """Get a specific message by ID (optional)."""
        messages = await self.get_messages(session_id)
        for message in messages:
            if message.message_id == message_id:
                return message
        return None


class BaseMemory(BaseBlock):
    """
    Base class for all memory systems in LLMBlocks.
    
    Provides core memory functionality including message storage,
    retrieval, and session management across different backends.
    """
    
    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self.config = config
        self.backend: Optional[MemoryBackend] = None
        self.session_id: str = str(uuid.uuid4())
        self.logger = get_logger(f"{self.__class__.__name__}")
    
    async def initialize(self) -> None:
        """Initialize the memory system."""
        if self.status != BlockStatus.UNINITIALIZED:
            return
        
        try:
            self.status = BlockStatus.INITIALIZING
            
            # Initialize backend
            await self._initialize_backend()
            
            self.status = BlockStatus.READY
            self.logger.info("Memory system initialized", session_id=self.session_id)
            
        except Exception as e:
            self.status = BlockStatus.ERROR
            raise MemoryError(f"Failed to initialize memory: {e}") from e
    
    async def close(self) -> None:
        """Close the memory system."""
        if self.status in [BlockStatus.STOPPED, BlockStatus.UNINITIALIZED]:
            return
        
        try:
            self.status = BlockStatus.STOPPING
            
            if self.backend:
                await self.backend.close()
            
            self.status = BlockStatus.STOPPED
            self.logger.info("Memory system closed")
            
        except Exception as e:
            self.status = BlockStatus.ERROR
            raise MemoryError(f"Failed to close memory: {e}") from e
    
    @abstractmethod
    async def _initialize_backend(self) -> None:
        """Initialize the specific backend implementation."""
        pass
    
    # Core memory operations
    async def add_message(
        self, 
        role: Union[str, MemoryRole], 
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> MemoryMessage:
        """Add a message to memory."""
        if self.status != BlockStatus.READY:
            raise MemoryError("Memory system not ready")
        
        # Convert role if needed
        if isinstance(role, str):
            role = MemoryRole(role)
        
        # Create message
        message = MemoryMessage(
            role=role,
            content=content,
            metadata=metadata or {},
            **kwargs
        )
        
        # Store in backend
        await self.backend.store_message(self.session_id, message)
        
        self.logger.debug(
            "Message added to memory",
            session_id=self.session_id,
            message_id=message.message_id,
            role=role.value
        )
        
        return message
    
    async def get_messages(
        self, 
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[MemoryMessage]:
        """Get messages from memory."""
        if self.status != BlockStatus.READY:
            raise MemoryError("Memory system not ready")
        
        return await self.backend.get_messages(self.session_id, limit, offset)
    
    async def clear_memory(self) -> int:
        """Clear all messages from current session."""
        if self.status != BlockStatus.READY:
            raise MemoryError("Memory system not ready")
        
        count = await self.backend.clear_session(self.session_id)
        self.logger.info("Memory cleared", session_id=self.session_id, messages_deleted=count)
        return count
    
    async def delete_messages(self, message_ids: List[str]) -> int:
        """Delete specific messages."""
        if self.status != BlockStatus.READY:
            raise MemoryError("Memory system not ready")
        
        count = await self.backend.delete_messages(self.session_id, message_ids)
        self.logger.info(
            "Messages deleted", 
            session_id=self.session_id, 
            messages_deleted=count
        )
        return count
    
    async def search_messages(self, query: str, limit: int = 10) -> List[MemoryMessage]:
        """Search messages by content."""
        if self.status != BlockStatus.READY:
            raise MemoryError("Memory system not ready")
        
        return await self.backend.search_messages(self.session_id, query, limit)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if self.status != BlockStatus.READY:
            raise MemoryError("Memory system not ready")
        
        stats = await self.backend.get_session_stats(self.session_id)
        stats.update({
            "session_id": self.session_id,
            "backend_type": self.config.backend_type,
            "status": self.status.value
        })
        return stats
    
    # Session management
    def set_session_id(self, session_id: str) -> None:
        """Set the current session ID."""
        self.session_id = session_id
        self.logger.debug("Session ID changed", new_session_id=session_id)
    
    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self.session_id
    
    async def list_sessions(self) -> List[str]:
        """List all available sessions."""
        if self.status != BlockStatus.READY:
            raise MemoryError("Memory system not ready")
        
        return await self.backend.get_sessions()
    
    # Context manager support
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    # String representation
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"session_id='{self.session_id}', "
            f"backend='{self.config.backend_type}', "
            f"status='{self.status.value}'"
            f")"
        )
