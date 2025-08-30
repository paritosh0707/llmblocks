"""
Conversation Memory for LLMBlocks.

This module provides conversation-specific memory management,
including chat history, context windows, and conversation summarization.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, UTC

from pydantic import Field, ConfigDict
from .base import BaseMemory, MemoryConfig, MemoryMessage, MemoryRole, MemoryError
from .langchain_integration import LangChainCompatibilityMixin
from ...utils.logging import get_logger


class ConversationConfig(MemoryConfig):
    """Configuration for conversation memory."""
    model_config = ConfigDict(extra="allow")
    
    # Conversation-specific settings
    max_context_messages: int = Field(default=50, description="Maximum messages in context window")
    summarize_threshold: int = Field(default=100, description="Messages before summarization")
    include_system_messages: bool = Field(default=True, description="Include system messages in context")
    
    # Context management
    context_strategy: str = Field(default="sliding_window", description="Context management strategy")
    preserve_recent: int = Field(default=10, description="Always preserve N recent messages")
    
    # Summarization settings
    enable_summarization: bool = Field(default=False, description="Enable automatic summarization")
    summary_model: Optional[str] = Field(default=None, description="Model for summarization")
    summary_prompt: str = Field(
        default="Summarize the following conversation concisely, preserving key context:",
        description="Prompt for summarization"
    )


class ConversationMemory(BaseMemory, LangChainCompatibilityMixin):
    """
    Conversation memory system for chat applications.
    
    Features:
    - Sliding context window
    - Automatic summarization
    - Role-based message filtering
    - Context optimization
    - LangChain compatibility
    """
    
    def __init__(self, config: ConversationConfig):
        super().__init__(config)
        self.conversation_config = config
        self._summary: Optional[str] = None
        self._context_cache: Optional[List[MemoryMessage]] = None
        self._cache_timestamp: Optional[datetime] = None
    
    async def _initialize_impl(self) -> None:
        """Initialize the conversation memory implementation."""
        await self._initialize_backend()
    
    async def _initialize_backend(self) -> None:
        """Initialize the conversation memory backend."""
        from .backends import get_backend
        
        self.backend = await get_backend(
            self.config.backend_type,
            self.config.backend_config
        )
        await self.backend.initialize()
    
    # Conversation-specific methods
    async def add_user_message(self, content: str, **kwargs) -> MemoryMessage:
        """Add a user message to the conversation."""
        return await self.add_message(MemoryRole.USER, content, **kwargs)
    
    async def add_assistant_message(self, content: str, **kwargs) -> MemoryMessage:
        """Add an assistant message to the conversation."""
        return await self.add_message(MemoryRole.ASSISTANT, content, **kwargs)
    
    async def add_system_message(self, content: str, **kwargs) -> MemoryMessage:
        """Add a system message to the conversation."""
        return await self.add_message(MemoryRole.SYSTEM, content, **kwargs)
    
    async def get_conversation_history(
        self, 
        include_system: Optional[bool] = None,
        limit: Optional[int] = None
    ) -> List[MemoryMessage]:
        """Get conversation history with filtering."""
        if include_system is None:
            include_system = self.conversation_config.include_system_messages
        
        messages = await self.get_messages(limit=limit)
        
        if not include_system:
            messages = [
                msg for msg in messages 
                if msg.role != MemoryRole.SYSTEM
            ]
        
        return messages
    
    async def get_context_window(self) -> List[MemoryMessage]:
        """
        Get the current context window for the conversation.
        
        This applies the context strategy to return the most relevant
        messages for the current conversation state.
        """
        # Check cache
        if self._is_cache_valid():
            return self._context_cache
        
        # Get all messages
        all_messages = await self.get_conversation_history()
        
        # Apply context strategy
        context_messages = await self._apply_context_strategy(all_messages)
        
        # Update cache
        self._context_cache = context_messages
        self._cache_timestamp = datetime.now(UTC)
        
        return context_messages
    
    async def _apply_context_strategy(self, messages: List[MemoryMessage]) -> List[MemoryMessage]:
        """Apply the configured context strategy."""
        strategy = self.conversation_config.context_strategy
        max_messages = self.conversation_config.max_context_messages
        
        if strategy == "sliding_window":
            return await self._sliding_window_strategy(messages, max_messages)
        elif strategy == "summarized":
            return await self._summarized_strategy(messages, max_messages)
        elif strategy == "priority_based":
            return await self._priority_based_strategy(messages, max_messages)
        else:
            # Default to sliding window
            return await self._sliding_window_strategy(messages, max_messages)
    
    async def _sliding_window_strategy(
        self, 
        messages: List[MemoryMessage], 
        max_messages: int
    ) -> List[MemoryMessage]:
        """Simple sliding window - keep most recent messages."""
        if len(messages) <= max_messages:
            return messages
        
        return messages[-max_messages:]
    
    async def _summarized_strategy(
        self, 
        messages: List[MemoryMessage], 
        max_messages: int
    ) -> List[MemoryMessage]:
        """Summarized context - summarize old messages, keep recent ones."""
        if len(messages) <= max_messages:
            return messages
        
        preserve_recent = min(self.conversation_config.preserve_recent, max_messages - 1)  # Leave room for summary
        recent_messages = messages[-preserve_recent:]
        old_messages = messages[:-preserve_recent]
        
        # Generate summary if needed
        if old_messages and not self._summary:
            self._summary = await self._generate_summary(old_messages)
        
        # Create summary message
        context_messages = []
        if self._summary:
            summary_message = MemoryMessage(
                role=MemoryRole.SYSTEM,
                content=f"Previous conversation summary: {self._summary}",
                metadata={"type": "summary", "summarized_messages": len(old_messages)}
            )
            context_messages.append(summary_message)
        
        # Add recent messages (ensure we don't exceed max_messages)
        remaining_slots = max_messages - len(context_messages)
        context_messages.extend(recent_messages[-remaining_slots:])
        
        return context_messages
    
    async def _priority_based_strategy(
        self, 
        messages: List[MemoryMessage], 
        max_messages: int
    ) -> List[MemoryMessage]:
        """Priority-based context - keep important messages."""
        if len(messages) <= max_messages:
            return messages
        
        # Score messages by importance
        scored_messages = []
        for msg in messages:
            score = self._calculate_message_priority(msg)
            scored_messages.append((score, msg))
        
        # Sort by score (descending) and take top messages
        scored_messages.sort(key=lambda x: x[0], reverse=True)
        selected_messages = [msg for _, msg in scored_messages[:max_messages]]
        
        # Sort selected messages by timestamp
        selected_messages.sort(key=lambda x: x.timestamp)
        
        return selected_messages
    
    def _calculate_message_priority(self, message: MemoryMessage) -> float:
        """Calculate priority score for a message."""
        score = 0.0
        
        # Recent messages get higher scores
        age_hours = (datetime.now(UTC) - message.timestamp).total_seconds() / 3600
        recency_score = max(0, 10 - age_hours)  # Decay over 10 hours
        score += recency_score
        
        # System messages are important
        if message.role == MemoryRole.SYSTEM:
            score += 5.0
        
        # Messages with function calls are important
        if message.function_call or message.tool_calls:
            score += 3.0
        
        # Longer messages might be more important
        content_score = min(2.0, len(message.content) / 100)
        score += content_score
        
        # Check for important keywords
        important_keywords = ["error", "important", "remember", "note", "warning"]
        for keyword in important_keywords:
            if keyword.lower() in message.content.lower():
                score += 1.0
        
        return score
    
    async def _generate_summary(self, messages: List[MemoryMessage]) -> str:
        """Generate a summary of the given messages."""
        if not self.conversation_config.enable_summarization:
            return "Previous conversation context (summarization disabled)"
        
        # For now, create a simple summary
        # In a full implementation, this would use an LLM
        user_messages = [msg for msg in messages if msg.role == MemoryRole.USER]
        assistant_messages = [msg for msg in messages if msg.role == MemoryRole.ASSISTANT]
        
        summary_parts = []
        summary_parts.append(f"Conversation with {len(user_messages)} user messages and {len(assistant_messages)} assistant responses.")
        
        if user_messages:
            recent_topics = []
            for msg in user_messages[-3:]:  # Last 3 user messages
                if len(msg.content) > 20:
                    recent_topics.append(msg.content[:50] + "...")
                else:
                    recent_topics.append(msg.content)
            
            if recent_topics:
                summary_parts.append(f"Recent topics: {'; '.join(recent_topics)}")
        
        return " ".join(summary_parts)
    
    def _is_cache_valid(self) -> bool:
        """Check if the context cache is still valid."""
        if not self._context_cache or not self._cache_timestamp:
            return False
        
        # Cache is valid for 5 minutes
        cache_age = (datetime.now(UTC) - self._cache_timestamp).total_seconds()
        return cache_age < 300  # 5 minutes
    
    def _invalidate_cache(self) -> None:
        """Invalidate the context cache."""
        self._context_cache = None
        self._cache_timestamp = None
    
    async def add_message(self, role: Union[str, MemoryRole], content: str, **kwargs) -> MemoryMessage:
        """Override to invalidate cache when adding messages."""
        message = await super().add_message(role, content, **kwargs)
        self._invalidate_cache()
        
        # Check if we need to trigger summarization
        if self.conversation_config.enable_summarization:
            message_count = len(await self.get_messages())
            if message_count >= self.conversation_config.summarize_threshold:
                await self._trigger_summarization()
        
        return message
    
    async def _trigger_summarization(self) -> None:
        """Trigger summarization of old messages."""
        try:
            all_messages = await self.get_messages()
            preserve_recent = self.conversation_config.preserve_recent
            
            if len(all_messages) > preserve_recent:
                old_messages = all_messages[:-preserve_recent]
                self._summary = await self._generate_summary(old_messages)
                
                self.logger.info(
                    "Conversation summarized",
                    session_id=self.session_id,
                    summarized_messages=len(old_messages)
                )
        except Exception as e:
            self.logger.error(
                "Failed to generate summary",
                session_id=self.session_id,
                error=str(e)
            )
    
    # LangChain compatibility methods
    def to_langchain_format(self, messages: Optional[List[MemoryMessage]] = None) -> List[Dict[str, Any]]:
        """Convert messages to LangChain format."""
        if messages is None:
            # This would be async in real usage, but for compatibility we'll handle it
            import asyncio
            messages = asyncio.run(self.get_context_window())
        
        return [msg.to_langchain_format() for msg in messages]
    
    async def get_langchain_messages(self) -> List[Dict[str, Any]]:
        """Get messages in LangChain format (async version)."""
        messages = await self.get_context_window()
        return self.to_langchain_format(messages)
    
    # Statistics and debugging
    async def get_conversation_stats(self) -> Dict[str, Any]:
        """Get detailed conversation statistics."""
        stats = await self.get_stats()
        
        messages = await self.get_messages()
        role_counts = {}
        for msg in messages:
            role_counts[msg.role.value] = role_counts.get(msg.role.value, 0) + 1
        
        context_messages = await self.get_context_window()
        
        stats.update({
            "total_messages": len(messages),
            "context_messages": len(context_messages),
            "role_distribution": role_counts,
            "has_summary": self._summary is not None,
            "cache_valid": self._is_cache_valid(),
            "context_strategy": self.conversation_config.context_strategy,
            "max_context_messages": self.conversation_config.max_context_messages
        })
        
        return stats
