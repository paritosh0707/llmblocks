"""
LangChain and LangGraph integration for LLMBlocks Memory.

This module provides seamless compatibility with LangChain memory systems
and LangGraph state management, allowing LLMBlocks memory to be used as
drop-in replacements for LangChain memory components.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union, Sequence
from abc import ABC, abstractmethod

# LangChain imports
try:
    from langchain_core.memory import BaseMemory as LangChainBaseMemory
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
    from langchain.schema import BaseMessage as LegacyBaseMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback types for when LangChain is not available
    LangChainBaseMemory = object
    BaseMessage = object
    BaseChatMessageHistory = object
    Runnable = object
    LANGCHAIN_AVAILABLE = False

# LangGraph imports
try:
    from langgraph.graph import StateGraph, Graph
    from langgraph.checkpoint import BaseCheckpointSaver
    from langgraph.channels import Channel
    LANGGRAPH_AVAILABLE = True
except ImportError:
    StateGraph = object
    BaseCheckpointSaver = object
    LANGGRAPH_AVAILABLE = False

from .base import BaseMemory, MemoryMessage, MemoryRole
from ...utils.logging import get_logger


class LangChainMemoryAdapter(LangChainBaseMemory if LANGCHAIN_AVAILABLE else object):
    """
    Adapter to make LLMBlocks memory compatible with LangChain.
    
    This allows LLMBlocks memory systems to be used anywhere
    LangChain memory is expected.
    """
    
    def __init__(self, llmblocks_memory: BaseMemory, memory_key: str = "history"):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for LangChainMemoryAdapter")
        
        self.llmblocks_memory = llmblocks_memory
        self.memory_key = memory_key
        self.logger = get_logger("LangChainMemoryAdapter")
    
    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return [self.memory_key]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables synchronously."""
        # For sync compatibility, we need to run async code
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, we can't use run_until_complete
                # This is a limitation - ideally LangChain would be fully async
                self.logger.warning("Cannot load memory synchronously in async context")
                return {self.memory_key: ""}
            else:
                messages = loop.run_until_complete(self._load_messages())
                return {self.memory_key: self._format_messages(messages)}
        except Exception as e:
            self.logger.error(f"Failed to load memory: {e}")
            return {self.memory_key: ""}
    
    async def aload_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables asynchronously."""
        try:
            messages = await self._load_messages()
            return {self.memory_key: self._format_messages(messages)}
        except Exception as e:
            self.logger.error(f"Failed to load memory: {e}")
            return {self.memory_key: ""}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context synchronously."""
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop.run_until_complete(self._save_context(inputs, outputs))
        except Exception as e:
            self.logger.error(f"Failed to save context: {e}")
    
    async def asave_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context asynchronously."""
        await self._save_context(inputs, outputs)
    
    def clear(self) -> None:
        """Clear memory synchronously."""
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop.run_until_complete(self.llmblocks_memory.clear_memory())
        except Exception as e:
            self.logger.error(f"Failed to clear memory: {e}")
    
    async def aclear(self) -> None:
        """Clear memory asynchronously."""
        await self.llmblocks_memory.clear_memory()
    
    async def _load_messages(self) -> List[MemoryMessage]:
        """Load messages from LLMBlocks memory."""
        if hasattr(self.llmblocks_memory, 'get_context_window'):
            return await self.llmblocks_memory.get_context_window()
        else:
            return await self.llmblocks_memory.get_messages()
    
    def _format_messages(self, messages: List[MemoryMessage]) -> str:
        """Format messages as a string for LangChain compatibility."""
        formatted_lines = []
        for msg in messages:
            role_name = msg.role.value.title()
            formatted_lines.append(f"{role_name}: {msg.content}")
        return "\n".join(formatted_lines)
    
    async def _save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context to LLMBlocks memory."""
        # Extract user input
        user_input = inputs.get("input") or inputs.get("question") or inputs.get("human_input")
        if user_input:
            await self.llmblocks_memory.add_message(MemoryRole.USER, user_input)
        
        # Extract AI output
        ai_output = outputs.get("output") or outputs.get("answer") or outputs.get("response")
        if ai_output:
            await self.llmblocks_memory.add_message(MemoryRole.ASSISTANT, ai_output)


class LangChainChatMessageHistory(BaseChatMessageHistory if LANGCHAIN_AVAILABLE else object):
    """
    LangChain ChatMessageHistory implementation using LLMBlocks memory.
    
    This provides the BaseChatMessageHistory interface for LangChain
    applications while using LLMBlocks memory as the backend.
    """
    
    def __init__(self, llmblocks_memory: BaseMemory, session_id: Optional[str] = None):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for LangChainChatMessageHistory")
        
        self.llmblocks_memory = llmblocks_memory
        if session_id:
            self.llmblocks_memory.set_session_id(session_id)
        self.logger = get_logger("LangChainChatMessageHistory")
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Get messages as LangChain BaseMessage objects."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # In async context, return empty list (limitation)
                self.logger.warning("Cannot get messages synchronously in async context")
                return []
            else:
                llm_messages = loop.run_until_complete(self.llmblocks_memory.get_messages())
                return [self._to_langchain_message(msg) for msg in llm_messages]
        except Exception as e:
            self.logger.error(f"Failed to get messages: {e}")
            return []
    
    async def aget_messages(self) -> List[BaseMessage]:
        """Get messages asynchronously."""
        try:
            llm_messages = await self.llmblocks_memory.get_messages()
            return [self._to_langchain_message(msg) for msg in llm_messages]
        except Exception as e:
            self.logger.error(f"Failed to get messages: {e}")
            return []
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message synchronously."""
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop.run_until_complete(self._add_message(message))
        except Exception as e:
            self.logger.error(f"Failed to add message: {e}")
    
    async def aadd_message(self, message: BaseMessage) -> None:
        """Add a message asynchronously."""
        await self._add_message(message)
    
    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add multiple messages synchronously."""
        for message in messages:
            self.add_message(message)
    
    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add multiple messages asynchronously."""
        for message in messages:
            await self.aadd_message(message)
    
    def clear(self) -> None:
        """Clear all messages synchronously."""
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop.run_until_complete(self.llmblocks_memory.clear_memory())
        except Exception as e:
            self.logger.error(f"Failed to clear messages: {e}")
    
    async def aclear(self) -> None:
        """Clear all messages asynchronously."""
        await self.llmblocks_memory.clear_memory()
    
    def _to_langchain_message(self, msg: MemoryMessage) -> BaseMessage:
        """Convert LLMBlocks message to LangChain message."""
        content = msg.content
        additional_kwargs = {
            "timestamp": msg.timestamp.isoformat(),
            "message_id": msg.message_id,
            **msg.metadata
        }
        
        if msg.role == MemoryRole.USER:
            return HumanMessage(content=content, additional_kwargs=additional_kwargs)
        elif msg.role == MemoryRole.ASSISTANT:
            return AIMessage(content=content, additional_kwargs=additional_kwargs)
        elif msg.role == MemoryRole.SYSTEM:
            return SystemMessage(content=content, additional_kwargs=additional_kwargs)
        elif msg.role == MemoryRole.FUNCTION:
            return FunctionMessage(content=content, name=msg.metadata.get("function_name", "unknown"))
        else:
            # Default to HumanMessage for unknown roles
            return HumanMessage(content=content, additional_kwargs=additional_kwargs)
    
    async def _add_message(self, message: BaseMessage) -> None:
        """Add a LangChain message to LLMBlocks memory."""
        # Convert LangChain message to LLMBlocks format
        if isinstance(message, HumanMessage):
            role = MemoryRole.USER
        elif isinstance(message, AIMessage):
            role = MemoryRole.ASSISTANT
        elif isinstance(message, SystemMessage):
            role = MemoryRole.SYSTEM
        elif isinstance(message, FunctionMessage):
            role = MemoryRole.FUNCTION
        else:
            role = MemoryRole.USER  # Default
        
        metadata = getattr(message, 'additional_kwargs', {})
        await self.llmblocks_memory.add_message(role, message.content, metadata=metadata)


class LangGraphCheckpointSaver(BaseCheckpointSaver if LANGGRAPH_AVAILABLE else object):
    """
    LangGraph checkpoint saver using LLMBlocks memory as backend.
    
    This allows LangGraph workflows to persist state using
    LLMBlocks memory systems.
    """
    
    def __init__(self, llmblocks_memory: BaseMemory):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph is required for LangGraphCheckpointSaver")
        
        self.llmblocks_memory = llmblocks_memory
        self.logger = get_logger("LangGraphCheckpointSaver")
    
    async def aget_tuple(self, config: RunnableConfig) -> Optional[Dict[str, Any]]:
        """Get checkpoint tuple."""
        try:
            session_id = config.get("configurable", {}).get("thread_id", "default")
            self.llmblocks_memory.set_session_id(f"langgraph_{session_id}")
            
            # Get the latest state message
            messages = await self.llmblocks_memory.get_messages(limit=1)
            if messages and messages[0].metadata.get("type") == "checkpoint":
                return {
                    "checkpoint": messages[0].metadata.get("checkpoint"),
                    "metadata": messages[0].metadata.get("checkpoint_metadata", {}),
                    "config": config
                }
            return None
        except Exception as e:
            self.logger.error(f"Failed to get checkpoint: {e}")
            return None
    
    async def aput_tuple(self, config: RunnableConfig, checkpoint: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """Save checkpoint tuple."""
        try:
            session_id = config.get("configurable", {}).get("thread_id", "default")
            self.llmblocks_memory.set_session_id(f"langgraph_{session_id}")
            
            # Save checkpoint as a special message
            await self.llmblocks_memory.add_message(
                MemoryRole.SYSTEM,
                "LangGraph checkpoint",
                metadata={
                    "type": "checkpoint",
                    "checkpoint": checkpoint,
                    "checkpoint_metadata": metadata
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")


# Utility functions for easy integration
def to_langchain_memory(llmblocks_memory: BaseMemory, memory_key: str = "history") -> LangChainMemoryAdapter:
    """Convert LLMBlocks memory to LangChain memory."""
    return LangChainMemoryAdapter(llmblocks_memory, memory_key)


def to_langchain_chat_history(llmblocks_memory: BaseMemory, session_id: Optional[str] = None) -> LangChainChatMessageHistory:
    """Convert LLMBlocks memory to LangChain chat message history."""
    return LangChainChatMessageHistory(llmblocks_memory, session_id)


def to_langgraph_checkpoint_saver(llmblocks_memory: BaseMemory) -> LangGraphCheckpointSaver:
    """Convert LLMBlocks memory to LangGraph checkpoint saver."""
    return LangGraphCheckpointSaver(llmblocks_memory)


async def from_langchain_memory(langchain_memory: LangChainBaseMemory, backend_type: str = "in_memory"):
    """
    Create LLMBlocks memory from existing LangChain memory.
    
    This is useful for migrating from LangChain memory to LLMBlocks.
    """
    # Import here to avoid circular dependency
    from .factory import get_conversation_memory
    
    # Create LLMBlocks memory
    memory = await get_conversation_memory(backend_type=backend_type)

    
    # Try to extract existing messages if possible
    try:
        if hasattr(langchain_memory, 'chat_memory') and hasattr(langchain_memory.chat_memory, 'messages'):
            for msg in langchain_memory.chat_memory.messages:
                if hasattr(msg, 'content'):
                    if isinstance(msg, HumanMessage):
                        await memory.add_user_message(msg.content)
                    elif isinstance(msg, AIMessage):
                        await memory.add_assistant_message(msg.content)
                    elif isinstance(msg, SystemMessage):
                        await memory.add_system_message(msg.content)
    except Exception as e:
        get_logger("from_langchain_memory").warning(f"Could not migrate existing messages: {e}")
    
    return memory


# LangChain Runnable integration
class LLMBlocksMemoryRunnable(Runnable if LANGCHAIN_AVAILABLE else object):
    """
    Make LLMBlocks memory work as a LangChain Runnable.
    
    This allows memory to be used in LangChain Expression Language (LCEL) chains.
    """
    
    def __init__(self, llmblocks_memory: BaseMemory):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for LLMBlocksMemoryRunnable")
        
        self.llmblocks_memory = llmblocks_memory
        self.logger = get_logger("LLMBlocksMemoryRunnable")
    
    async def ainvoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Invoke the memory runnable."""
        try:
            # Load memory context
            if hasattr(self.llmblocks_memory, 'get_context_window'):
                messages = await self.llmblocks_memory.get_context_window()
            else:
                messages = await self.llmblocks_memory.get_messages()
            
            # Format for output
            formatted_history = []
            for msg in messages:
                formatted_history.append({
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                })
            
            return {
                **input,
                "memory_context": formatted_history,
                "memory_summary": f"{len(messages)} messages in context"
            }
        except Exception as e:
            self.logger.error(f"Failed to invoke memory runnable: {e}")
            return {**input, "memory_context": [], "memory_summary": "Error loading memory"}
    
    def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Synchronous invoke (runs async version)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new event loop for sync execution
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.ainvoke(input, config))
                    return future.result()
            else:
                return loop.run_until_complete(self.ainvoke(input, config))
        except Exception as e:
            self.logger.error(f"Failed to invoke memory runnable synchronously: {e}")
            return {**input, "memory_context": [], "memory_summary": "Error loading memory"}


def create_memory_runnable(llmblocks_memory: BaseMemory) -> LLMBlocksMemoryRunnable:
    """Create a LangChain Runnable from LLMBlocks memory."""
    return LLMBlocksMemoryRunnable(llmblocks_memory)


# Integration helpers
class LangChainCompatibilityMixin:
    """
    Mixin to add LangChain compatibility methods to memory classes.
    
    Add this to any LLMBlocks memory class to get LangChain integration.
    """
    
    def to_langchain_memory(self, memory_key: str = "history") -> LangChainMemoryAdapter:
        """Convert to LangChain memory."""
        return to_langchain_memory(self, memory_key)
    
    def to_langchain_chat_history(self, session_id: Optional[str] = None) -> LangChainChatMessageHistory:
        """Convert to LangChain chat message history."""
        return to_langchain_chat_history(self, session_id)
    
    def to_langgraph_checkpoint_saver(self) -> LangGraphCheckpointSaver:
        """Convert to LangGraph checkpoint saver."""
        return to_langgraph_checkpoint_saver(self)
    
    def to_runnable(self) -> LLMBlocksMemoryRunnable:
        """Convert to LangChain Runnable."""
        return create_memory_runnable(self)


# Compatibility checks
def check_langchain_compatibility() -> Dict[str, bool]:
    """Check what LangChain/LangGraph features are available."""
    return {
        "langchain_available": LANGCHAIN_AVAILABLE,
        "langgraph_available": LANGGRAPH_AVAILABLE,
        "memory_adapter": LANGCHAIN_AVAILABLE,
        "chat_history": LANGCHAIN_AVAILABLE,
        "checkpoint_saver": LANGGRAPH_AVAILABLE,
        "runnable": LANGCHAIN_AVAILABLE
    }
