"""
Base LLM Provider for LLMBlocks.

This module provides the base interface and configuration for all LLM providers
with LangChain/LangGraph integration, async support, streaming, and comprehensive error handling.
"""

import asyncio
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
import uuid

from pydantic import BaseModel, Field, SecretStr
from tenacity import retry, stop_after_attempt, wait_exponential

# LangChain imports
from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.messages import (
    BaseMessage, 
    HumanMessage, 
    AIMessage, 
    SystemMessage,
    FunctionMessage,
    ToolMessage
)
from langchain_core.outputs import ChatResult, LLMResult
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

# LangGraph imports (optional - will be imported when needed)
# from langgraph.graph import StateGraph, MessagesState
# from langgraph.checkpoint.memory import MemorySaver

from ...core.base_block import BlockType, BlockConfig, BlockStatus, BlockMetadata
from ...utils.logging import get_logger, log_performance
from ...utils.exceptions import (
    LLMProviderError,
    LLMConnectionError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMResponseError,
    ErrorCodes
)


class LLMRole(Enum):
    """Message roles for LLM conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


@dataclass
class LLMMessage:
    """Message in an LLM conversation - compatible with LangChain messages."""
    role: LLMRole
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_langchain_message(self) -> BaseMessage:
        """Convert to LangChain message format."""
        if self.role == LLMRole.SYSTEM:
            return SystemMessage(content=self.content, additional_kwargs=self.metadata)
        elif self.role == LLMRole.USER:
            return HumanMessage(content=self.content, additional_kwargs=self.metadata)
        elif self.role == LLMRole.ASSISTANT:
            kwargs = self.metadata.copy()
            if self.function_call:
                kwargs["function_call"] = self.function_call
            if self.tool_calls:
                kwargs["tool_calls"] = self.tool_calls
            return AIMessage(content=self.content, additional_kwargs=kwargs)
        elif self.role == LLMRole.FUNCTION:
            return FunctionMessage(content=self.content, name=self.name or "")
        elif self.role == LLMRole.TOOL:
            return ToolMessage(content=self.content, tool_call_id=self.metadata.get("tool_call_id", ""))
        else:
            return HumanMessage(content=self.content, additional_kwargs=self.metadata)
    
    @classmethod
    def from_langchain_message(cls, message: BaseMessage) -> "LLMMessage":
        """Create from LangChain message format."""
        if isinstance(message, SystemMessage):
            role = LLMRole.SYSTEM
        elif isinstance(message, HumanMessage):
            role = LLMRole.USER
        elif isinstance(message, AIMessage):
            role = LLMRole.ASSISTANT
        elif isinstance(message, FunctionMessage):
            role = LLMRole.FUNCTION
        elif isinstance(message, ToolMessage):
            role = LLMRole.TOOL
        else:
            role = LLMRole.USER
        
        # Extract additional data
        kwargs = getattr(message, 'additional_kwargs', {})
        function_call = kwargs.pop('function_call', None)
        tool_calls = kwargs.pop('tool_calls', None)
        
        return cls(
            role=role,
            content=message.content,
            name=getattr(message, 'name', None),
            function_call=function_call,
            tool_calls=tool_calls,
            metadata=kwargs
        )


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    role: LLMRole = LLMRole.ASSISTANT
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Function/tool calling support
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class LLMProviderConfig(BlockConfig):
    """Base configuration for LLM providers."""
    
    # Provider identification
    provider_name: str
    model: str
    
    # Authentication
    api_key: Optional[Union[str, SecretStr]] = None
    api_base: Optional[str] = None
    
    # Generation parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[List[str]] = None
    
    # Streaming and async
    stream: bool = False
    async_mode: bool = True
    
    # Connection and retry settings
    timeout: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=1.0, gt=0)
    connection_pool_size: int = Field(default=10, gt=0)
    
    # Rate limiting
    requests_per_minute: Optional[int] = Field(default=None, gt=0)
    tokens_per_minute: Optional[int] = Field(default=None, gt=0)
    
    # Function/tool calling
    functions: Optional[List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


class BaseLLMProvider(BaseChatModel):
    """
    Base class for all LLM providers - extends LangChain's BaseChatModel.
    
    This class provides:
    - LangChain/LangGraph compatibility
    - Unified interface for different LLM providers
    - Async support with connection pooling
    - Streaming response handling
    - Comprehensive error handling and retry logic
    - Rate limiting and quota management
    - Function/tool calling support
    """
    
    def __init__(self, config: Union[LLMProviderConfig, Dict[str, Any]], **kwargs):
        """
        Initialize the LLM provider.
        
        Args:
            config: Provider configuration
            **kwargs: Additional configuration parameters
        """
        # Validate and convert config
        if isinstance(config, dict):
            config = LLMProviderConfig(**config, **kwargs)
        elif kwargs:
            config_dict = config.model_dump()
            config_dict.update(kwargs)
            config = LLMProviderConfig(**config_dict)
        
        # Initialize LangChain BaseChatModel
        super().__init__()
        
        self._provider_config = config
        self._logger = get_logger(f"{self.__class__.__name__}")
        
        # BaseBlock-like attributes (stored as private to avoid Pydantic conflicts)
        self._block_id = str(uuid.uuid4())
        self._block_type = BlockType.LLM_PROVIDER
        self._status = BlockStatus.UNINITIALIZED
        self._created_at = datetime.now(UTC)
        self._updated_at = self._created_at
        
        # Initialize metadata as a regular dict attribute for LangChain compatibility
        self.metadata = {
            "block_id": self._block_id,
            "provider": self.provider_name,
            "model": self.model_name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
        
        # Connection management
        self._client = None
        self._async_client = None
        self._connection_pool = None
        
        # Rate limiting
        self._request_times: List[datetime] = []
        self._token_usage: List[tuple[datetime, int]] = []
        self._rate_limit_lock = asyncio.Lock()
        
        # Metrics
        self._total_requests = 0
        self._total_tokens = 0
        self._total_errors = 0
        self._average_response_time = 0.0
    
    # BaseBlock-like properties
    @property
    def block_id(self) -> str:
        """Get the block ID."""
        return self._block_id
    
    @property
    def block_type(self) -> BlockType:
        """Get the block type."""
        return self._block_type
    
    @property
    def status(self) -> BlockStatus:
        """Get the current status."""
        return self._status
    
    @status.setter
    def status(self, value: BlockStatus) -> None:
        """Set the current status."""
        self._status = value
        self._updated_at = datetime.now(UTC)
        # Update metadata dict
        if hasattr(self, 'metadata'):
            self.update_metadata()
    
    @property
    def created_at(self) -> datetime:
        """Get the creation timestamp."""
        return self._created_at
    
    @property
    def updated_at(self) -> datetime:
        """Get the last update timestamp."""
        return self._updated_at
    
    @property
    def provider_config(self) -> LLMProviderConfig:
        """Get the provider configuration."""
        return self._provider_config
    
    @property
    def logger(self):
        """Get the logger instance."""
        return self._logger
    
    def update_metadata(self) -> None:
        """Update the metadata dict with current values."""
        self.metadata.update({
            "status": self.status.value,
            "updated_at": self.updated_at.isoformat()
        })
    
    @property
    def is_ready(self) -> bool:
        """Check if the provider is ready for use."""
        return self.status in [BlockStatus.READY, BlockStatus.RUNNING]
    
    # BaseBlock-like lifecycle methods
    async def initialize(self) -> None:
        """Initialize the provider."""
        try:
            self.status = BlockStatus.INITIALIZING
            await self._initialize_clients()
            self.status = BlockStatus.READY
            self.logger.info("Provider initialized successfully")
        except Exception as e:
            self.status = BlockStatus.ERROR
            self.logger.error(f"Provider initialization failed: {e}")
            raise
    
    async def start(self) -> None:
        """Start the provider (if not already running)."""
        if self.status == BlockStatus.READY:
            self.status = BlockStatus.RUNNING
            self.logger.info("Provider started")
    
    async def stop(self) -> None:
        """Stop the provider."""
        if self.status == BlockStatus.RUNNING:
            self.status = BlockStatus.STOPPING
            await self.cleanup()
            self.status = BlockStatus.STOPPED
            self.logger.info("Provider stopped")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            await self._close_async_client()
            self._close_sync_client()
            self.logger.info("Provider cleanup completed")
        except Exception as e:
            self.logger.error(f"Provider cleanup failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            if hasattr(self, '_test_connection'):
                await self._test_connection()
            
            return {
                "is_healthy": True,
                "status": self.status.value,
                "provider": self.provider_name,
                "model": self.model_name,
                "total_requests": self._total_requests,
                "total_errors": self._total_errors,
                "uptime": (datetime.now(UTC) - self.created_at).total_seconds()
            }
        except Exception as e:
            return {
                "is_healthy": False,
                "status": self.status.value,
                "error": str(e),
                "provider": self.provider_name
            }
    
    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self.provider_config.provider_name
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.provider_config.model
    
    @property
    def is_streaming_enabled(self) -> bool:
        """Check if streaming is enabled."""
        return self.provider_config.stream
    
    @property
    def provider_stats(self) -> Dict[str, Any]:
        """Get provider statistics."""
        return {
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_errors": self._total_errors,
            "average_response_time": self._average_response_time,
            "is_streaming_enabled": self.is_streaming_enabled,
            "connection_pool_size": self.provider_config.connection_pool_size
        }
    
    async def _initialize_impl(self) -> None:
        """Initialize the provider implementation."""
        try:
            # Initialize clients
            await self._initialize_clients()
            
            # Test connection
            await self._test_connection()
            
            self.logger.info(
                "LLM provider initialized",
                provider=self.provider_name,
                model=self.model_name
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to initialize LLM provider",
                provider=self.provider_name,
                error=str(e)
            )
            raise
    
    async def _cleanup_impl(self) -> None:
        """Clean up provider resources."""
        try:
            if self._async_client:
                await self._close_async_client()
            
            if self._client:
                await self._close_client()
            
            self.logger.info(
                "LLM provider cleaned up",
                provider=self.provider_name
            )
            
        except Exception as e:
            self.logger.error(
                "Error during provider cleanup",
                provider=self.provider_name,
                error=str(e)
            )
    
    async def _health_check_impl(self) -> Dict[str, Any]:
        """Perform provider-specific health check."""
        health = {
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "client_initialized": self._client is not None,
            "async_client_initialized": self._async_client is not None,
        }
        
        try:
            # Test a simple request
            test_response = await self.generate(
                messages=[LLMMessage(role=LLMRole.USER, content="Hello")],
                max_tokens=5
            )
            health["test_request_successful"] = True
            health["test_response_length"] = len(test_response.content)
            
        except Exception as e:
            health["test_request_successful"] = False
            health["test_error"] = str(e)
        
        return health
    
    # LangChain compatibility methods
    
    @property
    def _llm_type(self) -> str:
        """Return type of language model."""
        return self.provider_name
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat result synchronously (required by LangChain)."""
        # Convert to async and run
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, we need to handle this differently
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self._agenerate_helper(messages, stop, **kwargs))
                return future.result()
        else:
            return loop.run_until_complete(self._agenerate_helper(messages, stop, **kwargs))
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat result asynchronously (required by LangChain)."""
        return await self._agenerate_helper(messages, stop, **kwargs)
    
    async def _agenerate_helper(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Helper method for async generation."""
        # Convert LangChain messages to our format
        llm_messages = [LLMMessage.from_langchain_message(msg) for msg in messages]
        
        # Add stop sequences to kwargs
        if stop:
            kwargs["stop"] = stop
        
        # Generate response using our method
        response = await self.generate(llm_messages, **kwargs)
        
        # Convert back to LangChain format
        ai_message = AIMessage(
            content=response.content,
            additional_kwargs={
                "function_call": response.function_call,
                "tool_calls": response.tool_calls,
                "finish_reason": response.finish_reason,
                "usage": response.usage,
                "model": response.model,
                "provider": response.provider,
                "request_id": response.request_id,
                **response.metadata
            }
        )
        
        from langchain_core.outputs import ChatGeneration
        chat_generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[chat_generation])
    
    def get_langchain_runnable(self) -> Runnable:
        """Get a LangChain Runnable interface for this provider."""
        return self
    
    def create_langgraph_node(self, node_name: str = "llm") -> Callable:
        """Create a LangGraph node function for this provider."""
        from ...utils.langgraph_compat import warn_about_compatibility
        
        # Issue compatibility warnings if needed
        warn_about_compatibility()
        
        # Use MRO-safe approach that avoids importing MessagesState directly
        # This bypasses the known MRO conflict in certain LangGraph versions
        
        async def llm_node(state) -> Dict[str, Any]:
            """LangGraph node that processes messages using this LLM provider."""
            self.logger.debug(f"LangGraph node '{node_name}' processing state: {type(state)}")
            
            # Handle different state formats (dict, MessagesState, or custom objects)
            if hasattr(state, 'get') and callable(getattr(state, 'get')):
                # Dict-like state
                messages = state.get("messages", [])
            elif hasattr(state, 'messages'):
                # MessagesState-like object
                messages = state.messages
            else:
                # Fallback: assume it's iterable or has messages attribute
                try:
                    messages = list(state) if hasattr(state, '__iter__') else []
                except (TypeError, AttributeError):
                    messages = []
            
            if not messages:
                self.logger.debug(f"No messages found in state, returning empty")
                return {"messages": []}
            
            self.logger.debug(f"Processing {len(messages)} messages")
            
            # Generate response using our provider
            result = await self._agenerate(messages)
            
            # Return the new message
            new_message = result.generations[0].message
            self.logger.debug(f"Generated response: {new_message.content[:100]}...")
            
            return {"messages": [new_message]}
        
        # Set the function name for debugging
        llm_node.__name__ = node_name
        
        return llm_node
    
    def create_langgraph_graph(self, node_name: str = "llm"):
        """Create a complete LangGraph graph with this provider."""
        from ...utils.langgraph_compat import create_compatible_graph
        
        llm_node = self.create_langgraph_node(node_name)
        success, graph, error = create_compatible_graph(llm_node, node_name)
        
        if success:
            self.logger.info(f"Created LangGraph graph with node '{node_name}'")
        else:
            self.logger.warning(f"Using fallback graph implementation: {error}")
        
        return graph
    
    # Abstract methods that providers must implement
    @abstractmethod
    async def _initialize_clients(self) -> None:
        """Initialize provider-specific clients."""
        pass
    
    @abstractmethod
    async def _close_async_client(self) -> None:
        """Close asynchronous client."""
        pass
    
    @abstractmethod
    def _close_sync_client(self) -> None:
        """Close synchronous client."""
        pass
    
    # Main API methods (our custom interface)
    
    @log_performance("llm_generate")
    async def generate(
        self,
        messages: Union[str, Dict[str, str], List[Union[str, Dict[str, str], LLMMessage]]],
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: Can be:
                - str: Single user message (e.g., "Hello!")
                - dict: Single message (e.g., {"role": "user", "content": "Hello!"})
                - List: Multiple messages in any format above or LLMMessage objects
            **kwargs: Additional generation parameters
            
        Returns:
            LLM response
            
        Raises:
            LLMProviderError: If generation fails
        """
        if not self.is_ready:
            raise LLMProviderError("Provider is not ready")
        
        # Apply rate limiting
        await self._apply_rate_limiting()
        
        # Convert input to LLMMessage format
        llm_messages = self._normalize_messages(messages)
        
        # Merge generation parameters
        generation_params = self._merge_generation_params(**kwargs)
        
        try:
            # Track request
            self._total_requests += 1
            start_time = datetime.now(UTC)
            
            # Generate response
            response = await self._generate_impl(llm_messages, **generation_params)
            
            # Update metrics
            end_time = datetime.now(UTC)
            response_time = (end_time - start_time).total_seconds()
            self._update_metrics(response, response_time)
            
            self.logger.info(
                "Generated LLM response",
                provider=self.provider_name,
                model=self.model_name,
                response_time=response_time,
                tokens_used=response.usage.get("total_tokens", 0) if response.usage else 0
            )
            
            return response
            
        except Exception as e:
            self._total_errors += 1
            self.logger.error(
                "Failed to generate LLM response",
                provider=self.provider_name,
                error=str(e)
            )
            
            # Convert to appropriate exception type
            if "rate limit" in str(e).lower():
                raise LLMRateLimitError(
                    f"Rate limit exceeded: {e}",
                    error_code=ErrorCodes.LLM_RATE_LIMIT
                ) from e
            elif "timeout" in str(e).lower():
                raise LLMTimeoutError(
                    f"Request timeout: {e}",
                    error_code=ErrorCodes.LLM_TIMEOUT
                ) from e
            elif "auth" in str(e).lower():
                raise LLMAuthenticationError(
                    f"Authentication failed: {e}",
                    error_code=ErrorCodes.LLM_AUTH_FAILED
                ) from e
            else:
                raise LLMProviderError(
                    f"Generation failed: {e}",
                    error_code=ErrorCodes.LLM_RESPONSE_INVALID
                ) from e
    
    @log_performance("llm_generate_stream")
    async def generate_stream(
        self,
        messages: Union[str, Dict[str, str], List[Union[str, Dict[str, str], LLMMessage]]],
        **kwargs
    ) -> AsyncIterator[LLMResponse]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            messages: Can be:
                - str: Single user message (e.g., "Hello!")
                - dict: Single message (e.g., {"role": "user", "content": "Hello!"})
                - List: Multiple messages in any format above or LLMMessage objects
            **kwargs: Additional generation parameters
            
        Yields:
            Streaming LLM response chunks
            
        Raises:
            LLMProviderError: If streaming fails
        """
        if not self.is_ready:
            raise LLMProviderError("Provider is not ready")
        
        # Convert input to LLMMessage format
        llm_messages = self._normalize_messages(messages)
        
        if not self.is_streaming_enabled:
            # Fall back to non-streaming
            response = await self.generate(llm_messages, **kwargs)
            yield response
            return
        
        # Apply rate limiting
        await self._apply_rate_limiting()
        
        # Merge generation parameters
        generation_params = self._merge_generation_params(stream=True, **kwargs)
        
        try:
            # Track request
            self._total_requests += 1
            start_time = datetime.now(UTC)
            
            # Generate streaming response
            async for chunk in self._generate_stream_impl(llm_messages, **generation_params):
                yield chunk
            
            # Update metrics
            end_time = datetime.now(UTC)
            response_time = (end_time - start_time).total_seconds()
            self._average_response_time = (
                (self._average_response_time * (self._total_requests - 1) + response_time)
                / self._total_requests
            )
            
        except Exception as e:
            self._total_errors += 1
            self.logger.error(
                "Failed to generate streaming LLM response",
                provider=self.provider_name,
                error=str(e)
            )
            raise LLMProviderError(f"Streaming generation failed: {e}") from e
    
    async def batch_generate(
        self,
        batch_messages: List[List[LLMMessage]],
        **kwargs
    ) -> List[LLMResponse]:
        """
        Generate responses for multiple message lists in batch.
        
        Args:
            batch_messages: List of message lists
            **kwargs: Additional generation parameters
            
        Returns:
            List of LLM responses
        """
        if not batch_messages:
            return []
        
        # Process batch with concurrency control
        semaphore = asyncio.Semaphore(self.provider_config.connection_pool_size)
        
        async def generate_single(messages: List[LLMMessage]) -> LLMResponse:
            async with semaphore:
                return await self.generate(messages, **kwargs)
        
        tasks = [generate_single(messages) for messages in batch_messages]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                self.logger.error(
                    "Batch generation failed for item",
                    index=i,
                    error=str(response)
                )
                # Create error response
                results.append(LLMResponse(
                    content=f"Error: {response}",
                    finish_reason="error",
                    metadata={"error": str(response)}
                ))
            else:
                results.append(response)
        
        return results
    
    # Rate limiting
    
    async def _apply_rate_limiting(self) -> None:
        """Apply rate limiting before making requests."""
        if not (self.provider_config.requests_per_minute or self.provider_config.tokens_per_minute):
            return
        
        async with self._rate_limit_lock:
            now = datetime.now(UTC)
            
            # Clean old entries (older than 1 minute)
            cutoff = now.timestamp() - 60
            
            if self.provider_config.requests_per_minute:
                self._request_times = [
                    req_time for req_time in self._request_times
                    if req_time.timestamp() > cutoff
                ]
                
                # Check request rate limit
                if len(self._request_times) >= self.provider_config.requests_per_minute:
                    sleep_time = 60 - (now.timestamp() - self._request_times[0].timestamp())
                    if sleep_time > 0:
                        self.logger.warning(
                            "Rate limit reached, sleeping",
                            sleep_time=sleep_time
                        )
                        await asyncio.sleep(sleep_time)
                
                self._request_times.append(now)
            
            if self.provider_config.tokens_per_minute:
                self._token_usage = [
                    (usage_time, tokens) for usage_time, tokens in self._token_usage
                    if usage_time.timestamp() > cutoff
                ]
                
                # Check token rate limit
                total_tokens = sum(tokens for _, tokens in self._token_usage)
                if total_tokens >= self.provider_config.tokens_per_minute:
                    sleep_time = 60 - (now.timestamp() - self._token_usage[0][0].timestamp())
                    if sleep_time > 0:
                        self.logger.warning(
                            "Token rate limit reached, sleeping",
                            sleep_time=sleep_time
                        )
                        await asyncio.sleep(sleep_time)
    
    def _update_metrics(self, response: LLMResponse, response_time: float) -> None:
        """Update provider metrics."""
        # Update response time
        self._average_response_time = (
            (self._average_response_time * (self._total_requests - 1) + response_time)
            / self._total_requests
        )
        
        # Update token usage
        if response.usage:
            tokens_used = response.usage.get("total_tokens", 0)
            self._total_tokens += tokens_used
            
            if self.provider_config.tokens_per_minute:
                self._token_usage.append((datetime.now(UTC), tokens_used))
    
    def _merge_generation_params(self, **kwargs) -> Dict[str, Any]:
        """Merge generation parameters with config defaults."""
        params = {
            "temperature": self.provider_config.temperature,
            "max_tokens": self.provider_config.max_tokens,
            "top_p": self.provider_config.top_p,
            "frequency_penalty": self.provider_config.frequency_penalty,
            "presence_penalty": self.provider_config.presence_penalty,
            "stop": self.provider_config.stop,
            "stream": self.provider_config.stream,
        }
        
        # Add function/tool calling parameters
        if self.provider_config.functions:
            params["functions"] = self.provider_config.functions
        if self.provider_config.tools:
            params["tools"] = self.provider_config.tools
        if self.provider_config.function_call:
            params["function_call"] = self.provider_config.function_call
        if self.provider_config.tool_choice:
            params["tool_choice"] = self.provider_config.tool_choice
        
        # Override with provided kwargs
        params.update(kwargs)
        
        # Remove None values
        return {k: v for k, v in params.items() if v is not None}
    

    
    # Utility methods
    
    def _normalize_messages(self, messages: Union[str, Dict[str, str], List[Union[str, Dict[str, str], LLMMessage]]]) -> List[LLMMessage]:
        """
        Convert various input formats to List[LLMMessage].
        
        Args:
            messages: Can be:
                - str: Single user message
                - dict: Single message with role/content
                - List: Multiple messages in any format
                
        Returns:
            List of LLMMessage objects
        """
        # Handle single string input
        if isinstance(messages, str):
            return [LLMMessage(role=LLMRole.USER, content=messages)]
        
        # Handle single dict input
        if isinstance(messages, dict):
            role = LLMRole(messages.get("role", "user"))
            content = messages.get("content", "")
            return [LLMMessage(role=role, content=content)]
        
        # Handle list input
        if isinstance(messages, list):
            normalized = []
            for msg in messages:
                if isinstance(msg, str):
                    normalized.append(LLMMessage(role=LLMRole.USER, content=msg))
                elif isinstance(msg, dict):
                    role = LLMRole(msg.get("role", "user"))
                    content = msg.get("content", "")
                    normalized.append(LLMMessage(role=role, content=content))
                elif isinstance(msg, LLMMessage):
                    normalized.append(msg)
                else:
                    raise ValueError(f"Unsupported message type: {type(msg)}")
            return normalized
        
        # Handle already normalized LLMMessage
        if isinstance(messages, LLMMessage):
            return [messages]
        
        raise ValueError(f"Unsupported messages type: {type(messages)}")
    
    def _messages_to_provider_format(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert LLMMessage objects to provider-specific format."""
        provider_messages = []
        
        for message in messages:
            provider_message = {
                "role": message.role.value,
                "content": message.content
            }
            
            if message.name:
                provider_message["name"] = message.name
            
            if message.function_call:
                provider_message["function_call"] = message.function_call
            
            if message.tool_calls:
                provider_message["tool_calls"] = message.tool_calls
            
            provider_messages.append(provider_message)
        
        return provider_messages
    
    def _provider_response_to_llm_response(
        self,
        provider_response: Any,
        request_id: Optional[str] = None
    ) -> LLMResponse:
        """Convert provider-specific response to LLMResponse."""
        # This is a base implementation that should be overridden by subclasses
        return LLMResponse(
            content=str(provider_response),
            provider=self.provider_name,
            model=self.model_name,
            request_id=request_id or str(uuid.uuid4())
        )
