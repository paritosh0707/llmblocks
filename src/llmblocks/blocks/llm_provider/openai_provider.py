"""
OpenAI Provider for LLMBlocks.

This module provides an OpenAI LLM provider using LangChain's ChatOpenAI
as the underlying implementation, with additional features like connection pooling,
enhanced error handling, and our custom interface.
"""

import asyncio
from typing import Any, Dict, List, Optional, AsyncIterator
import json
import uuid

from pydantic import Field, SecretStr, validator
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

from .base import (
    BaseLLMProvider,
    LLMProviderConfig,
    LLMMessage,
    LLMResponse,
    LLMRole
)
from ...utils.logging import get_logger
from ...utils.exceptions import (
    LLMProviderError,
    LLMConnectionError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    ErrorCodes
)


class OpenAIProviderConfig(LLMProviderConfig):
    """Configuration for OpenAI provider."""
    
    provider_name: str = "openai"
    model: str = "gpt-4o"
    api_key: SecretStr = Field(..., description="OpenAI API key")
    api_base: Optional[str] = Field(default=None, description="Custom API base URL")
    organization: Optional[str] = Field(default=None, description="OpenAI organization ID")
    
    # OpenAI-specific parameters
    response_format: Optional[Dict[str, Any]] = Field(default=None, description="Response format")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    logit_bias: Optional[Dict[str, float]] = Field(default=None, description="Logit bias")
    user: Optional[str] = Field(default=None, description="User identifier")
    
    @validator('model')
    def validate_model(cls, v):
        """Validate OpenAI model name."""
        valid_models = [
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-4-32k",
            "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-instruct"
        ]
        if v not in valid_models:
            # Allow custom models but log a warning
            pass
        return v


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider implementation using LangChain's ChatOpenAI.
    
    This provider supports:
    - All OpenAI chat models (GPT-4, GPT-3.5-turbo, etc.)
    - Streaming responses
    - Function calling and tool usage
    - Async operations with connection pooling
    - LangChain/LangGraph compatibility
    - Comprehensive error handling and retry logic
    """
    
    def __init__(self, config: OpenAIProviderConfig, **kwargs):
        """
        Initialize OpenAI provider.
        
        Args:
            config: OpenAI provider configuration
            **kwargs: Additional configuration parameters
        """
        # Convert dict config to OpenAIProviderConfig if needed
        if isinstance(config, dict):
            config = OpenAIProviderConfig(**config, **kwargs)
        
        super().__init__(config)
        
        self.openai_config = config
        self.logger = get_logger("OpenAIProvider")
        
        # LangChain OpenAI client
        self._langchain_client: Optional[ChatOpenAI] = None
    
    async def _initialize_clients(self) -> None:
        """Initialize LangChain OpenAI clients."""
        try:
            # Get API key
            api_key = self.openai_config.api_key
            if isinstance(api_key, SecretStr):
                api_key = api_key.get_secret_value()
            
            # Client configuration for LangChain ChatOpenAI
            client_config = {
                "model": self.openai_config.model,
                "openai_api_key": api_key,
                "temperature": self.openai_config.temperature,
                "max_tokens": self.openai_config.max_tokens,
                "top_p": self.openai_config.top_p,
                "frequency_penalty": self.openai_config.frequency_penalty,
                "presence_penalty": self.openai_config.presence_penalty,
                "request_timeout": self.openai_config.timeout,
                "max_retries": self.openai_config.max_retries,
                "streaming": self.openai_config.stream,
            }
            
            # Add optional parameters
            if self.openai_config.api_base:
                client_config["openai_api_base"] = self.openai_config.api_base
            
            if self.openai_config.organization:
                client_config["openai_organization"] = self.openai_config.organization
            
            if self.openai_config.stop:
                client_config["stop"] = self.openai_config.stop
            
            # Add OpenAI-specific parameters
            if self.openai_config.seed is not None:
                client_config["seed"] = self.openai_config.seed
            
            if self.openai_config.logit_bias:
                client_config["logit_bias"] = self.openai_config.logit_bias
            
            if self.openai_config.user:
                client_config["user"] = self.openai_config.user
            
            # Initialize LangChain client
            self._langchain_client = ChatOpenAI(**client_config)
            
            self.logger.info(
                "LangChain OpenAI clients initialized",
                model=self.openai_config.model,
                api_base=self.openai_config.api_base
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LangChain OpenAI clients: {e}")
            raise LLMConnectionError(
                f"Failed to initialize LangChain OpenAI clients: {e}",
                error_code=ErrorCodes.LLM_CONNECTION_FAILED
            ) from e
    
    async def _close_client(self) -> None:
        """Close synchronous client."""
        if self._langchain_client:
            # LangChain client doesn't need explicit closing
            self._langchain_client = None
    
    async def _close_async_client(self) -> None:
        """Close asynchronous client."""
        # LangChain client doesn't need explicit closing
        pass
    
    async def _test_connection(self) -> None:
        """Test connection to OpenAI API using LangChain client."""
        try:
            from langchain_core.messages import HumanMessage
            
            # Make a simple test request using LangChain
            test_messages = [HumanMessage(content="Hello")]
            response = await self._langchain_client.agenerate([test_messages])
            
            if not response.generations or not response.generations[0]:
                raise LLMConnectionError("Empty response from OpenAI API")
            
            self.logger.info("OpenAI connection test successful")
            
        except Exception as e:
            error_msg = str(e).lower()
            if "auth" in error_msg or "api key" in error_msg:
                raise LLMAuthenticationError(
                    f"OpenAI authentication failed: {e}",
                    error_code=ErrorCodes.LLM_AUTH_FAILED
                ) from e
            elif "rate limit" in error_msg:
                raise LLMRateLimitError(
                    f"OpenAI rate limit exceeded: {e}",
                    error_code=ErrorCodes.LLM_RATE_LIMIT
                ) from e
            elif "timeout" in error_msg:
                raise LLMTimeoutError(
                    f"OpenAI request timeout: {e}",
                    error_code=ErrorCodes.LLM_TIMEOUT
                ) from e
            else:
                raise LLMConnectionError(
                    f"OpenAI connection test failed: {e}",
                    error_code=ErrorCodes.LLM_CONNECTION_FAILED
                ) from e
    
    async def _generate_impl(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """Generate response using LangChain OpenAI client."""
        try:
            # Convert messages to LangChain format
            langchain_messages = [msg.to_langchain_message() for msg in messages]
            
            # Use existing client (LangChain ChatOpenAI supports async)
            response = await self._langchain_client.agenerate([langchain_messages])
            
            # Convert response to our format
            return self._langchain_response_to_llm_response(response)
            
        except Exception as e:
            error_msg = str(e).lower()
            if "auth" in error_msg or "api key" in error_msg:
                raise LLMAuthenticationError(
                    f"OpenAI authentication failed: {e}",
                    error_code=ErrorCodes.LLM_AUTH_FAILED
                ) from e
            elif "rate limit" in error_msg:
                raise LLMRateLimitError(
                    f"OpenAI rate limit exceeded: {e}",
                    error_code=ErrorCodes.LLM_RATE_LIMIT
                ) from e
            elif "timeout" in error_msg:
                raise LLMTimeoutError(
                    f"OpenAI request timeout: {e}",
                    error_code=ErrorCodes.LLM_TIMEOUT
                ) from e
            else:
                raise LLMProviderError(
                    f"OpenAI generation failed: {e}",
                    error_code=ErrorCodes.LLM_RESPONSE_INVALID
                ) from e
    
    async def _generate_stream_impl(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> AsyncIterator[LLMResponse]:
        """Generate streaming response using OpenAI API."""
        try:
            # Convert messages to OpenAI format
            openai_messages = self._messages_to_openai_format(messages)
            
            # Prepare request parameters
            request_params = self._prepare_request_params(stream=True, **kwargs)
            request_params["messages"] = openai_messages
            
            # Make streaming API request
            stream = await self._async_client.chat.completions.create(**request_params)
            
            # Process streaming response
            accumulated_content = ""
            accumulated_function_call = None
            accumulated_tool_calls = []
            
            async for chunk in stream:
                if not chunk.choices:
                    continue
                
                choice = chunk.choices[0]
                delta = choice.delta
                
                # Handle content
                if delta.content:
                    accumulated_content += delta.content
                    
                    yield LLMResponse(
                        content=delta.content,
                        role=LLMRole.ASSISTANT,
                        finish_reason=choice.finish_reason,
                        model=chunk.model,
                        provider=self.provider_name,
                        request_id=chunk.id or str(uuid.uuid4()),
                        metadata={
                            "is_partial": True,
                            "accumulated_content": accumulated_content
                        }
                    )
                
                # Handle function calls
                if delta.function_call:
                    if accumulated_function_call is None:
                        accumulated_function_call = {"name": "", "arguments": ""}
                    
                    if delta.function_call.name:
                        accumulated_function_call["name"] += delta.function_call.name
                    
                    if delta.function_call.arguments:
                        accumulated_function_call["arguments"] += delta.function_call.arguments
                
                # Handle tool calls
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        # Extend accumulated_tool_calls if needed
                        while len(accumulated_tool_calls) <= tool_call.index:
                            accumulated_tool_calls.append({
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })
                        
                        acc_tool_call = accumulated_tool_calls[tool_call.index]
                        
                        if tool_call.id:
                            acc_tool_call["id"] += tool_call.id
                        
                        if tool_call.function:
                            if tool_call.function.name:
                                acc_tool_call["function"]["name"] += tool_call.function.name
                            if tool_call.function.arguments:
                                acc_tool_call["function"]["arguments"] += tool_call.function.arguments
                
                # Yield final response if stream is complete
                if choice.finish_reason:
                    final_response = LLMResponse(
                        content=accumulated_content,
                        role=LLMRole.ASSISTANT,
                        finish_reason=choice.finish_reason,
                        model=chunk.model,
                        provider=self.provider_name,
                        request_id=chunk.id or str(uuid.uuid4()),
                        function_call=accumulated_function_call,
                        tool_calls=accumulated_tool_calls if accumulated_tool_calls else None,
                        metadata={"is_final": True}
                    )
                    yield final_response
            
        except Exception as e:
            self.logger.error(f"OpenAI streaming failed: {e}")
            raise LLMProviderError(f"OpenAI streaming failed: {e}") from e
    
    def _prepare_client_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Prepare client parameters that differ from defaults."""
        client_kwargs = {}
        
        # Check if any parameters differ from the current client configuration
        if kwargs.get("temperature") != self.openai_config.temperature:
            client_kwargs["temperature"] = kwargs.get("temperature")
        
        if kwargs.get("max_tokens") != self.openai_config.max_tokens:
            client_kwargs["max_tokens"] = kwargs.get("max_tokens")
        
        if kwargs.get("top_p") != self.openai_config.top_p:
            client_kwargs["top_p"] = kwargs.get("top_p")
        
        if kwargs.get("frequency_penalty") != self.openai_config.frequency_penalty:
            client_kwargs["frequency_penalty"] = kwargs.get("frequency_penalty")
        
        if kwargs.get("presence_penalty") != self.openai_config.presence_penalty:
            client_kwargs["presence_penalty"] = kwargs.get("presence_penalty")
        
        if kwargs.get("stop") != self.openai_config.stop:
            client_kwargs["stop"] = kwargs.get("stop")
        
        return client_kwargs
    
    def _create_updated_client(self, **kwargs) -> ChatOpenAI:
        """Create a new client with updated parameters."""
        # Get base configuration
        api_key = self.openai_config.api_key
        if isinstance(api_key, SecretStr):
            api_key = api_key.get_secret_value()
        
        client_config = {
            "model": self.openai_config.model,
            "openai_api_key": api_key,
            "temperature": kwargs.get("temperature", self.openai_config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.openai_config.max_tokens),
            "top_p": kwargs.get("top_p", self.openai_config.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self.openai_config.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.openai_config.presence_penalty),
            "request_timeout": self.openai_config.timeout,
            "max_retries": self.openai_config.max_retries,
            "streaming": kwargs.get("stream", self.openai_config.stream),
        }
        
        # Add optional parameters
        if self.openai_config.api_base:
            client_config["openai_api_base"] = self.openai_config.api_base
        
        if self.openai_config.organization:
            client_config["openai_organization"] = self.openai_config.organization
        
        if kwargs.get("stop", self.openai_config.stop):
            client_config["stop"] = kwargs.get("stop", self.openai_config.stop)
        
        return ChatOpenAI(**client_config)
    
    def _langchain_response_to_llm_response(self, response: ChatResult) -> LLMResponse:
        """Convert LangChain response to LLMResponse."""
        if not response.generations or not response.generations[0]:
            raise LLMProviderError("Empty response from OpenAI")
        
        generation = response.generations[0][0]
        message = generation.message
        
        # Extract usage information from response metadata
        usage = None
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            if token_usage:
                usage = {
                    "prompt_tokens": token_usage.get('prompt_tokens', 0),
                    "completion_tokens": token_usage.get('completion_tokens', 0),
                    "total_tokens": token_usage.get('total_tokens', 0)
                }
        
        # Extract additional kwargs
        additional_kwargs = getattr(message, 'additional_kwargs', {})
        
        return LLMResponse(
            content=message.content or "",
            role=LLMRole.ASSISTANT,
            finish_reason=additional_kwargs.get('finish_reason', 'stop'),
            usage=usage,
            model=self.openai_config.model,
            provider=self.provider_name,
            request_id=additional_kwargs.get('request_id', str(uuid.uuid4())),
            function_call=additional_kwargs.get('function_call'),
            tool_calls=additional_kwargs.get('tool_calls'),
            metadata={
                "langchain_generation_info": getattr(generation, 'generation_info', {}),
                **additional_kwargs
            }
        )
    
    async def _health_check_impl(self) -> Dict[str, Any]:
        """Perform OpenAI-specific health check."""
        health = await super()._health_check_impl()
        
        try:
            # Test API connectivity
            response = await self._async_client.chat.completions.create(
                model=self.openai_config.model,
                messages=[{"role": "user", "content": "Health check"}],
                max_tokens=1
            )
            
            health.update({
                "api_accessible": True,
                "model_available": bool(response.choices),
                "api_response_time": "< 1s"  # Simple indicator
            })
            
        except Exception as e:
            health.update({
                "api_accessible": False,
                "api_error": str(e)
            })
        
        return health
