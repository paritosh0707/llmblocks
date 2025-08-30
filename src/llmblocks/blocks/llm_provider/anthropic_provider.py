"""
Anthropic Claude Provider for LLMBlocks.

This module provides an Anthropic Claude LLM provider with async support,
streaming, and comprehensive error handling.
"""

import asyncio
from typing import Any, Dict, List, Optional, AsyncIterator
import json
import uuid

from pydantic import Field, SecretStr, validator
import anthropic
from anthropic import AsyncAnthropic

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


class AnthropicProviderConfig(LLMProviderConfig):
    """Configuration for Anthropic Claude provider."""
    
    provider_name: str = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    api_key: SecretStr = Field(..., description="Anthropic API key")
    api_base: Optional[str] = Field(default=None, description="Custom API base URL")
    
    # Anthropic-specific parameters
    system: Optional[str] = Field(default=None, description="System message")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Request metadata")
    
    @validator('model')
    def validate_model(cls, v):
        """Validate Anthropic model name."""
        valid_models = [
            "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
            "claude-2.1", "claude-2.0", "claude-instant-1.2"
        ]
        if v not in valid_models:
            # Allow custom models but log a warning
            pass
        return v


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude LLM provider implementation.
    
    This provider supports:
    - All Claude models (Opus, Sonnet, Haiku, etc.)
    - Streaming responses
    - System messages
    - Async operations with connection pooling
    - Comprehensive error handling and retry logic
    """
    
    def __init__(self, config: AnthropicProviderConfig, **kwargs):
        """
        Initialize Anthropic provider.
        
        Args:
            config: Anthropic provider configuration
            **kwargs: Additional configuration parameters
        """
        # Convert dict config to AnthropicProviderConfig if needed
        if isinstance(config, dict):
            config = AnthropicProviderConfig(**config, **kwargs)
        
        super().__init__(config)
        
        self.anthropic_config = config
        self.logger = get_logger("AnthropicProvider")
        
        # Anthropic clients
        self._sync_client: Optional[anthropic.Anthropic] = None
        self._async_client: Optional[AsyncAnthropic] = None
    
    async def _initialize_clients(self) -> None:
        """Initialize Anthropic clients."""
        try:
            # Get API key
            api_key = self.anthropic_config.api_key
            if isinstance(api_key, SecretStr):
                api_key = api_key.get_secret_value()
            
            # Client configuration
            client_config = {
                "api_key": api_key,
                "timeout": self.anthropic_config.timeout,
                "max_retries": self.anthropic_config.max_retries,
            }
            
            # Add optional parameters
            if self.anthropic_config.api_base:
                client_config["base_url"] = self.anthropic_config.api_base
            
            # Initialize clients
            self._sync_client = anthropic.Anthropic(**client_config)
            self._async_client = AsyncAnthropic(**client_config)
            
            self.logger.info(
                "Anthropic clients initialized",
                model=self.anthropic_config.model,
                api_base=self.anthropic_config.api_base
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic clients: {e}")
            raise LLMConnectionError(
                f"Failed to initialize Anthropic clients: {e}",
                error_code=ErrorCodes.LLM_CONNECTION_FAILED
            ) from e
    
    async def _close_client(self) -> None:
        """Close synchronous client."""
        if self._sync_client:
            # Anthropic sync client doesn't need explicit closing
            self._sync_client = None
    
    async def _close_async_client(self) -> None:
        """Close asynchronous client."""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None
    
    async def _test_connection(self) -> None:
        """Test connection to Anthropic API."""
        try:
            # Make a simple test request
            response = await self._async_client.messages.create(
                model=self.anthropic_config.model,
                max_tokens=5,
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            if not response.content:
                raise LLMConnectionError("Empty response from Anthropic API")
            
            self.logger.info("Anthropic connection test successful")
            
        except anthropic.AuthenticationError as e:
            raise LLMAuthenticationError(
                f"Anthropic authentication failed: {e}",
                error_code=ErrorCodes.LLM_AUTH_FAILED
            ) from e
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(
                f"Anthropic rate limit exceeded: {e}",
                error_code=ErrorCodes.LLM_RATE_LIMIT
            ) from e
        except Exception as e:
            raise LLMConnectionError(
                f"Anthropic connection test failed: {e}",
                error_code=ErrorCodes.LLM_CONNECTION_FAILED
            ) from e
    
    async def _generate_impl(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic API."""
        try:
            # Convert messages to Anthropic format
            anthropic_messages, system_message = self._messages_to_anthropic_format(messages)
            
            # Prepare request parameters
            request_params = self._prepare_request_params(system_message, **kwargs)
            request_params["messages"] = anthropic_messages
            
            # Make API request
            response = await self._async_client.messages.create(**request_params)
            
            # Convert response
            return self._anthropic_response_to_llm_response(response)
            
        except anthropic.AuthenticationError as e:
            raise LLMAuthenticationError(
                f"Anthropic authentication failed: {e}",
                error_code=ErrorCodes.LLM_AUTH_FAILED
            ) from e
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(
                f"Anthropic rate limit exceeded: {e}",
                error_code=ErrorCodes.LLM_RATE_LIMIT
            ) from e
        except anthropic.APITimeoutError as e:
            raise LLMTimeoutError(
                f"Anthropic request timeout: {e}",
                error_code=ErrorCodes.LLM_TIMEOUT
            ) from e
        except Exception as e:
            raise LLMProviderError(
                f"Anthropic generation failed: {e}",
                error_code=ErrorCodes.LLM_RESPONSE_INVALID
            ) from e
    
    async def _generate_stream_impl(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> AsyncIterator[LLMResponse]:
        """Generate streaming response using Anthropic API."""
        try:
            # Convert messages to Anthropic format
            anthropic_messages, system_message = self._messages_to_anthropic_format(messages)
            
            # Prepare request parameters
            request_params = self._prepare_request_params(system_message, stream=True, **kwargs)
            request_params["messages"] = anthropic_messages
            
            # Make streaming API request
            stream = await self._async_client.messages.create(**request_params)
            
            # Process streaming response
            accumulated_content = ""
            
            async for event in stream:
                if event.type == "content_block_start":
                    # Start of content block
                    continue
                elif event.type == "content_block_delta":
                    # Content delta
                    if hasattr(event.delta, 'text'):
                        text_chunk = event.delta.text
                        accumulated_content += text_chunk
                        
                        yield LLMResponse(
                            content=text_chunk,
                            role=LLMRole.ASSISTANT,
                            model=self.anthropic_config.model,
                            provider=self.provider_name,
                            request_id=str(uuid.uuid4()),
                            metadata={
                                "is_partial": True,
                                "accumulated_content": accumulated_content
                            }
                        )
                elif event.type == "content_block_stop":
                    # End of content block
                    continue
                elif event.type == "message_delta":
                    # Message metadata update
                    continue
                elif event.type == "message_stop":
                    # End of message
                    final_response = LLMResponse(
                        content=accumulated_content,
                        role=LLMRole.ASSISTANT,
                        finish_reason="stop",
                        model=self.anthropic_config.model,
                        provider=self.provider_name,
                        request_id=str(uuid.uuid4()),
                        metadata={"is_final": True}
                    )
                    yield final_response
                    break
            
        except Exception as e:
            self.logger.error(f"Anthropic streaming failed: {e}")
            raise LLMProviderError(f"Anthropic streaming failed: {e}") from e
    
    def _messages_to_anthropic_format(self, messages: List[LLMMessage]) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """Convert LLMMessage objects to Anthropic format."""
        anthropic_messages = []
        system_message = None
        
        for message in messages:
            if message.role == LLMRole.SYSTEM:
                # Anthropic uses separate system parameter
                system_message = message.content
            elif message.role in [LLMRole.USER, LLMRole.ASSISTANT]:
                anthropic_messages.append({
                    "role": message.role.value,
                    "content": message.content
                })
            # Skip function/tool messages for now (not supported by Anthropic)
        
        return anthropic_messages, system_message
    
    def _prepare_request_params(self, system_message: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Prepare request parameters for Anthropic API."""
        params = {
            "model": self.anthropic_config.model,
            "max_tokens": kwargs.get("max_tokens", self.anthropic_config.max_tokens) or 1024,
            "temperature": kwargs.get("temperature", self.anthropic_config.temperature),
            "top_p": kwargs.get("top_p", self.anthropic_config.top_p),
            "stream": kwargs.get("stream", self.anthropic_config.stream),
        }
        
        # Add system message
        if system_message or self.anthropic_config.system:
            params["system"] = system_message or self.anthropic_config.system
        
        # Add stop sequences
        stop = kwargs.get("stop", self.anthropic_config.stop)
        if stop:
            params["stop_sequences"] = stop
        
        # Add metadata
        if self.anthropic_config.metadata:
            params["metadata"] = self.anthropic_config.metadata
        
        # Remove None values
        return {k: v for k, v in params.items() if v is not None}
    
    def _anthropic_response_to_llm_response(self, response: Any) -> LLMResponse:
        """Convert Anthropic response to LLMResponse."""
        if not response.content:
            raise LLMProviderError("Empty response from Anthropic")
        
        # Extract content (Anthropic returns list of content blocks)
        content = ""
        for content_block in response.content:
            if hasattr(content_block, 'text'):
                content += content_block.text
        
        # Extract usage information
        usage = None
        if hasattr(response, 'usage') and response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        
        return LLMResponse(
            content=content,
            role=LLMRole.ASSISTANT,
            finish_reason=getattr(response, 'stop_reason', 'stop'),
            usage=usage,
            model=response.model,
            provider=self.provider_name,
            request_id=response.id,
            metadata={
                "anthropic_response_id": response.id,
                "stop_reason": getattr(response, 'stop_reason', None),
                "stop_sequence": getattr(response, 'stop_sequence', None)
            }
        )
    
    async def _health_check_impl(self) -> Dict[str, Any]:
        """Perform Anthropic-specific health check."""
        health = await super()._health_check_impl()
        
        try:
            # Test API connectivity
            response = await self._async_client.messages.create(
                model=self.anthropic_config.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "Health check"}]
            )
            
            health.update({
                "api_accessible": True,
                "model_available": bool(response.content),
                "api_response_time": "< 1s"  # Simple indicator
            })
            
        except Exception as e:
            health.update({
                "api_accessible": False,
                "api_error": str(e)
            })
        
        return health
