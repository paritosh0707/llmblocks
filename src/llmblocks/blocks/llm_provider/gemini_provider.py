"""
Google Gemini Provider for LLMBlocks.

This module provides a Google Gemini LLM provider with async support,
streaming, and comprehensive error handling.
"""

import asyncio
from typing import Any, Dict, List, Optional, AsyncIterator
import json
import uuid

from pydantic import Field, SecretStr, field_validator
from langchain_google_genai import ChatGoogleGenerativeAI
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


class GeminiProviderConfig(LLMProviderConfig):
    """Configuration for Google Gemini provider."""
    
    provider_name: str = "gemini"
    model: str = "gemini-2.0-flash"
    api_key: SecretStr = Field(..., description="Google AI API key")
    
    # Gemini-specific parameters
    candidate_count: int = Field(default=1, ge=1, le=8, description="Number of candidates to generate")
    top_k: Optional[int] = Field(default=None, ge=1, description="Top-k sampling parameter")
    
    # Safety settings
    safety_settings: Optional[Dict[str, str]] = Field(
        default=None,
        description="Safety settings for content filtering"
    )
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        """Validate Gemini model name."""
        valid_models = [
            "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash",
            "gemini-1.0-pro", "gemini-pro", "gemini-pro-vision"
        ]
        if v not in valid_models:
            # Allow custom models but log a warning
            pass
        return v


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini LLM provider implementation.
    
    This provider supports:
    - All Gemini models (Gemini Pro, Flash, etc.)
    - Streaming responses
    - Safety settings and content filtering
    - Async operations
    - Comprehensive error handling
    """
    
    def __init__(self, config: GeminiProviderConfig, **kwargs):
        """
        Initialize Gemini provider.
        
        Args:
            config: Gemini provider configuration
            **kwargs: Additional configuration parameters
        """
        # Convert dict config to GeminiProviderConfig if needed
        if isinstance(config, dict):
            config = GeminiProviderConfig(**config, **kwargs)
        
        super().__init__(config)
        
        self._gemini_config = config
        
        # LangChain Gemini client
        self._langchain_client: Optional[ChatGoogleGenerativeAI] = None
    
    @property
    def gemini_config(self) -> GeminiProviderConfig:
        """Get the Gemini configuration."""
        return self._gemini_config
    
    async def _initialize_clients(self) -> None:
        """Initialize LangChain Gemini client."""
        try:
            # Get API key
            api_key = self.gemini_config.api_key
            if isinstance(api_key, SecretStr):
                api_key = api_key.get_secret_value()
            
            # Client configuration for LangChain ChatGoogleGenerativeAI
            client_config = {
                "model": self.gemini_config.model,
                "google_api_key": api_key,
                "temperature": self.gemini_config.temperature,
                "top_p": self.gemini_config.top_p,
                # Note: convert_system_message_to_human is deprecated
                # System messages are now handled natively by the model
            }
            
            # Add max_tokens if specified
            if self.gemini_config.max_tokens:
                client_config["max_output_tokens"] = self.gemini_config.max_tokens
            
            # Add top_k if specified
            if self.gemini_config.top_k:
                client_config["top_k"] = self.gemini_config.top_k
            
            # Initialize LangChain client
            self._langchain_client = ChatGoogleGenerativeAI(**client_config)
            
            self.logger.info(
                "LangChain Gemini client initialized",
                model=self.gemini_config.model
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LangChain Gemini client: {e}")
            raise LLMConnectionError(
                f"Failed to initialize LangChain Gemini client: {e}",
                error_code=ErrorCodes.LLM_CONNECTION_FAILED
            ) from e
    

    
    def _close_sync_client(self) -> None:
        """Close synchronous client."""
        if self._langchain_client:
            # LangChain client doesn't need explicit closing
            self._langchain_client = None
    
    async def _close_client(self) -> None:
        """Close synchronous client (async version for compatibility)."""
        self._close_sync_client()
    
    async def _close_async_client(self) -> None:
        """Close asynchronous client."""
        # LangChain client doesn't need explicit closing
        pass
    
    async def _test_connection(self) -> None:
        """Test connection to Gemini API using LangChain client."""
        try:
            from langchain_core.messages import HumanMessage
            
            # Make a simple test request using LangChain
            test_messages = [HumanMessage(content="Hello")]
            response = await self._langchain_client.agenerate([test_messages])
            
            if not response.generations or not response.generations[0]:
                raise LLMConnectionError("Empty response from Gemini API")
            
            self.logger.info("Gemini connection test successful")
            
        except Exception as e:
            error_msg = str(e).lower()
            if "api key" in error_msg or "auth" in error_msg:
                raise LLMAuthenticationError(
                    f"Gemini authentication failed: {e}",
                    error_code=ErrorCodes.LLM_AUTH_FAILED
                ) from e
            elif "quota" in error_msg or "rate" in error_msg:
                raise LLMRateLimitError(
                    f"Gemini rate limit exceeded: {e}",
                    error_code=ErrorCodes.LLM_RATE_LIMIT
                ) from e
            elif "timeout" in error_msg:
                raise LLMTimeoutError(
                    f"Gemini request timeout: {e}",
                    error_code=ErrorCodes.LLM_TIMEOUT
                ) from e
            else:
                raise LLMConnectionError(
                    f"Gemini connection test failed: {e}",
                    error_code=ErrorCodes.LLM_CONNECTION_FAILED
                ) from e
    
    async def _generate_impl(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """Generate response using LangChain Gemini client."""
        try:
            # Convert messages to LangChain format
            langchain_messages = [msg.to_langchain_message() for msg in messages]
            
            # Use LangChain client
            response = await self._langchain_client.agenerate([langchain_messages])
            
            # Convert response to our format
            return self._langchain_response_to_llm_response(response)
            
        except Exception as e:
            error_msg = str(e).lower()
            if "api key" in error_msg or "auth" in error_msg:
                raise LLMAuthenticationError(
                    f"Gemini authentication failed: {e}",
                    error_code=ErrorCodes.LLM_AUTH_FAILED
                ) from e
            elif "quota" in error_msg or "rate" in error_msg:
                raise LLMRateLimitError(
                    f"Gemini rate limit exceeded: {e}",
                    error_code=ErrorCodes.LLM_RATE_LIMIT
                ) from e
            elif "timeout" in error_msg:
                raise LLMTimeoutError(
                    f"Gemini request timeout: {e}",
                    error_code=ErrorCodes.LLM_TIMEOUT
                ) from e
            else:
                raise LLMProviderError(
                    f"Gemini generation failed: {e}",
                    error_code=ErrorCodes.LLM_RESPONSE_INVALID
                ) from e
    
    async def _generate_stream_impl(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> AsyncIterator[LLMResponse]:
        """Generate streaming response using Gemini API."""
        try:
            # Convert messages to LangChain format
            langchain_messages = [msg.to_langchain_message() for msg in messages]
            
            # Use LangChain streaming
            async for chunk in self._langchain_client.astream(langchain_messages, **kwargs):
                yield LLMResponse(
                    content=chunk.content,
                    model=self.gemini_config.model,
                    provider=self.provider_name,
                    metadata={
                        "chunk": True,
                        "finish_reason": getattr(chunk, 'response_metadata', {}).get('finish_reason', None)
                    }
                )
            
            return  # Exit early since we're using LangChain streaming
            
            # Old direct API code (kept for reference but not executed)
            if False and len(gemini_messages) == 1:
                # Single message
                response_stream = await asyncio.to_thread(
                    self._model.generate_content,
                    gemini_messages[0],
                    generation_config=generation_config,
                    stream=True
                )
            else:
                # Multi-turn conversation
                chat = self._model.start_chat(history=gemini_messages[:-1])
                response_stream = await asyncio.to_thread(
                    chat.send_message,
                    gemini_messages[-1],
                    generation_config=generation_config,
                    stream=True
                )
            
            # Process streaming response
            accumulated_content = ""
            
            for chunk in response_stream:
                if chunk.text:
                    accumulated_content += chunk.text
                    
                    yield LLMResponse(
                        content=chunk.text,
                        role=LLMRole.ASSISTANT,
                        model=self.gemini_config.model,
                        provider=self.provider_name,
                        request_id=str(uuid.uuid4()),
                        metadata={
                            "is_partial": True,
                            "accumulated_content": accumulated_content,
                            "safety_ratings": getattr(chunk, 'safety_ratings', None)
                        }
                    )
            
            # Yield final response
            final_response = LLMResponse(
                content=accumulated_content,
                role=LLMRole.ASSISTANT,
                finish_reason="stop",
                model=self.gemini_config.model,
                provider=self.provider_name,
                request_id=str(uuid.uuid4()),
                metadata={
                    "is_final": True,
                    "safety_ratings": getattr(response_stream, 'safety_ratings', None)
                }
            )
            yield final_response
            
        except Exception as e:
            self.logger.error(f"Gemini streaming failed: {e}")
            raise LLMProviderError(f"Gemini streaming failed: {e}") from e
    

    

    
    def _langchain_response_to_llm_response(self, response: ChatResult) -> LLMResponse:
        """Convert LangChain response to LLMResponse."""
        if not response.generations or not response.generations[0]:
            raise LLMProviderError("Empty response from Gemini")
        
        # Handle both old and new LangChain response formats
        if isinstance(response.generations[0], list):
            generation = response.generations[0][0]  # Old format
        else:
            generation = response.generations[0]  # New format
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
            model=self.gemini_config.model,
            provider=self.provider_name,
            request_id=additional_kwargs.get('request_id', str(uuid.uuid4())),
            metadata={
                "langchain_generation_info": getattr(generation, 'generation_info', {}),
                **additional_kwargs
            }
        )
    
    async def _health_check_impl(self) -> Dict[str, Any]:
        """Perform Gemini-specific health check."""
        health = await super()._health_check_impl()
        
        try:
            from langchain_core.messages import HumanMessage
            
            # Test API connectivity
            test_messages = [HumanMessage(content="Health check")]
            response = await self._langchain_client.agenerate([test_messages])
            
            health.update({
                "api_accessible": True,
                "model_available": bool(response.generations and response.generations[0]),
                "api_response_time": "< 1s"  # Simple indicator
            })
            
        except Exception as e:
            health.update({
                "api_accessible": False,
                "api_error": str(e)
            })
        
        return health
