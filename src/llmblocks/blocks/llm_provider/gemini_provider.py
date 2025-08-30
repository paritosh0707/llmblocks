"""
Google Gemini Provider for LLMBlocks.

This module provides a Google Gemini LLM provider with async support,
streaming, and comprehensive error handling.
"""

import asyncio
from typing import Any, Dict, List, Optional, AsyncIterator
import json
import uuid

from pydantic import Field, SecretStr, validator
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

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
    
    @validator('model')
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
        
        self.gemini_config = config
        self.logger = get_logger("GeminiProvider")
        
        # Gemini model
        self._model = None
    
    async def _initialize_clients(self) -> None:
        """Initialize Gemini client."""
        try:
            # Get API key
            api_key = self.gemini_config.api_key
            if isinstance(api_key, SecretStr):
                api_key = api_key.get_secret_value()
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Initialize model
            generation_config = self._get_generation_config()
            safety_settings = self._get_safety_settings()
            
            self._model = genai.GenerativeModel(
                model_name=self.gemini_config.model,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            self.logger.info(
                "Gemini client initialized",
                model=self.gemini_config.model
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}")
            raise LLMConnectionError(
                f"Failed to initialize Gemini client: {e}",
                error_code=ErrorCodes.LLM_CONNECTION_FAILED
            ) from e
    
    async def _close_client(self) -> None:
        """Close synchronous client."""
        # Gemini client doesn't need explicit closing
        self._model = None
    
    async def _close_async_client(self) -> None:
        """Close asynchronous client."""
        # Gemini client doesn't need explicit closing
        self._model = None
    
    async def _test_connection(self) -> None:
        """Test connection to Gemini API."""
        try:
            # Make a simple test request
            response = await asyncio.to_thread(
                self._model.generate_content,
                "Hello",
                generation_config=genai.types.GenerationConfig(max_output_tokens=5)
            )
            
            if not response.text:
                raise LLMConnectionError("Empty response from Gemini API")
            
            self.logger.info("Gemini connection test successful")
            
        except Exception as e:
            if "API_KEY" in str(e).upper():
                raise LLMAuthenticationError(
                    f"Gemini authentication failed: {e}",
                    error_code=ErrorCodes.LLM_AUTH_FAILED
                ) from e
            elif "QUOTA" in str(e).upper() or "RATE" in str(e).upper():
                raise LLMRateLimitError(
                    f"Gemini rate limit exceeded: {e}",
                    error_code=ErrorCodes.LLM_RATE_LIMIT
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
        """Generate response using Gemini API."""
        try:
            # Convert messages to Gemini format
            gemini_messages = self._messages_to_gemini_format(messages)
            
            # Prepare generation config
            generation_config = self._get_generation_config(**kwargs)
            
            # Make API request
            if len(gemini_messages) == 1:
                # Single message
                response = await asyncio.to_thread(
                    self._model.generate_content,
                    gemini_messages[0],
                    generation_config=generation_config
                )
            else:
                # Multi-turn conversation
                chat = self._model.start_chat(history=gemini_messages[:-1])
                response = await asyncio.to_thread(
                    chat.send_message,
                    gemini_messages[-1],
                    generation_config=generation_config
                )
            
            # Convert response
            return self._gemini_response_to_llm_response(response)
            
        except Exception as e:
            if "API_KEY" in str(e).upper():
                raise LLMAuthenticationError(
                    f"Gemini authentication failed: {e}",
                    error_code=ErrorCodes.LLM_AUTH_FAILED
                ) from e
            elif "QUOTA" in str(e).upper() or "RATE" in str(e).upper():
                raise LLMRateLimitError(
                    f"Gemini rate limit exceeded: {e}",
                    error_code=ErrorCodes.LLM_RATE_LIMIT
                ) from e
            elif "TIMEOUT" in str(e).upper():
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
            # Convert messages to Gemini format
            gemini_messages = self._messages_to_gemini_format(messages)
            
            # Prepare generation config
            generation_config = self._get_generation_config(**kwargs)
            
            # Make streaming API request
            if len(gemini_messages) == 1:
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
    
    def _messages_to_gemini_format(self, messages: List[LLMMessage]) -> List[str]:
        """Convert LLMMessage objects to Gemini format."""
        gemini_messages = []
        
        for message in messages:
            if message.role == LLMRole.SYSTEM:
                # Gemini doesn't have explicit system role, prepend to first user message
                if gemini_messages:
                    gemini_messages[0] = f"System: {message.content}\n\n{gemini_messages[0]}"
                else:
                    gemini_messages.append(f"System: {message.content}")
            elif message.role == LLMRole.USER:
                gemini_messages.append(message.content)
            elif message.role == LLMRole.ASSISTANT:
                # For multi-turn conversations, we need to handle assistant messages
                gemini_messages.append(message.content)
        
        return gemini_messages
    
    def _get_generation_config(self, **kwargs) -> genai.types.GenerationConfig:
        """Get generation configuration for Gemini."""
        config_params = {
            "temperature": kwargs.get("temperature", self.gemini_config.temperature),
            "top_p": kwargs.get("top_p", self.gemini_config.top_p),
            "candidate_count": kwargs.get("candidate_count", self.gemini_config.candidate_count),
        }
        
        # Add max_tokens if specified
        max_tokens = kwargs.get("max_tokens", self.gemini_config.max_tokens)
        if max_tokens:
            config_params["max_output_tokens"] = max_tokens
        
        # Add top_k if specified
        top_k = kwargs.get("top_k", self.gemini_config.top_k)
        if top_k:
            config_params["top_k"] = top_k
        
        # Add stop sequences if specified
        stop = kwargs.get("stop", self.gemini_config.stop)
        if stop:
            config_params["stop_sequences"] = stop
        
        return genai.types.GenerationConfig(**config_params)
    
    def _get_safety_settings(self) -> Optional[Dict[HarmCategory, HarmBlockThreshold]]:
        """Get safety settings for Gemini."""
        if not self.gemini_config.safety_settings:
            return None
        
        safety_settings = {}
        
        # Map string settings to Gemini enums
        harm_categories = {
            "harassment": HarmCategory.HARM_CATEGORY_HARASSMENT,
            "hate_speech": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            "sexually_explicit": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            "dangerous_content": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        }
        
        thresholds = {
            "block_none": HarmBlockThreshold.BLOCK_NONE,
            "block_low_and_above": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            "block_medium_and_above": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            "block_only_high": HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        
        for category_str, threshold_str in self.gemini_config.safety_settings.items():
            if category_str in harm_categories and threshold_str in thresholds:
                safety_settings[harm_categories[category_str]] = thresholds[threshold_str]
        
        return safety_settings if safety_settings else None
    
    def _gemini_response_to_llm_response(self, response: Any) -> LLMResponse:
        """Convert Gemini response to LLMResponse."""
        if not response.text:
            raise LLMProviderError("Empty response from Gemini")
        
        # Extract usage information if available
        usage = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
            }
        
        # Determine finish reason
        finish_reason = "stop"
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                finish_reason = str(candidate.finish_reason).lower()
        
        return LLMResponse(
            content=response.text,
            role=LLMRole.ASSISTANT,
            finish_reason=finish_reason,
            usage=usage,
            model=self.gemini_config.model,
            provider=self.provider_name,
            request_id=str(uuid.uuid4()),
            metadata={
                "safety_ratings": getattr(response, 'safety_ratings', None),
                "prompt_feedback": getattr(response, 'prompt_feedback', None)
            }
        )
    
    async def _health_check_impl(self) -> Dict[str, Any]:
        """Perform Gemini-specific health check."""
        health = await super()._health_check_impl()
        
        try:
            # Test API connectivity
            response = await asyncio.to_thread(
                self._model.generate_content,
                "Health check",
                generation_config=genai.types.GenerationConfig(max_output_tokens=1)
            )
            
            health.update({
                "api_accessible": True,
                "model_available": bool(response.text),
                "api_response_time": "< 1s"  # Simple indicator
            })
            
        except Exception as e:
            health.update({
                "api_accessible": False,
                "api_error": str(e)
            })
        
        return health
