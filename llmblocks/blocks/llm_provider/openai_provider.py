"""
OpenAI Provider Module

This module provides a robust, Pydantic-validated wrapper around LangChain's OpenAI LLM.
It includes comprehensive validation for all OpenAI configuration parameters and
user-friendly error messages for debugging configuration issues.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator
from langchain.llms import OpenAI
from llmblocks.blocks.llm_provider.base import BaseLLMProvider


class OpenAICredentials(BaseModel):
    """
    Pydantic model for validating OpenAI API credentials and configuration.
    
    This model ensures all parameters are valid before instantiating the OpenAI LLM,
    providing clear error messages for any validation failures.
    
    Required Fields:
    - api_key: Your OpenAI API key (required for authentication)
    
    Optional Fields (with defaults):
    - model_name: OpenAI model to use (default: "gpt-3.5-turbo")
    - temperature: Controls randomness in responses, 0.0-2.0 (default: 0.7)
    - max_tokens: Maximum tokens in response (default: None, uses model's limit)
    - n: Number of completions to generate (default: 1)
    - stop: Stop sequences to end generation (default: None)
    """
    
    # Required field - no default value
    api_key: str = Field(
        description="OpenAI API key for authentication (required)"
    )
    
    # Optional fields with defaults
    model_name: str = Field(
        default="gpt-4o",
        description="OpenAI model identifier (e.g., 'gpt-4o', 'gpt-4o-mini')"
    )
    temperature: float = Field(
        default=0.7,
        description="Controls randomness in responses (0.0 = deterministic, 2.0 = very random)"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate in the response"
    )
    n: int = Field(
        default=1,
        description="Number of completions to generate for each prompt"
    )
    stop: Optional[List[str]] = Field(
        default=None,
        description="List of stop sequences that will end text generation"
    )
    
    @validator('api_key')
    def validate_api_key(cls, v):
        """Validate that API key is non-empty."""
        if not v or not v.strip():
            raise ValueError("OpenAI API key is required and cannot be empty")
        return v.strip()
    
    @validator('model_name')
    def validate_model_name(cls, v):
        """Validate that model name is a valid OpenAI model identifier."""
        if not v or not v.strip():
            raise ValueError("model_name must be a non-empty string")
        return v.strip()
    
    @validator('temperature')
    def validate_temperature(cls, v):
        """Validate temperature is within valid range."""
        if not (0.0 <= v <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v
    
    @validator('n')
    def validate_n(cls, v):
        """Validate number of completions is at least 1."""
        if v < 1:
            raise ValueError("n (number of completions) must be at least 1")
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        """Validate max_tokens is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("max_tokens must be a positive integer")
        return v
    
    @validator('stop')
    def validate_stop(cls, v):
        """Validate stop sequences are non-empty strings if provided."""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("stop sequences must be a list of non-empty strings")
            for seq in v:
                if not isinstance(seq, str) or not seq.strip():
                    raise ValueError("stop sequences must be a list of non-empty strings")
        return v


class OpenAIProvider(BaseLLMProvider):
    """
    A robust wrapper around LangChain's OpenAI LLM with Pydantic validation.
    
    This class provides a clean interface for instantiating OpenAI LLMs with
    comprehensive parameter validation and clear error messages.
    """
    
    PROVIDER_NAME = "openai"
    
    def __init__(self, **kwargs):
        """
        Initialize the OpenAI provider with validated credentials.
        
        Args:
            **kwargs: Configuration parameters to pass to OpenAICredentials
            
        Raises:
            ValueError: If validation fails or LLM instantiation fails
        """
        try:
            # Parse and validate credentials
            self.credentials = OpenAICredentials(**kwargs)
            
            # Extract validated parameters for LangChain
            llm_kwargs = {
                'openai_api_key': self.credentials.api_key,
                'model_name': self.credentials.model_name,
                'temperature': self.credentials.temperature,
                'n': self.credentials.n,
            }
            
            # Add optional parameters if provided
            if self.credentials.max_tokens is not None:
                llm_kwargs['max_tokens'] = self.credentials.max_tokens
            if self.credentials.stop is not None:
                llm_kwargs['stop'] = self.credentials.stop
            
            # Instantiate the LangChain OpenAI LLM
            self.llm = OpenAI(**llm_kwargs)
            
        except Exception as err:
            # Re-raise with clear error message
            raise ValueError(f"Invalid OpenAI credentials: {err}")
    
    def get_llm(self):
        """
        Get the underlying LangChain OpenAI LLM instance.
        
        Returns:
            The LangChain OpenAI LLM instance
        """
        return self.llm


if __name__ == "__main__":
    # Run examples from the separate examples module
    from .examples import run_examples
    run_examples()
