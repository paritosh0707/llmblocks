"""
Gemini Provider Module

This module provides a robust, Pydantic-validated wrapper around LangChain's Gemini LLM.
It includes comprehensive validation for all Gemini configuration parameters and
user-friendly error messages for debugging configuration issues.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, SecretStr
from langchain_google_genai import ChatGoogleGenerativeAI
from llmblocks.blocks.llm_provider.base import BaseLLMProvider


class GeminiCredentials(BaseModel):
    """
    Pydantic model for validating Google Gemini API credentials and configuration.
    
    This model ensures all parameters are valid before instantiating the Gemini LLM,
    providing clear error messages for any validation failures.
    
    Required Fields:
    - api_key: Your Google API key for Gemini access
    
    Optional Fields (with defaults):
    - model_name: Gemini model to use (default: "gemini-pro")
    - temperature: Controls randomness in responses, 0.0-2.0 (default: 0.7)
    - max_tokens: Maximum tokens in response (default: None, uses model's limit)
    - n: Number of completions to generate (default: 1)
    - stop: Stop sequences to end generation (default: None)
    """
    
    # Required field - no default value
    api_key: SecretStr = Field(
        description="Google API key for Gemini access"
    )
    
    # Optional fields with defaults
    model_name: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model identifier (e.g., 'gemini-2.0-flash', 'gemini-2.0-flash-lite')"
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
    
    @field_validator('api_key')
    def validate_api_key(cls, v):
        """Validate that API key is non-empty."""
        if not v or not v.get_secret_value().strip():
            raise ValueError("Google API key is required and cannot be empty")
        return v
    
    @field_validator('model_name')
    def validate_model_name(cls, v):
        """Validate that model name is a valid Gemini model identifier."""
        if not v or not v.strip():
            raise ValueError("model_name must be a non-empty string")
        return v.strip()
    
    @field_validator('temperature')
    def validate_temperature(cls, v):
        """Validate temperature is within valid range."""
        if not (0.0 <= v <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v
    
    @field_validator('n')
    def validate_n(cls, v):
        """Validate number of completions is at least 1."""
        if v < 1:
            raise ValueError("n (number of completions) must be at least 1")
        return v
    
    @field_validator('max_tokens')
    def validate_max_tokens(cls, v):
        """Validate max_tokens is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("max_tokens must be a positive integer")
        return v
    
    @field_validator('stop')
    def validate_stop(cls, v):
        """Validate stop sequences are non-empty strings if provided."""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("stop sequences must be a list of non-empty strings")
            for seq in v:
                if not isinstance(seq, str) or not seq.strip():
                    raise ValueError("stop sequences must be a list of non-empty strings")
        return v


class GeminiProvider(BaseLLMProvider):
    """
    A robust wrapper around LangChain's Gemini LLM with Pydantic validation.
    
    This class provides a clean interface for instantiating Gemini LLMs with
    comprehensive parameter validation and clear error messages.
    """
    
    @property
    def PROVIDER_NAME(self) -> Literal["gemini"]:
        return "gemini"
    
    def __init__(self, **kwargs):
        """
        Initialize the Gemini provider with validated credentials.
        
        Args:
            **kwargs: Configuration parameters to pass to GeminiCredentials
            
        Raises:
            ValueError: If validation fails or LLM instantiation fails
        """
        try:
            # Parse and validate credentials
            self.credentials = GeminiCredentials(**kwargs)
            
            # Extract validated parameters for LangChain
            llm_kwargs = {
                'google_api_key': self.credentials.api_key.get_secret_value(),
                'model': self.credentials.model_name,
                'temperature': self.credentials.temperature,
                'n': self.credentials.n,
            }
            
            # Add optional parameters if provided
            if self.credentials.max_tokens is not None:
                llm_kwargs['max_output_tokens'] = self.credentials.max_tokens
            if self.credentials.stop is not None:
                llm_kwargs['stop'] = self.credentials.stop
            
            # Instantiate the LangChain Gemini LLM
            self.llm = ChatGoogleGenerativeAI(**llm_kwargs)
            
        except Exception as err:
            # Re-raise with clear error message
            raise ValueError(f"Invalid Gemini credentials: {err}")
    
    def get_llm(self):
        """
        Get the underlying LangChain Gemini LLM instance.
        
        Returns:
            The LangChain Gemini LLM instance
        """
        return self.llm


if __name__ == "__main__":
    pass