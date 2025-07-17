from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, SecretStr
from langchain_openai import AzureChatOpenAI
from llmblocks.blocks.llm_provider.base import BaseLLMProvider


class AzureOpenAICredentials(BaseModel):
    """
    Pydantic model for validating Azure OpenAI API credentials and configuration.
    
    This model ensures all parameters are valid before instantiating the Azure OpenAI LLM,
    providing clear error messages for any validation failures.
    
    Required Fields:
    - api_key: Your Azure OpenAI API key
    - api_version: Azure OpenAI API version (e.g., "2023-05-15")
    - deployment_endpoint: Azure OpenAI endpoint URL
    - deployment_name: Name of your Azure OpenAI deployment (e.g., "gpt-4", "gpt-35-turbo")
    
    Optional Fields (with defaults):
    - temperature: Controls randomness in responses, 0.0-2.0 (default: 0.7)
    - max_tokens: Maximum tokens in response (default: None, uses model's limit)
    - n: Number of completions to generate (default: 1)
    - stop: Stop sequences to end generation (default: None)
    """
    
    # Required fields - no default values
    api_key: SecretStr = Field(
        description="Azure OpenAI API key for authentication"
    )
    api_version: str = Field(
        default="2023-05-15",
        description="Azure OpenAI API version (e.g., '2023-05-15')"
    )
    deployment_endpoint: str = Field(
        description="Azure OpenAI endpoint URL"
    )
    deployment_name: str = Field(
        description="Azure OpenAI deployment name (e.g., 'gpt-4', 'gpt-35-turbo')"
    )
    
    # Optional fields with defaults
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
        if not v or not v.strip():
            raise ValueError("Azure OpenAI API key is required and cannot be empty")
        return v.strip()
    
    @field_validator('api_version')
    def validate_api_version(cls, v):
        """Validate that API version is non-empty."""
        if not v or not v.strip():
            raise ValueError("Azure OpenAI API version is required and cannot be empty")
        return v.strip()
    
    @field_validator('deployment_endpoint')
    def validate_deployment_endpoint(cls, v):
        """Validate that Azure endpoint is non-empty."""
        if not v or not v.strip():
            raise ValueError("Azure OpenAI endpoint is required and cannot be empty")
        return v.strip()
    
    @field_validator('deployment_name')
    def validate_deployment_name(cls, v):
        """Validate that deployment name is non-empty."""
        if not v or not v.strip():
            raise ValueError("Azure OpenAI deployment name is required and cannot be empty")
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


class AzureOpenAIProvider(BaseLLMProvider):
    """
    A robust wrapper around LangChain's Azure OpenAI LLM with Pydantic validation.
    
    This class provides a clean interface for instantiating Azure OpenAI LLMs with
    comprehensive parameter validation and clear error messages.
    """
    
    @property
    def PROVIDER_NAME(self) -> Literal["azure_openai"]:
        return "azure_openai"
    
    def __init__(self, **kwargs):
        """
        Initialize the Azure OpenAI provider with validated credentials.
        
        Args:
            **kwargs: Configuration parameters to pass to AzureOpenAICredentials
            
        Raises:
            ValueError: If validation fails or LLM instantiation fails
        """
        try:
            # Parse and validate credentials
            self.credentials = AzureOpenAICredentials(**kwargs)
            
            # Extract validated parameters for LangChain
            llm_kwargs = {
                'api_key': self.credentials.api_key.get_secret_value(),
                'api_version': self.credentials.api_version,
                'azure_endpoint': self.credentials.deployment_endpoint,
                'azure_deployment': self.credentials.deployment_name,
                'temperature': self.credentials.temperature,
                'n': self.credentials.n,
            }
            
            # Add optional parameters if provided
            if self.credentials.max_tokens is not None:
                llm_kwargs['max_tokens'] = self.credentials.max_tokens
            if self.credentials.stop is not None:
                llm_kwargs['stop'] = self.credentials.stop
            
            # Instantiate the LangChain Azure OpenAI LLM
            self.llm = AzureChatOpenAI(**llm_kwargs)
            
        except Exception as err:
            # Re-raise with clear error message
            raise ValueError(f"Invalid Azure OpenAI credentials: {err}")
    
    def get_llm(self):
        """
        Get the underlying LangChain Azure OpenAI LLM instance.
        
        Returns:
            The LangChain Azure OpenAI LLM instance
        """
        return self.llm


if __name__ == "__main__":
    pass 