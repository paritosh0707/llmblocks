"""
Exception hierarchy for LLMBlocks.

This module provides a comprehensive set of exceptions for different
error scenarios in the LLMBlocks system.
"""

from typing import Optional, Any, Dict


class LLMBlocksError(Exception):
    """Base exception for all LLMBlocks errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message='{self.message}', error_code='{self.error_code}')"


# Configuration and Validation Errors

class ConfigurationError(LLMBlocksError):
    """Base exception for configuration-related errors."""
    pass


class BlockValidationError(ConfigurationError):
    """Raised when block configuration validation fails."""
    pass


class ConfigurationNotFoundError(ConfigurationError):
    """Raised when a required configuration file or section is not found."""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration is invalid or malformed."""
    pass


# Block System Errors

class BlockError(LLMBlocksError):
    """Base exception for block-related errors."""
    pass


class BlockInitializationError(BlockError):
    """Raised when block initialization fails."""
    pass


class BlockStartError(BlockError):
    """Raised when block start operation fails."""
    pass


class BlockStopError(BlockError):
    """Raised when block stop operation fails."""
    pass


class BlockNotReadyError(BlockError):
    """Raised when attempting to use a block that is not ready."""
    pass


class BlockNotFoundError(BlockError):
    """Raised when a requested block is not found in the registry."""
    pass


class BlockDependencyError(BlockError):
    """Raised when block dependencies cannot be resolved."""
    pass


# LLM Provider Errors

class LLMProviderError(LLMBlocksError):
    """Base exception for LLM provider errors."""
    pass


class LLMConnectionError(LLMProviderError):
    """Raised when connection to LLM service fails."""
    pass


class LLMAuthenticationError(LLMProviderError):
    """Raised when LLM authentication fails."""
    pass


class LLMRateLimitError(LLMProviderError):
    """Raised when LLM rate limits are exceeded."""
    pass


class LLMQuotaExceededError(LLMProviderError):
    """Raised when LLM quota is exceeded."""
    pass


class LLMTimeoutError(LLMProviderError):
    """Raised when LLM request times out."""
    pass


class LLMResponseError(LLMProviderError):
    """Raised when LLM returns an invalid or unexpected response."""
    pass


# Memory System Errors

class MemoryError(LLMBlocksError):
    """Base exception for memory system errors."""
    pass


class MemoryConnectionError(MemoryError):
    """Raised when connection to memory backend fails."""
    pass


class MemoryAuthenticationError(MemoryError):
    """Raised when memory backend authentication fails."""
    pass


class MemoryQuotaExceededError(MemoryError):
    """Raised when memory quota is exceeded."""
    pass


class MemorySerializationError(MemoryError):
    """Raised when memory serialization/deserialization fails."""
    pass


# Tool System Errors

class ToolError(LLMBlocksError):
    """Base exception for tool-related errors."""
    pass


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not found."""
    pass


class ToolExecutionError(ToolError):
    """Raised when tool execution fails."""
    pass


class ToolValidationError(ToolError):
    """Raised when tool input validation fails."""
    pass


class ToolTimeoutError(ToolError):
    """Raised when tool execution times out."""
    pass


# RAG System Errors

class RAGError(LLMBlocksError):
    """Base exception for RAG system errors."""
    pass


class DocumentProcessingError(RAGError):
    """Raised when document processing fails."""
    pass


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""
    pass


class VectorStoreError(RAGError):
    """Raised when vector store operations fail."""
    pass


class RetrievalError(RAGError):
    """Raised when document retrieval fails."""
    pass


# Agent System Errors

class AgentError(LLMBlocksError):
    """Base exception for agent system errors."""
    pass


class AgentPlanningError(AgentError):
    """Raised when agent planning fails."""
    pass


class AgentExecutionError(AgentError):
    """Raised when agent execution fails."""
    pass


class AgentToolError(AgentError):
    """Raised when agent tool usage fails."""
    pass


# Workflow Errors

class WorkflowError(LLMBlocksError):
    """Base exception for workflow errors."""
    pass


class WorkflowValidationError(WorkflowError):
    """Raised when workflow validation fails."""
    pass


class WorkflowExecutionError(WorkflowError):
    """Raised when workflow execution fails."""
    pass


class WorkflowTimeoutError(WorkflowError):
    """Raised when workflow execution times out."""
    pass


# CLI and Playground Errors

class CLIError(LLMBlocksError):
    """Base exception for CLI-related errors."""
    pass


class PlaygroundError(LLMBlocksError):
    """Base exception for playground-related errors."""
    pass


# Network and External Service Errors

class NetworkError(LLMBlocksError):
    """Base exception for network-related errors."""
    pass


class ExternalServiceError(LLMBlocksError):
    """Base exception for external service errors."""
    pass


class APIError(ExternalServiceError):
    """Raised when external API calls fail."""
    pass


# Utility function to create detailed error messages

def create_error_message(
    error_type: str,
    operation: str,
    details: Optional[Dict[str, Any]] = None,
    suggestion: Optional[str] = None
) -> str:
    """
    Create a detailed error message with context.
    
    Args:
        error_type: Type of error that occurred
        operation: Operation that was being performed
        details: Additional error details
        suggestion: Suggested solution or workaround
        
    Returns:
        Formatted error message
    """
    message = f"{error_type} occurred during {operation}"
    
    if details:
        detail_str = ", ".join(f"{k}: {v}" for k, v in details.items())
        message += f" - {detail_str}"
    
    if suggestion:
        message += f"\n\nSuggestion: {suggestion}"
    
    return message


# Error code constants

class ErrorCodes:
    """Standard error codes for LLMBlocks."""
    
    # Configuration errors
    CONFIG_VALIDATION_FAILED = "CONFIG_001"
    CONFIG_NOT_FOUND = "CONFIG_002"
    CONFIG_INVALID = "CONFIG_003"
    
    # Block errors
    BLOCK_INIT_FAILED = "BLOCK_001"
    BLOCK_START_FAILED = "BLOCK_002"
    BLOCK_STOP_FAILED = "BLOCK_003"
    BLOCK_NOT_READY = "BLOCK_004"
    BLOCK_NOT_FOUND = "BLOCK_005"
    BLOCK_DEPENDENCY_FAILED = "BLOCK_006"
    
    # LLM provider errors
    LLM_CONNECTION_FAILED = "LLM_001"
    LLM_AUTH_FAILED = "LLM_002"
    LLM_RATE_LIMIT = "LLM_003"
    LLM_QUOTA_EXCEEDED = "LLM_004"
    LLM_TIMEOUT = "LLM_005"
    LLM_RESPONSE_INVALID = "LLM_006"
    
    # Memory errors
    MEMORY_CONNECTION_FAILED = "MEM_001"
    MEMORY_AUTH_FAILED = "MEM_002"
    MEMORY_QUOTA_EXCEEDED = "MEM_003"
    MEMORY_SERIALIZATION_FAILED = "MEM_004"
    
    # Tool errors
    TOOL_NOT_FOUND = "TOOL_001"
    TOOL_EXECUTION_FAILED = "TOOL_002"
    TOOL_VALIDATION_FAILED = "TOOL_003"
    TOOL_TIMEOUT = "TOOL_004"
    
    # RAG errors
    RAG_DOCUMENT_PROCESSING_FAILED = "RAG_001"
    RAG_EMBEDDING_FAILED = "RAG_002"
    RAG_VECTOR_STORE_FAILED = "RAG_003"
    RAG_RETRIEVAL_FAILED = "RAG_004"
    
    # Agent errors
    AGENT_PLANNING_FAILED = "AGENT_001"
    AGENT_EXECUTION_FAILED = "AGENT_002"
    AGENT_TOOL_FAILED = "AGENT_003"
    
    # Workflow errors
    WORKFLOW_VALIDATION_FAILED = "WORKFLOW_001"
    WORKFLOW_EXECUTION_FAILED = "WORKFLOW_002"
    WORKFLOW_TIMEOUT = "WORKFLOW_003"
