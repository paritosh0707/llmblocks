"""
Base Block System for LLMBlocks.

This module provides the foundational block interface that all components
inherit from, including lifecycle management, validation, and error handling.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import uuid

from pydantic import BaseModel, Field, ValidationError
from structlog import get_logger

from ..utils.exceptions import BlockInitializationError, BlockValidationError


class BlockStatus(Enum):
    """Status of a block instance."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    STOPPING = "stopping"
    STOPPED = "stopped"


class BlockType(Enum):
    """Type of block for categorization."""
    LLM_PROVIDER = "llm_provider"
    MEMORY = "memory"
    TOOL = "tool"
    RAG = "rag"
    AGENT = "agent"
    WORKFLOW = "workflow"
    COMPONENT = "component"


@dataclass
class BlockMetadata:
    """Metadata for a block instance."""
    block_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)


class BlockConfig(BaseModel):
    """Base configuration for blocks."""
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: bool = True
    timeout: Optional[float] = None
    retry_attempts: int = 3
    retry_delay: float = 1.0
    max_concurrent_requests: Optional[int] = None
    log_level: str = "INFO"
    
    class Config:
        extra = "allow"  # Allow additional configuration fields


class BaseBlock(ABC):
    """
    Base class for all blocks in the LLMBlocks system.
    
    This class provides:
    - Lifecycle management (initialize, start, stop, cleanup)
    - Configuration validation and management
    - Error handling and recovery
    - Logging and monitoring
    - Health checking and status reporting
    """
    
    def __init__(
        self,
        config: Optional[Union[BlockConfig, Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Initialize a new block instance.
        
        Args:
            config: Block configuration (can be dict or BlockConfig instance)
            **kwargs: Additional configuration parameters
        """
        self.metadata = BlockMetadata()
        self.status = BlockStatus.UNINITIALIZED
        self.config = self._validate_config(config, **kwargs)
        self.logger = get_logger(f"{self.__class__.__name__}:{self.metadata.block_id}")
        self._error_count = 0
        self._last_error = None
        self._start_time = None
        self._request_count = 0
        self._error_lock = asyncio.Lock()
        
        # Initialize metadata
        self.metadata.name = self.config.name or self.__class__.__name__
        self.metadata.description = self.config.description or self.__doc__ or ""
        
        self.logger.info(
            "Block initialized",
            block_id=self.metadata.block_id,
            name=self.metadata.name,
            status=self.status.value
        )
    
    @property
    def block_type(self) -> BlockType:
        """Get the type of this block."""
        return BlockType.COMPONENT
    
    @property
    def is_ready(self) -> bool:
        """Check if the block is ready to process requests."""
        return self.status == BlockStatus.READY
    
    @property
    def is_running(self) -> bool:
        """Check if the block is currently running."""
        return self.status == BlockStatus.RUNNING
    
    @property
    def is_error(self) -> bool:
        """Check if the block is in an error state."""
        return self.status == BlockStatus.ERROR
    
    async def initialize(self) -> None:
        """
        Initialize the block and prepare it for operation.
        
        This method should be overridden by subclasses to perform
        any necessary setup, connection establishment, etc.
        """
        if self.status != BlockStatus.UNINITIALIZED:
            raise BlockInitializationError(
                f"Cannot initialize block in {self.status.value} state"
            )
        
        try:
            self.status = BlockStatus.INITIALIZING
            self.logger.info("Initializing block", block_id=self.metadata.block_id)
            
            # Call subclass initialization
            await self._initialize_impl()
            
            self.status = BlockStatus.READY
            self.logger.info(
                "Block initialized successfully",
                block_id=self.metadata.block_id,
                status=self.status.value
            )
            
        except Exception as e:
            self.status = BlockStatus.ERROR
            self._last_error = e
            self.logger.error(
                "Block initialization failed",
                block_id=self.metadata.block_id,
                error=str(e),
                exc_info=True
            )
            raise BlockInitializationError(f"Initialization failed: {e}") from e
    
    async def start(self) -> None:
        """
        Start the block and begin processing requests.
        
        This method should be overridden by subclasses that need
        to start background processes or listeners.
        """
        if self.status != BlockStatus.READY:
            raise BlockInitializationError(
                f"Cannot start block in {self.status.value} state"
            )
        
        try:
            self.status = BlockStatus.RUNNING
            self._start_time = datetime.utcnow()
            self.logger.info("Block started", block_id=self.metadata.block_id)
            
            # Call subclass start implementation
            await self._start_impl()
            
        except Exception as e:
            self.status = BlockStatus.ERROR
            self._last_error = e
            self.logger.error(
                "Block start failed",
                block_id=self.metadata.block_id,
                error=str(e),
                exc_info=True
            )
            raise BlockInitializationError(f"Start failed: {e}") from e
    
    async def stop(self) -> None:
        """
        Stop the block and stop processing requests.
        
        This method should be overridden by subclasses that need
        to stop background processes or listeners.
        """
        if self.status not in [BlockStatus.RUNNING, BlockStatus.READY]:
            return
        
        try:
            self.status = BlockStatus.STOPPING
            self.logger.info("Stopping block", block_id=self.metadata.block_id)
            
            # Call subclass stop implementation
            await self._stop_impl()
            
            self.status = BlockStatus.STOPPED
            self.logger.info("Block stopped", block_id=self.metadata.block_id)
            
        except Exception as e:
            self.status = BlockStatus.ERROR
            self._last_error = e
            self.logger.error(
                "Block stop failed",
                block_id=self.metadata.block_id,
                error=str(e),
                exc_info=True
            )
            raise BlockInitializationError(f"Stop failed: {e}") from e
    
    async def cleanup(self) -> None:
        """
        Clean up resources used by the block.
        
        This method should be overridden by subclasses to perform
        any necessary cleanup, connection closing, etc.
        """
        try:
            self.logger.info("Cleaning up block", block_id=self.metadata.block_id)
            
            # Call subclass cleanup implementation
            await self._cleanup_impl()
            
            self.logger.info("Block cleanup completed", block_id=self.metadata.block_id)
            
        except Exception as e:
            self.logger.error(
                "Block cleanup failed",
                block_id=self.metadata.block_id,
                error=str(e),
                exc_info=True
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the block.
        
        Returns:
            Dictionary containing health status and metrics
        """
        health = {
            "block_id": self.metadata.block_id,
            "name": self.metadata.name,
            "status": self.status.value,
            "is_healthy": self.status in [BlockStatus.READY, BlockStatus.RUNNING],
            "error_count": self._error_count,
            "last_error": str(self._last_error) if self._last_error else None,
            "request_count": self._request_count,
            "uptime": None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self._start_time:
            health["uptime"] = (datetime.utcnow() - self._start_time).total_seconds()
        
        # Call subclass health check implementation
        try:
            custom_health = await self._health_check_impl()
            health.update(custom_health)
        except Exception as e:
            self.logger.warning(
                "Custom health check failed",
                block_id=self.metadata.block_id,
                error=str(e)
            )
        
        return health
    
    async def reset(self) -> None:
        """
        Reset the block to its initial state.
        
        This will clear errors and reset counters, but won't
        reinitialize the block.
        """
        async with self._error_lock:
            self._error_count = 0
            self._last_error = None
            self._request_count = 0
        
        if self.status == BlockStatus.ERROR:
            self.status = BlockStatus.UNINITIALIZED
        
        self.logger.info("Block reset", block_id=self.metadata.block_id)
    
    def _validate_config(
        self,
        config: Optional[Union[BlockConfig, Dict[str, Any]]],
        **kwargs
    ) -> BlockConfig:
        """
        Validate and merge configuration.
        
        Args:
            config: Block configuration
            **kwargs: Additional configuration parameters
            
        Returns:
            Validated BlockConfig instance
        """
        try:
            if isinstance(config, dict):
                config_dict = {**config, **kwargs}
            elif config is not None:
                config_dict = {**config.dict(), **kwargs}
            else:
                config_dict = kwargs
            
            return BlockConfig(**config_dict)
            
        except ValidationError as e:
            raise BlockValidationError(f"Configuration validation failed: {e}") from e
    
    # Abstract methods that subclasses must implement
    
    @abstractmethod
    async def _initialize_impl(self) -> None:
        """
        Subclass-specific initialization logic.
        
        This method should be implemented by subclasses to perform
        any necessary setup, connection establishment, etc.
        """
        pass
    
    async def _start_impl(self) -> None:
        """
        Subclass-specific start logic.
        
        This method can be overridden by subclasses that need
        to start background processes or listeners.
        """
        pass
    
    async def _stop_impl(self) -> None:
        """
        Subclass-specific stop logic.
        
        This method can be overridden by subclasses that need
        to stop background processes or listeners.
        """
        pass
    
    async def _cleanup_impl(self) -> None:
        """
        Subclass-specific cleanup logic.
        
        This method can be overridden by subclasses to perform
        any necessary cleanup, connection closing, etc.
        """
        pass
    
    async def _health_check_impl(self) -> Dict[str, Any]:
        """
        Subclass-specific health check logic.
        
        This method can be overridden by subclasses to provide
        custom health metrics and checks.
        
        Returns:
            Dictionary containing custom health information
        """
        return {}
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        asyncio.create_task(self.cleanup())
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    def __repr__(self) -> str:
        """String representation of the block."""
        return (
            f"{self.__class__.__name__}("
            f"id={self.metadata.block_id}, "
            f"name={self.metadata.name}, "
            f"status={self.status.value})"
        )
