from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Generator
from dataclasses import dataclass, field
import logging
from pathlib import Path

try:
    from langchain_core.runnables.base import Runnable
except ImportError:
    # Fallback for older LangChain versions
    from langchain.schema.runnable import Runnable

@dataclass
class ComponentConfig:
    """Base configuration for all components."""
    name: str
    description: Optional[str] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseComponent(ABC):
    """Base class for all LLMBlocks components."""
    
    def __init__(self, config: Optional[ComponentConfig] = None, **kwargs):
        self.config = config or ComponentConfig(name=self.__class__.__name__)
        self.logger = logging.getLogger(f"llmblocks.{self.__class__.__name__}")
        self._setup_logging()
        
        # Update config with any additional kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.config.metadata[key] = value
    
    def _setup_logging(self):
        """Setup component-specific logging."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the component."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass
    
    def get_config(self) -> ComponentConfig:
        """Get the component configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update component configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.config.metadata[key] = value
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


class BaseBlock(Runnable, BaseComponent):
    """
    Unified base class for all LLMBlocks that is compatible with LangChain's Runnable interface.
    
    This class provides:
    - Standardized execution methods: invoke(), batch(), stream()
    - Support for chaining with the | operator
    - Context manager support for initialization/cleanup
    - Logging and configuration management
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None, **kwargs):
        BaseComponent.__init__(self, config, **kwargs)
        Runnable.__init__(self)
    
    @abstractmethod
    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute the block with a single input.
        
        Args:
            input: The input to process. Can be a string, dict, or other structured data.
            config: Optional configuration overrides.
            
        Returns:
            The processed output.
        """
        pass
    
    def batch(self, inputs: List[Any], config: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Execute the block with multiple inputs in batch.
        
        Args:
            inputs: List of inputs to process.
            config: Optional configuration overrides.
            
        Returns:
            List of processed outputs.
        """
        return [self.invoke(input, config) for input in inputs]
    
    def stream(self, input: Any, config: Optional[Dict[str, Any]] = None) -> Generator[Any, None, None]:
        """
        Execute the block with streaming output.
        
        Args:
            input: The input to process.
            config: Optional configuration overrides.
            
        Yields:
            Streamed output chunks.
        """
        # Default implementation: yield the full result
        result = self.invoke(input, config)
        if isinstance(result, str):
            # For string results, yield character by character for streaming effect
            for char in result:
                yield char
        else:
            # For other types, yield the full result
            yield result
    
    def __repr__(self) -> str:
        """Return a string representation of the block."""
        return f"{self.__class__.__name__}(name='{self.config.name}')"
    
    def __or__(self, other):
        """
        Enable chaining with the | operator.
        
        Args:
            other: Another Runnable or BaseBlock to chain with.
            
        Returns:
            A chained Runnable.
        """
        return self.chain(other)