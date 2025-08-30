"""
Core module for LLMBlocks.

This module provides the foundational components including:
- Base block system
- Block registry
- Configuration management
- Logging and error handling
"""

from .base_block import (
    BaseBlock,
    BlockStatus,
    BlockType,
    BlockMetadata,
    BlockConfig
)

from .registry import (
    BlockRegistry,
    BlockInfo,
    get_registry,
    set_registry,
    clear_global_registry
)

from .config import (
    ConfigManager,
    ConfigSource,
    LLMBlocksConfig,
    get_config,
    set_config,
    load_config_from_file,
    get_config_value,
    set_config_value
)

__all__ = [
    # Base block system
    "BaseBlock",
    "BlockStatus", 
    "BlockType",
    "BlockMetadata",
    "BlockConfig",
    
    # Block registry
    "BlockRegistry",
    "BlockInfo",
    "get_registry",
    "set_registry",
    "clear_global_registry",
    
    # Configuration management
    "ConfigManager",
    "ConfigSource",
    "LLMBlocksConfig",
    "get_config",
    "set_config",
    "load_config_from_file",
    "get_config_value",
    "set_config_value",
]
