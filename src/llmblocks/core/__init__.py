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

from .utils import (
    generate_id,
    get_timestamp,
    format_timestamp,
    hash_dict,
    load_yaml_file,
    save_yaml_file,
    load_json_file,
    save_json_file,
    merge_dicts,
    flatten_dict,
    unflatten_dict,
    safe_get,
    safe_set,
    retry_async,
    measure_time,
    validate_url,
    sanitize_filename,
    chunk_list,
    deep_merge
)

from .tracing import (
    TraceLevel,
    TraceEvent,
    TraceCollector,
    get_tracer,
    set_tracer,
    trace_span,
    trace_async_span,
    trace_function,
    trace_event,
    trace_error,
    trace_llm_call,
    trace_block_lifecycle
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
    
    # Utilities
    "generate_id",
    "get_timestamp",
    "format_timestamp",
    "hash_dict",
    "load_yaml_file",
    "save_yaml_file",
    "load_json_file",
    "save_json_file",
    "merge_dicts",
    "flatten_dict",
    "unflatten_dict",
    "safe_get",
    "safe_set",
    "retry_async",
    "measure_time",
    "validate_url",
    "sanitize_filename",
    "chunk_list",
    "deep_merge",
    
    # Tracing
    "TraceLevel",
    "TraceEvent",
    "TraceCollector",
    "get_tracer",
    "set_tracer",
    "trace_span",
    "trace_async_span",
    "trace_function",
    "trace_event",
    "trace_error",
    "trace_llm_call",
    "trace_block_lifecycle",
]
