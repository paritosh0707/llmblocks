"""
Block Registry System for LLMBlocks.

This module provides a centralized registry for managing different types
of blocks, including discovery, registration, and lifecycle management.
"""

import asyncio
from typing import Any, Dict, List, Optional, Type, Union, Callable
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import inspect
import uuid

from .base_block import BaseBlock, BlockType, BlockStatus
from .config import ConfigManager
from ..utils.logging import get_logger
from ..utils.exceptions import (
    BlockNotFoundError,
    BlockDependencyError,
    BlockInitializationError
)


@dataclass
class BlockInfo:
    """Information about a registered block."""
    block_id: str
    name: str
    block_type: BlockType
    class_path: str
    description: str
    version: str
    author: str
    tags: List[str]
    dependencies: List[str]
    config_schema: Dict[str, Any]
    registered_at: datetime
    last_used: Optional[datetime] = None
    usage_count: int = 0
    is_active: bool = True


class BlockRegistry:
    """
    Central registry for managing blocks in the LLMBlocks system.
    
    This registry provides:
    - Dynamic block discovery and registration
    - Dependency resolution and management
    - Block lifecycle tracking
    - Configuration validation
    - Performance monitoring
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the block registry.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.logger = get_logger("BlockRegistry")
        
        # Block storage
        self._blocks: Dict[str, BlockInfo] = {}
        self._block_instances: Dict[str, BaseBlock] = {}
        self._block_classes: Dict[str, Type[BaseBlock]] = {}
        
        # Type-based indexing
        self._blocks_by_type: Dict[BlockType, List[str]] = defaultdict(list)
        self._blocks_by_tag: Dict[str, List[str]] = defaultdict(list)
        
        # Dependency tracking
        self._dependencies: Dict[str, List[str]] = defaultdict(list)
        self._reverse_dependencies: Dict[str, List[str]] = defaultdict(list)
        
        # Registry metadata
        self._registry_id = str(uuid.uuid4())
        self._created_at = datetime.utcnow()
        self._last_updated = datetime.utcnow()
        
        self.logger.info(
            "Block registry initialized",
            registry_id=self._registry_id,
            created_at=self._created_at.isoformat()
        )
    
    @property
    def registry_id(self) -> str:
        """Get the registry ID."""
        return self._registry_id
    
    @property
    def total_blocks(self) -> int:
        """Get the total number of registered blocks."""
        return len(self._blocks)
    
    @property
    def active_blocks(self) -> int:
        """Get the number of active blocks."""
        return sum(1 for block in self._blocks.values() if block.is_active)
    
    @property
    def registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "registry_id": self._registry_id,
            "total_blocks": self.total_blocks,
            "active_blocks": self.active_blocks,
            "blocks_by_type": {
                block_type.value: len(block_ids)
                for block_type, block_ids in self._blocks_by_type.items()
            },
            "created_at": self._created_at.isoformat(),
            "last_updated": self._last_updated.isoformat()
        }
    
    def register_block(
        self,
        block_class: Type[BaseBlock],
        name: Optional[str] = None,
        description: Optional[str] = None,
        version: str = "1.0.0",
        author: str = "",
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        config_schema: Optional[Dict[str, Any]] = None,
        override: bool = False
    ) -> str:
        """
        Register a block class in the registry.
        
        Args:
            block_class: The block class to register
            name: Custom name for the block (defaults to class name)
            description: Block description
            version: Block version
            author: Block author
            tags: List of tags for categorization
            dependencies: List of dependency block names
            config_schema: Configuration schema for the block
            override: Whether to override existing registration
            
        Returns:
            Block ID of the registered block
            
        Raises:
            ValueError: If block is already registered and override is False
        """
        # Validate block class
        if not inspect.isclass(block_class) or not issubclass(block_class, BaseBlock):
            raise ValueError(f"Invalid block class: {block_class}")
        
        # Generate block info
        block_name = name or block_class.__name__
        block_id = f"{block_name}_{version}_{uuid.uuid4().hex[:8]}"
        
        # Check if block is already registered
        if block_name in [b.name for b in self._blocks.values()] and not override:
            raise ValueError(f"Block '{block_name}' is already registered")
        
        # Create block info
        block_info = BlockInfo(
            block_id=block_id,
            name=block_name,
            block_type=block_class().block_type,
            class_path=f"{block_class.__module__}.{block_class.__name__}",
            description=description or block_class.__doc__ or "",
            version=version,
            author=author,
            tags=tags or [],
            dependencies=dependencies or [],
            config_schema=config_schema or {},
            registered_at=datetime.utcnow()
        )
        
        # Register the block
        self._blocks[block_id] = block_info
        self._block_classes[block_id] = block_class
        
        # Update type and tag indices
        self._blocks_by_type[block_info.block_type].append(block_id)
        for tag in block_info.tags:
            self._blocks_by_tag[tag].append(block_id)
        
        # Update dependencies
        self._dependencies[block_id] = dependencies or []
        for dep in dependencies or []:
            self._reverse_dependencies[dep].append(block_id)
        
        # Update registry
        self._last_updated = datetime.utcnow()
        
        self.logger.info(
            "Block registered",
            block_id=block_id,
            name=block_name,
            block_type=block_info.block_type.value,
            tags=tags,
            dependencies=dependencies
        )
        
        return block_id
    
    def unregister_block(self, block_id: str) -> bool:
        """
        Unregister a block from the registry.
        
        Args:
            block_id: ID of the block to unregister
            
        Returns:
            True if block was unregistered, False if not found
        """
        if block_id not in self._blocks:
            return False
        
        block_info = self._blocks[block_id]
        
        # Remove from type and tag indices
        self._blocks_by_type[block_info.block_type].remove(block_id)
        for tag in block_info.tags:
            if block_id in self._blocks_by_tag[tag]:
                self._blocks_by_tag[tag].remove(block_id)
        
        # Remove dependencies
        if block_id in self._dependencies:
            del self._dependencies[block_id]
        if block_id in self._reverse_dependencies:
            del self._reverse_dependencies[block_id]
        
        # Remove block instances
        if block_id in self._block_instances:
            del self._block_instances[block_id]
        
        # Remove block class
        if block_id in self._block_classes:
            del self._block_classes[block_id]
        
        # Remove block info
        del self._blocks[block_id]
        
        # Update registry
        self._last_updated = datetime.utcnow()
        
        self.logger.info(
            "Block unregistered",
            block_id=block_id,
            name=block_info.name
        )
        
        return True
    
    def get_block_info(self, block_id: str) -> Optional[BlockInfo]:
        """
        Get information about a registered block.
        
        Args:
            block_id: ID of the block
            
        Returns:
            BlockInfo if found, None otherwise
        """
        return self._blocks.get(block_id)
    
    def get_block_class(self, block_id: str) -> Optional[Type[BaseBlock]]:
        """
        Get the class of a registered block.
        
        Args:
            block_id: ID of the block
            
        Returns:
            Block class if found, None otherwise
        """
        return self._block_classes.get(block_id)
    
    def find_blocks(
        self,
        block_type: Optional[BlockType] = None,
        tags: Optional[List[str]] = None,
        name_pattern: Optional[str] = None,
        active_only: bool = True
    ) -> List[BlockInfo]:
        """
        Find blocks matching specified criteria.
        
        Args:
            block_type: Filter by block type
            tags: Filter by tags (all must match)
            name_pattern: Filter by name pattern
            active_only: Only return active blocks
            
        Returns:
            List of matching BlockInfo objects
        """
        candidates = []
        
        # Start with all blocks or filter by type
        if block_type:
            candidate_ids = self._blocks_by_type.get(block_type, [])
        else:
            candidate_ids = list(self._blocks.keys())
        
        # Filter by criteria
        for block_id in candidate_ids:
            block_info = self._blocks[block_id]
            
            # Check if block is active (if required)
            if active_only and not block_info.is_active:
                continue
            
            # Check tags
            if tags and not all(tag in block_info.tags for tag in tags):
                continue
            
            # Check name pattern
            if name_pattern and name_pattern.lower() not in block_info.name.lower():
                continue
            
            candidates.append(block_info)
        
        return candidates
    
    def list_blocks(
        self,
        block_type: Optional[BlockType] = None,
        include_inactive: bool = False
    ) -> List[BlockInfo]:
        """
        List all blocks of a specific type.
        
        Args:
            block_type: Block type to list (None for all types)
            include_inactive: Whether to include inactive blocks
            
        Returns:
            List of BlockInfo objects
        """
        if block_type:
            block_ids = self._blocks_by_type.get(block_type, [])
        else:
            block_ids = list(self._blocks.keys())
        
        blocks = [self._blocks[block_id] for block_id in block_ids]
        
        if not include_inactive:
            blocks = [block for block in blocks if block.is_active]
        
        return sorted(blocks, key=lambda x: x.name)
    
    def list_block_types(self) -> List[BlockType]:
        """List all registered block types."""
        return list(self._blocks_by_type.keys())
    
    def list_tags(self) -> List[str]:
        """List all registered tags."""
        return list(self._blocks_by_tag.keys())
    
    async def create_block_instance(
        self,
        block_id: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseBlock:
        """
        Create an instance of a registered block.
        
        Args:
            block_id: ID of the block to instantiate
            config: Block configuration
            **kwargs: Additional configuration parameters
            
        Returns:
            Block instance
            
        Raises:
            BlockNotFoundError: If block is not found
            BlockInitializationError: If block initialization fails
        """
        if block_id not in self._blocks:
            raise BlockNotFoundError(f"Block '{block_id}' not found in registry")
        
        block_info = self._blocks[block_id]
        block_class = self._block_classes[block_id]
        
        try:
            # Create block instance
            block_instance = block_class(config=config, **kwargs)
            
            # Initialize the block
            await block_instance.initialize()
            
            # Store instance
            self._block_instances[block_id] = block_instance
            
            # Update usage statistics
            block_info.last_used = datetime.utcnow()
            block_info.usage_count += 1
            
            self.logger.info(
                "Block instance created",
                block_id=block_id,
                name=block_info.name,
                block_type=block_info.block_type.value
            )
            
            return block_instance
            
        except Exception as e:
            self.logger.error(
                "Failed to create block instance",
                block_id=block_id,
                name=block_info.name,
                error=str(e)
            )
            raise BlockInitializationError(f"Failed to create block instance: {e}") from e
    
    def get_block_instance(self, block_id: str) -> Optional[BaseBlock]:
        """
        Get an existing block instance.
        
        Args:
            block_id: ID of the block
            
        Returns:
            Block instance if exists, None otherwise
        """
        return self._block_instances.get(block_id)
    
    async def destroy_block_instance(self, block_id: str) -> bool:
        """
        Destroy a block instance and clean up resources.
        
        Args:
            block_id: ID of the block
            
        Returns:
            True if instance was destroyed, False if not found
        """
        if block_id not in self._block_instances:
            return False
        
        block_instance = self._block_instances[block_id]
        
        try:
            # Stop and cleanup the block
            await block_instance.stop()
            await block_instance.cleanup()
            
            # Remove from instances
            del self._block_instances[block_id]
            
            self.logger.info(
                "Block instance destroyed",
                block_id=block_id
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to destroy block instance",
                block_id=block_id,
                error=str(e)
            )
            return False
    
    def check_dependencies(self, block_id: str) -> List[str]:
        """
        Check if a block's dependencies are satisfied.
        
        Args:
            block_id: ID of the block to check
            
        Returns:
            List of missing dependency names
        """
        if block_id not in self._dependencies:
            return []
        
        missing = []
        for dep_name in self._dependencies[block_id]:
            # Check if dependency exists in registry
            if not any(block.name == dep_name for block in self._blocks.values()):
                missing.append(dep_name)
        
        return missing
    
    def get_dependency_tree(self, block_id: str) -> Dict[str, Any]:
        """
        Get the dependency tree for a block.
        
        Args:
            block_id: ID of the block
            
        Returns:
            Dependency tree structure
        """
        if block_id not in self._blocks:
            return {}
        
        def build_tree(block_id: str, visited: set) -> Dict[str, Any]:
            if block_id in visited:
                return {"circular": True}
            
            visited.add(block_id)
            block_info = self._blocks[block_id]
            
            tree = {
                "block_id": block_id,
                "name": block_info.name,
                "dependencies": []
            }
            
            for dep_name in block_info.dependencies:
                # Find dependency block
                dep_block = next(
                    (b for b in self._blocks.values() if b.name == dep_name),
                    None
                )
                
                if dep_block:
                    tree["dependencies"].append(
                        build_tree(dep_block.block_id, visited.copy())
                    )
                else:
                    tree["dependencies"].append({
                        "name": dep_name,
                        "status": "missing"
                    })
            
            return tree
        
        return build_tree(block_id, set())
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all active block instances.
        
        Returns:
            Health status for all blocks
        """
        health_status = {
            "registry_healthy": True,
            "total_blocks": self.total_blocks,
            "active_instances": len(self._block_instances),
            "blocks": {}
        }
        
        for block_id, block_instance in self._block_instances.items():
            try:
                block_health = await block_instance.health_check()
                health_status["blocks"][block_id] = block_health
                
                if not block_health.get("is_healthy", True):
                    health_status["registry_healthy"] = False
                    
            except Exception as e:
                health_status["blocks"][block_id] = {
                    "error": str(e),
                    "is_healthy": False
                }
                health_status["registry_healthy"] = False
        
        return health_status
    
    def export_registry(self) -> Dict[str, Any]:
        """
        Export the registry state for persistence or debugging.
        
        Returns:
            Registry export data
        """
        return {
            "registry_id": self._registry_id,
            "created_at": self._created_at.isoformat(),
            "last_updated": self._last_updated.isoformat(),
            "blocks": {
                block_id: {
                    "name": info.name,
                    "block_type": info.block_type.value,
                    "class_path": info.class_path,
                    "description": info.description,
                    "version": info.version,
                    "author": info.author,
                    "tags": info.tags,
                    "dependencies": info.dependencies,
                    "config_schema": info.config_schema,
                    "registered_at": info.registered_at.isoformat(),
                    "last_used": info.last_used.isoformat() if info.last_used else None,
                    "usage_count": info.usage_count,
                    "is_active": info.is_active
                }
                for block_id, info in self._blocks.items()
            },
            "dependencies": dict(self._dependencies),
            "reverse_dependencies": dict(self._reverse_dependencies)
        }
    
    def clear_registry(self) -> None:
        """Clear all registered blocks and instances."""
        # Destroy all instances
        for block_id in list(self._block_instances.keys()):
            asyncio.create_task(self.destroy_block_instance(block_id))
        
        # Clear all data structures
        self._blocks.clear()
        self._block_instances.clear()
        self._block_classes.clear()
        self._blocks_by_type.clear()
        self._blocks_by_tag.clear()
        self._dependencies.clear()
        self._reverse_dependencies.clear()
        
        self._last_updated = datetime.utcnow()
        
        self.logger.info("Registry cleared")
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        return (
            f"BlockRegistry("
            f"id={self._registry_id}, "
            f"blocks={self.total_blocks}, "
            f"types={len(self._blocks_by_type)})"
        )


# Global registry instance
_global_registry: Optional[BlockRegistry] = None


def get_registry() -> BlockRegistry:
    """Get the global block registry instance."""
    global _global_registry
    
    if _global_registry is None:
        _global_registry = BlockRegistry()
    
    return _global_registry


def set_registry(registry: BlockRegistry) -> None:
    """Set the global block registry instance."""
    global _global_registry
    _global_registry = registry


def clear_global_registry() -> None:
    """Clear the global block registry."""
    global _global_registry
    if _global_registry:
        _global_registry.clear_registry()
        _global_registry = None
