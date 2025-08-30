"""
Configuration Management System for LLMBlocks.

This module provides a comprehensive configuration management system
that handles YAML/JSON configuration files, environment variables,
and configuration validation.
"""

import os
import yaml
import json
from typing import Any, Dict, List, Optional, Union, Set
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, UTC
import re

from pydantic import BaseModel, Field, ValidationError, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..utils.logging import get_logger
from ..utils.exceptions import (
    ConfigurationError,
    ConfigurationNotFoundError,
    InvalidConfigurationError
)


@dataclass
class ConfigSource:
    """Information about a configuration source."""
    source_type: str  # 'file', 'environment', 'default'
    path: Optional[str] = None
    priority: int = 0
    loaded_at: Optional[datetime] = None
    is_valid: bool = True
    error_message: Optional[str] = None


class ConfigValidator(BaseModel):
    """Base configuration validator."""
    model_config = ConfigDict(extra="allow")  # Allow additional fields


class LLMBlocksConfig(BaseSettings):
    """Main configuration for LLMBlocks."""
    
    # Application settings
    app_name: str = "LLMBlocks"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Security settings
    secret_key: str = Field(default_factory=lambda: os.urandom(32).hex())
    allowed_hosts: List[str] = Field(default_factory=lambda: ["*"])
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    
    # Database settings
    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    
    # LLM settings
    default_llm_provider: str = "openai"
    llm_timeout: int = 30
    llm_max_retries: int = 3
    
    # Memory settings
    default_memory_provider: str = "in-memory"
    memory_ttl: int = 3600  # 1 hour
    
    # RAG settings
    default_embedding_model: str = "all-MiniLM-L6-v2"
    vector_store_type: str = "chroma"
    
    # Agent settings
    max_tool_calls: int = 10
    agent_timeout: int = 300  # 5 minutes
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    
    model_config = SettingsConfigDict(
        env_prefix="LLMBLOCKS_",
        case_sensitive=False,
        extra="allow"
    )


class ConfigManager:
    """
    Configuration manager for LLMBlocks.
    
    This class provides:
    - Configuration file loading (YAML/JSON)
    - Environment variable injection
    - Configuration validation and merging
    - Configuration hot-reloading
    - Configuration export and import
    """
    
    def __init__(
        self,
        config_paths: Optional[List[Union[str, Path]]] = None,
        env_prefix: str = "LLMBLOCKS_",
        auto_load: bool = True
    ):
        """
        Initialize the configuration manager.
        
        Args:
            config_paths: List of configuration file paths to load
            env_prefix: Environment variable prefix
            auto_load: Whether to automatically load configuration
        """
        self.logger = get_logger("ConfigManager")
        self.env_prefix = env_prefix
        
        # Configuration storage
        self._config: Dict[str, Any] = {}
        self._sources: List[ConfigSource] = []
        self._watched_files: Set[Path] = set()
        self._default_config = self._get_default_config()
        
        # Load configuration if auto_load is enabled
        if auto_load:
            self.load_configuration(config_paths or [])
        
        self.logger.info(
            "Configuration manager initialized",
            env_prefix=env_prefix,
            config_sources=len(self._sources)
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "app": {
                "name": "LLMBlocks",
                "version": "0.1.0",
                "debug": False
            },
            "logging": {
                "level": "INFO",
                "format": "json",
                "file": None
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 1
            },
            "security": {
                "secret_key": os.urandom(32).hex(),
                "allowed_hosts": ["*"],
                "cors_origins": ["*"]
            },
            "llm": {
                "default_provider": "openai",
                "timeout": 30,
                "max_retries": 3
            },
            "memory": {
                "default_provider": "in-memory",
                "ttl": 3600
            },
            "rag": {
                "default_embedding_model": "all-MiniLM-L6-v2",
                "vector_store_type": "chroma"
            },
            "agent": {
                "max_tool_calls": 10,
                "timeout": 300
            }
        }
    
    def load_configuration(
        self,
        config_paths: Optional[List[Union[str, Path]]] = None
    ) -> None:
        """
        Load configuration from multiple sources.
        
        Args:
            config_paths: List of configuration file paths
        """
        # Clear existing configuration
        self._config.clear()
        self._sources.clear()
        
        # Load default configuration first
        self._load_default_config()
        
        # Load configuration files
        if config_paths:
            for config_path in config_paths:
                self.load_config_file(config_path)
        
        # Load environment variables
        self._load_environment_config()
        
        # Validate final configuration
        self._validate_configuration()
        
        self.logger.info(
            "Configuration loaded",
            sources=len(self._sources),
            config_keys=list(self._config.keys())
        )
    
    def _load_default_config(self) -> None:
        """Load default configuration."""
        self._config.update(self._default_config)
        self._sources.append(ConfigSource(
            source_type="default",
            priority=0,
            loaded_at=datetime.now(UTC),
            is_valid=True
        ))
    
    def load_config_file(self, config_path: Union[str, Path]) -> bool:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if file was loaded successfully, False otherwise
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            self.logger.warning(
                "Configuration file not found",
                path=str(config_path)
            )
            return False
        
        try:
            # Determine file type and load
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
            else:
                self.logger.warning(
                    "Unsupported configuration file format",
                    path=str(config_path),
                    suffix=config_path.suffix
                )
                return False
            
            # Merge configuration
            self._merge_config(file_config, priority=10)
            
            # Add to sources
            self._sources.append(ConfigSource(
                source_type="file",
                path=str(config_path),
                priority=10,
                loaded_at=datetime.now(UTC),
                is_valid=True
            ))
            
            # Watch file for changes
            self._watch_file(config_path)
            
            self.logger.info(
                "Configuration file loaded",
                path=str(config_path),
                config_keys=list(file_config.keys())
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to load configuration file",
                path=str(config_path),
                error=str(e)
            )
            
            # Add failed source
            self._sources.append(ConfigSource(
                source_type="file",
                path=str(config_path),
                priority=10,
                loaded_at=datetime.now(UTC),
                is_valid=False,
                error_message=str(e)
            ))
            
            return False
    
    def _load_environment_config(self) -> None:
        """Load configuration from environment variables."""
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # Remove prefix and convert to nested structure
                config_key = key[len(self.env_prefix):].lower()
                nested_keys = config_key.split('_')
                
                # Build nested structure
                current = env_config
                for nested_key in nested_keys[:-1]:
                    if nested_key not in current:
                        current[nested_key] = {}
                    current = current[nested_key]
                
                # Set final value
                current[nested_keys[-1]] = self._parse_env_value(value)
        
        if env_config:
            # Merge environment configuration with high priority
            self._merge_config(env_config, priority=20)
            
            self._sources.append(ConfigSource(
                source_type="environment",
                priority=20,
                loaded_at=datetime.now(UTC),
                is_valid=True
            ))
            
            self.logger.info(
                "Environment configuration loaded",
                env_keys=list(env_config.keys())
            )
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try to parse as JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try to parse as boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Try to parse as integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try to parse as float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _merge_config(self, new_config: Dict[str, Any], priority: int) -> None:
        """
        Merge new configuration with existing configuration.
        
        Args:
            new_config: New configuration to merge
            priority: Priority of this configuration source
        """
        def merge_dict(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    merge_dict(target[key], value)
                else:
                    target[key] = value
        
        merge_dict(self._config, new_config)
    
    def _validate_configuration(self) -> None:
        """Validate the final configuration."""
        try:
            # Validate using Pydantic model
            validated_config = LLMBlocksConfig(**self._config)
            
            # Update config with validated values
            self._config = validated_config.model_dump()
            
        except ValidationError as e:
            self.logger.error(
                "Configuration validation failed",
                errors=str(e)
            )
            raise InvalidConfigurationError(f"Configuration validation failed: {e}")
    
    def _watch_file(self, file_path: Path) -> None:
        """Watch a configuration file for changes."""
        # This is a placeholder for file watching functionality
        # In a real implementation, you might use watchdog or similar
        self._watched_files.add(file_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        current = self._config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        current = self._config
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the value
        current[keys[-1]] = value
        
        self.logger.debug(
            "Configuration value set",
            key=key,
            value=value
        )
    
    def has(self, key: str) -> bool:
        """
        Check if a configuration key exists.
        
        Args:
            key: Configuration key (supports dot notation)
            
        Returns:
            True if key exists, False otherwise
        """
        keys = key.split('.')
        current = self._config
        
        try:
            for k in keys:
                current = current[k]
            return True
        except (KeyError, TypeError):
            return False
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Configuration section as dictionary
        """
        return self.get(section, {})
    
    def reload(self) -> None:
        """Reload configuration from all sources."""
        self.logger.info("Reloading configuration")
        
        # Get current config paths
        config_paths = [
            source.path for source in self._sources
            if source.source_type == "file" and source.path
        ]
        
        # Reload configuration
        self.load_configuration(config_paths)
    
    def export_config(self, format: str = "yaml") -> str:
        """
        Export current configuration.
        
        Args:
            format: Export format ('yaml' or 'json')
            
        Returns:
            Configuration as string
        """
        if format.lower() == "yaml":
            return yaml.dump(self._config, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            return json.dumps(self._config, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def save_config(self, file_path: Union[str, Path], format: str = "yaml") -> bool:
        """
        Save current configuration to a file.
        
        Args:
            file_path: Path to save configuration
            format: File format ('yaml' or 'json')
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "yaml":
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self._config, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(
                "Configuration saved",
                path=str(file_path),
                format=format
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to save configuration",
                path=str(file_path),
                error=str(e)
            )
            return False
    
    def get_sources(self) -> List[ConfigSource]:
        """Get list of configuration sources."""
        return self._sources.copy()
    
    def get_source_info(self, source_type: str) -> List[ConfigSource]:
        """Get configuration sources of a specific type."""
        return [source for source in self._sources if source.source_type == source_type]
    
    def validate_config_file(self, file_path: Union[str, Path]) -> bool:
        """
        Validate a configuration file without loading it.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return False
            
            # Try to load and validate
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                return False
            
            # Validate using Pydantic model
            LLMBlocksConfig(**config)
            return True
            
        except Exception:
            return False
    
    def __repr__(self) -> str:
        """String representation of the configuration manager."""
        return (
            f"ConfigManager("
            f"sources={len(self._sources)}, "
            f"config_keys={list(self._config.keys())})"
        )


# Global configuration manager instance
_global_config: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _global_config
    
    if _global_config is None:
        _global_config = ConfigManager()
    
    return _global_config


def set_config(config: ConfigManager) -> None:
    """Set the global configuration manager instance."""
    global _global_config
    _global_config = config


def load_config_from_file(file_path: Union[str, Path]) -> bool:
    """Load configuration from a file using the global config manager."""
    config = get_config()
    return config.load_config_file(file_path)


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value using the global config manager."""
    config = get_config()
    return config.get(key, default)


def set_config_value(key: str, value: Any) -> None:
    """Set a configuration value using the global config manager."""
    config = get_config()
    config.set(key, value)
