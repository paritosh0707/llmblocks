"""
Configuration loader for LLMBlocks.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os

from .base_component import ComponentConfig
from ..blocks.rag import RAGConfig


class ConfigLoader:
    """Configuration loader for LLMBlocks components."""
    
    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration from {file_path}: {e}")
    
    @staticmethod
    def load_rag_config(file_path: str) -> RAGConfig:
        """Load RAG configuration from YAML file."""
        config_data = ConfigLoader.load_yaml(file_path)
        
        # Extract RAG-specific configuration
        rag_config = config_data.get('rag', {})
        
        # Create RAGConfig with loaded values
        config = RAGConfig(
            name=rag_config.get('name', 'rag_pipeline'),
            description=rag_config.get('description'),
            chunk_size=rag_config.get('chunk_size', 1000),
            chunk_overlap=rag_config.get('chunk_overlap', 200),
            text_splitter_type=rag_config.get('text_splitter_type', 'recursive'),
            vector_store_type=rag_config.get('vector_store_type', 'chroma'),
            embedding_model=rag_config.get('embedding_model', 'text-embedding-ada-002'),
            persist_directory=rag_config.get('persist_directory'),
            llm_provider=rag_config.get('llm_provider', 'openai'),
            llm_model=rag_config.get('llm_model', 'gpt-3.5-turbo'),
            llm_api_key=rag_config.get('llm_api_key'),
            llm_base_url=rag_config.get('llm_base_url'),
            llm_task=rag_config.get('llm_task'),
            temperature=rag_config.get('temperature', 0.0),
            max_tokens=rag_config.get('max_tokens', 1000),
            top_k=rag_config.get('top_k', 4),
            similarity_threshold=rag_config.get('similarity_threshold', 0.7),
            memory_enabled=rag_config.get('memory_enabled', False),
            memory_type=rag_config.get('memory_type', 'in_memory'),
            streaming_enabled=rag_config.get('streaming_enabled', False)
        )
        
        # Add any additional metadata
        if 'metadata' in rag_config:
            config.metadata.update(rag_config['metadata'])
        
        return config
    
    @staticmethod
    def load_agent_config(file_path: str) -> ComponentConfig:
        """Load agent configuration from YAML file."""
        config_data = ConfigLoader.load_yaml(file_path)
        
        # Extract agent-specific configuration
        agent_config = config_data.get('agent', {})
        
        # Create ComponentConfig with loaded values
        config = ComponentConfig(
            name=agent_config.get('name', 'agent'),
            description=agent_config.get('description'),
            enabled=agent_config.get('enabled', True)
        )
        
        # Add any additional metadata
        if 'metadata' in agent_config:
            config.metadata.update(agent_config['metadata'])
        
        return config
    
    @staticmethod
    def load_memory_config(file_path: str) -> ComponentConfig:
        """Load memory configuration from YAML file."""
        config_data = ConfigLoader.load_yaml(file_path)
        
        # Extract memory-specific configuration
        memory_config = config_data.get('memory', {})
        
        # Create ComponentConfig with loaded values
        config = ComponentConfig(
            name=memory_config.get('name', 'memory'),
            description=memory_config.get('description'),
            enabled=memory_config.get('enabled', True)
        )
        
        # Add any additional metadata
        if 'metadata' in memory_config:
            config.metadata.update(memory_config['metadata'])
        
        return config
    
    @staticmethod
    def load_tools_config(file_path: str) -> ComponentConfig:
        """Load tools configuration from YAML file."""
        config_data = ConfigLoader.load_yaml(file_path)
        
        # Extract tools-specific configuration
        tools_config = config_data.get('tools', {})
        
        # Create ComponentConfig with loaded values
        config = ComponentConfig(
            name=tools_config.get('name', 'tools'),
            description=tools_config.get('description'),
            enabled=tools_config.get('enabled', True)
        )
        
        # Add any additional metadata
        if 'metadata' in tools_config:
            config.metadata.update(tools_config['metadata'])
        
        return config
    
    @staticmethod
    def save_config(config: ComponentConfig, file_path: str) -> None:
        """Save configuration to YAML file."""
        path = Path(file_path)
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dictionary
        config_dict = {
            'name': config.name,
            'description': config.description,
            'enabled': config.enabled,
            'metadata': config.metadata
        }
        
        # Add specific fields for RAG config
        if isinstance(config, RAGConfig):
            config_dict.update({
                'chunk_size': config.chunk_size,
                'chunk_overlap': config.chunk_overlap,
                'text_splitter_type': config.text_splitter_type,
                'vector_store_type': config.vector_store_type,
                'embedding_model': config.embedding_model,
                'persist_directory': config.persist_directory,
                'llm_provider': config.llm_provider,
                'llm_model': config.llm_model,
                'llm_api_key': config.llm_api_key,
                'llm_base_url': config.llm_base_url,
                'llm_task': config.llm_task,
                'temperature': config.temperature,
                'max_tokens': config.max_tokens,
                'top_k': config.top_k,
                'similarity_threshold': config.similarity_threshold,
                'memory_enabled': config.memory_enabled,
                'memory_type': config.memory_type,
                'streaming_enabled': config.streaming_enabled
            })
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise RuntimeError(f"Error saving configuration to {file_path}: {e}")
    
    @staticmethod
    def get_default_config_path() -> str:
        """Get the default configuration directory path."""
        # Try to get from environment variable
        config_dir = os.getenv('LLMBLOCKS_CONFIG_DIR')
        
        if config_dir:
            return config_dir
        
        # Default to current working directory
        return str(Path.cwd() / 'config')
    
    @staticmethod
    def list_config_files(config_dir: Optional[str] = None) -> list[str]:
        """List available configuration files in the config directory."""
        if config_dir is None:
            config_dir = ConfigLoader.get_default_config_path()
        
        path = Path(config_dir)
        
        if not path.exists():
            return []
        
        config_files = []
        for file_path in path.glob('*.yaml'):
            config_files.append(str(file_path))
        
        for file_path in path.glob('*.yml'):
            config_files.append(str(file_path))
        
        return sorted(config_files)