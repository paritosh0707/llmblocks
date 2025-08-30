"""
LLMBlocks - A modular, enterprise-grade AI application framework.

A block-based architecture for building sophisticated AI applications.
"""

__version__ = "0.1.0"
__author__ = "LLMBlocks Team"
__email__ = "team@llmblocks.dev"

# Core exports
from .core.base_block import BaseBlock
from .core.registry import BlockRegistry
from .core.config import ConfigManager

# Component exports
from .components.chatbot import Chatbot
from .components.rag import RAGSystem
from .components.agent import MultiToolAgent

# Utility exports
from .utils.logging import get_logger
from .utils.exceptions import LLMBlocksError

__all__ = [
    # Core
    "BaseBlock",
    "BlockRegistry", 
    "ConfigManager",
    
    # Components
    "Chatbot",
    "RAGSystem",
    "MultiToolAgent",
    
    # Utilities
    "get_logger",
    "LLMBlocksError",
    
    # Version
    "__version__",
]
