from typing import Dict, Any, Optional, Type
from llmblocks.blocks.memory.base import BaseMemoryProvider
from llmblocks.blocks.memory.in_memory import InMemoryProvider

class MemoryFactory:
    _providers: Dict[str, Type[BaseMemoryProvider]] = {
        "in-memory": InMemoryProvider
    }

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseMemoryProvider]) -> None:
        """Register a new memory provider."""
        cls._providers[name] = provider_class

    @classmethod
    def create_provider(cls, provider_name: str = "in-memory", session_id: Optional[str] = None, **kwargs) -> BaseMemoryProvider:
        """Create a memory provider instance."""
        if provider_name not in cls._providers:
            raise ValueError(f"Unknown memory provider: {provider_name}")
        
        provider_class = cls._providers[provider_name]
        return provider_class(session_id=session_id, **kwargs)
