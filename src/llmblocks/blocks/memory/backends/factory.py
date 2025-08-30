"""
Backend factory for memory storage systems.
"""

from typing import Any, Dict, List, Type

from ..base import MemoryBackend, MemoryBackendError
from ....utils.logging import get_logger


class BackendFactory:
    """Factory for creating memory backends."""
    
    def __init__(self):
        self.logger = get_logger("BackendFactory")
        self._backends: Dict[str, Type[MemoryBackend]] = {}
        self._register_builtin_backends()
    
    def _register_builtin_backends(self):
        """Register built-in backends."""
        from .in_memory import InMemoryBackend
        self.register_backend("in_memory", InMemoryBackend)
        
        # Register other backends as they're implemented
        try:
            from .file_backend import FileBackend
            self.register_backend("file", FileBackend)
        except ImportError:
            pass
        
        try:
            from .redis_backend import RedisBackend
            self.register_backend("redis", RedisBackend)
        except ImportError:
            pass
    
    def register_backend(self, name: str, backend_class: Type[MemoryBackend]):
        """Register a backend type."""
        self._backends[name] = backend_class
        self.logger.debug(f"Registered backend: {name}")
    
    def list_backends(self) -> List[str]:
        """List available backends."""
        return list(self._backends.keys())
    
    async def create_backend(self, backend_type: str, config: Dict[str, Any]) -> MemoryBackend:
        """Create a backend instance."""
        if backend_type not in self._backends:
            available = ", ".join(self.list_backends())
            raise MemoryBackendError(
                f"Unknown backend type: {backend_type}. Available: {available}"
            )
        
        backend_class = self._backends[backend_type]
        return backend_class(config)


# Global factory instance
_backend_factory = BackendFactory()


async def get_backend(backend_type: str, config: Dict[str, Any]) -> MemoryBackend:
    """Get a backend instance."""
    return await _backend_factory.create_backend(backend_type, config)


def list_backends() -> List[str]:
    """List available backends."""
    return _backend_factory.list_backends()


def register_backend(name: str, backend_class: Type[MemoryBackend]):
    """Register a custom backend."""
    _backend_factory.register_backend(name, backend_class)
