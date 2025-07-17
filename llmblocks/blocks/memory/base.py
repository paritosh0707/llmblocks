from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseMemoryProvider(ABC):
    def __init__(self, session_id: Optional[str] = None, **kwargs):
        self.session_id = session_id or ""
        self._initialize(**kwargs)

    @abstractmethod
    def _initialize(self, **kwargs) -> None:
        """Initialize the memory provider with configuration."""
        pass

    @abstractmethod
    def add(self, message: Dict[str, Any]) -> None:
        """Add a message to memory."""
        pass

    @abstractmethod
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages from memory."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all messages from memory."""
        pass
